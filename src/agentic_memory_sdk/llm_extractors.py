from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from typing import Any

import httpx

from .auth import AwsIamDatabaseAuthProvider, CloudAuthProvider
from .models import (
    CompiledTBox,
    ExtractedEntity,
    ExtractedFact,
    ExtractionPayload,
)
from .ports import EmbeddingAdapter, OntologyExtractorAdapter


def _align_vector_dim(vec: list[float], target_dim: int | None) -> list[float]:
    """
    Optionally pads (with zeros) or truncates + L2-renormalizes an embedding vector
    when a model's native dimension (e.g. 768 or 1024) needs to match a fixed DB column
    dimension (e.g. 1536), or returns `vec` unchanged when `target_dim` is None or matches.
    """
    if not target_dim or len(vec) == target_dim:
        return vec
    if len(vec) > target_dim:
        sliced = vec[:target_dim]
        norm = math.sqrt(sum(x * x for x in sliced)) or 1.0
        return [x / norm for x in sliced]
    return vec + [0.0] * (target_dim - len(vec))


def _extract_json_object(raw_text: str) -> str:
    """Strips markdown code fences if an LLM wraps JSON output in ```json ... ```."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


# =============================================================================
# 1. Google Gemini / Vertex AI Adapters
# =============================================================================


class GeminiEmbeddingAdapter(EmbeddingAdapter):
    """Google GenAI (Gemini) embedding adapter using `google-genai` SDK."""

    def __init__(
        self,
        model: str = "text-embedding-004",
        api_key: str | None = None,
        output_dimensionality: int = 1536,
    ):
        self.model = model
        self.api_key = api_key
        self.output_dimensionality = output_dimensionality
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = (
                genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
            )
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        resp = await asyncio.to_thread(
            client.models.embed_content,
            model=self.model,
            contents=text,
        )
        return _align_vector_dim(
            list(resp.embeddings[0].values), self.output_dimensionality
        )


class GeminiOntologyExtractorAdapter(OntologyExtractorAdapter):
    """
    Compiles the active app's `CompiledTBox` into a structured prompt & Pydantic response schema
    for Google Gemini (`google-genai`), supporting the Two-Bucket extraction pattern.
    """

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        self.model = model
        self.api_key = api_key
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = (
                genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
            )
        return self._client

    @staticmethod
    def build_tbox_system_instruction(tbox: CompiledTBox) -> str:
        allowed_classes = ", ".join(sorted(tbox.classes.keys()))
        relation_lines = [
            f"- {r.predicate}: domain={r.domain_class}, range={r.range_class}, functional={r.is_functional}"
            for r in tbox.relations.values()
        ]
        relations_block = "\n".join(relation_lines)
        return (
            "You are an Ontology-Guided Knowledge Graph Extractor & Smart Memory Classifier.\n"
            "1. First, evaluate if the transcript is purely ephemeral social chitchat, pleasantries, or terminal sign-offs "
            "(e.g., 'Have a great weekend everyone, talk Monday!'). If so, set `is_ephemeral_chitchat=True` and return empty lists.\n"
            "2. Pay close attention to conversational context: if an agent asks a question ('Did checkout-api fail over to us-west-2?') "
            "and the user replies briefly ('Yes' or 'Confirmed'), treat that as a confirmed fact and extract it!\n"
            "3. For each extracted entity and fact, assign `scope`:\n"
            "   - 'user_global' for personal user preferences, communication style, role, or timezone.\n"
            "   - 'domain_shared' for domain/system/customer facts conforming to the TBox.\n"
            "   - 'app_private' for app-specific workflow state or candidate concepts.\n"
            f"ALLOWED TBOX CLASSES: [{allowed_classes}]\n"
            f"ALLOWED TBOX RELATIONS:\n{relations_block}\n"
            "If an important entity does not fit a specific domain class, classify it as 'p:CandidateConcept'."
        )

    async def extract(self, transcript: str, tbox: CompiledTBox) -> ExtractionPayload:
        client = self._get_client()
        from google.genai import types

        sys_instruction = self.build_tbox_system_instruction(tbox)
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model=self.model,
            contents=transcript,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                response_mime_type="application/json",
                response_schema=ExtractionPayload,
                temperature=0.0,
            ),
        )
        if getattr(resp, "parsed", None) and isinstance(resp.parsed, ExtractionPayload):
            return resp.parsed
        return ExtractionPayload.model_validate_json(_extract_json_object(resp.text))


# =============================================================================
# 2. Amazon Bedrock Adapters (Titan / Cohere Embeddings + Claude / Nova Converse)
# =============================================================================


class BedrockEmbeddingAdapter(EmbeddingAdapter):
    """
    Enterprise Amazon Bedrock Embedding Adapter (`bedrock-runtime`) supporting:
      - Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`, native dims 256/512/1024)
      - Amazon Titan Embeddings G1 (`amazon.titan-embed-text-v1`, native dim 1536)
      - Cohere Embed v3 on Bedrock (`cohere.embed-english-v3` / `cohere.embed-multilingual-v3`)
    Supports cross-account IAM AssumeRole via `aws_auth_provider` and optional
    `target_dim` alignment (e.g. 1536) for fixed pgvector schemas.
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        dimensions: int = 1024,
        target_dim: int | None = None,
        normalize: bool = True,
        aws_auth_provider: AwsIamDatabaseAuthProvider | None = None,
        bedrock_client: Any = None,
    ):
        self.model_id = model_id
        self.region = region
        self.dimensions = dimensions
        self.target_dim = target_dim
        self.normalize = normalize
        self.aws_auth_provider = aws_auth_provider
        self._client = bedrock_client

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3

            session = (
                self.aws_auth_provider.get_boto3_session()
                if self.aws_auth_provider is not None
                else boto3.Session(region_name=self.region)
            )
            self._client = session.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _invoke_sync(self, text: str) -> list[float]:
        client = self._get_client()
        if "cohere" in self.model_id.lower():
            body = json.dumps(
                {
                    "texts": [text],
                    "input_type": "search_document",
                    "truncate": "END",
                }
            )
        elif "titan-embed-text-v2" in self.model_id.lower():
            body = json.dumps(
                {
                    "inputText": text,
                    "dimensions": self.dimensions,
                    "normalize": self.normalize,
                }
            )
        else:
            # amazon.titan-embed-text-v1 (1536 dims)
            body = json.dumps({"inputText": text})

        resp = client.invoke_model(
            modelId=self.model_id,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        raw_body = resp["body"].read()
        data = json.loads(raw_body.decode("utf-8") if isinstance(raw_body, bytes) else raw_body)
        if "embedding" in data:
            vec = list(data["embedding"])
        elif "embeddings" in data and data["embeddings"]:
            vec = list(data["embeddings"][0])
        else:
            raise ValueError(f"Unexpected Bedrock embedding response format: {list(data.keys())}")
        return _align_vector_dim(vec, self.target_dim)

    async def embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._invoke_sync, text)


class BedrockOntologyExtractorAdapter(OntologyExtractorAdapter):
    """
    Enterprise Amazon Bedrock Ontology Extractor (`bedrock-runtime` Converse API).
    Works seamlessly across:
      - Anthropic Claude 3.5 / 3.7 / 4 Sonnet & Haiku (`us.anthropic.claude-3-5-sonnet-20241022-v2:0`)
      - Amazon Nova Pro / Lite (`us.amazon.nova-pro-v1:0`, `us.amazon.nova-lite-v1:0`)
      - Meta Llama 3.3 Instruct (`us.meta.llama3-3-70b-instruct-v1:0`)
    Uses Bedrock `converse` tool-use structured output (`toolConfig`) with automatic
    fallback to JSON text parsing.
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
        aws_auth_provider: AwsIamDatabaseAuthProvider | None = None,
        bedrock_client: Any = None,
    ):
        self.model_id = model_id
        self.region = region
        self.aws_auth_provider = aws_auth_provider
        self._client = bedrock_client

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3

            session = (
                self.aws_auth_provider.get_boto3_session()
                if self.aws_auth_provider is not None
                else boto3.Session(region_name=self.region)
            )
            self._client = session.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _converse_sync(self, transcript: str, tbox: CompiledTBox) -> ExtractionPayload:
        client = self._get_client()
        sys_instruction = GeminiOntologyExtractorAdapter.build_tbox_system_instruction(tbox)
        schema = ExtractionPayload.model_json_schema()

        resp = client.converse(
            modelId=self.model_id,
            system=[{"text": sys_instruction}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": transcript}],
                }
            ],
            inferenceConfig={"temperature": 0.0},
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": "emit_extraction_payload",
                            "description": "Emit structured ontology entities and bi-temporal facts.",
                            "inputSchema": {"json": schema},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": "emit_extraction_payload"}},
            },
        )
        content_blocks = (
            resp.get("output", {}).get("message", {}).get("content", [])
        )
        for block in content_blocks:
            if "toolUse" in block and "input" in block["toolUse"]:
                return ExtractionPayload.model_validate(block["toolUse"]["input"])
            if "text" in block and block["text"].strip():
                return ExtractionPayload.model_validate_json(
                    _extract_json_object(block["text"])
                )
        return ExtractionPayload(entities=[], facts=[])

    async def extract(self, transcript: str, tbox: CompiledTBox) -> ExtractionPayload:
        return await asyncio.to_thread(self._converse_sync, transcript, tbox)


# =============================================================================
# 3. OpenAI / Azure OpenAI / vLLM / LiteLLM (OpenAI-Compatible HTTP) Adapters
# =============================================================================


class OpenAICompatibleEmbeddingAdapter(EmbeddingAdapter):
    """
    OpenAI-compatible Embedding Adapter supporting:
      - OpenAI (`text-embedding-3-small`, `text-embedding-3-large` with native `dimensions=1536`)
      - Azure OpenAI Service (with API key or `AzureEntraDatabaseAuthProvider` Bearer token)
      - vLLM, LiteLLM Proxy, Together AI, Fireworks, or LM Studio (`/v1/embeddings`)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        dimensions: int | None = 1536,
        auth_provider: CloudAuthProvider | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions
        self.auth_provider = auth_provider
        self._http = http_client

    async def embed(self, text: str) -> list[float]:
        token = (
            await self.auth_provider.get_password()
            if self.auth_provider is not None
            else self.api_key
        )
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        payload: dict[str, Any] = {"model": self.model, "input": text}
        if self.dimensions is not None and "text-embedding-3" in self.model:
            payload["dimensions"] = self.dimensions

        if self._http is not None:
            resp = await self._http.post(
                f"{self.base_url}/embeddings", headers=headers, json=payload
            )
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/embeddings", headers=headers, json=payload
                )
        resp.raise_for_status()
        data = resp.json()
        vec = list(data["data"][0]["embedding"])
        return _align_vector_dim(vec, self.dimensions)


class OpenAICompatibleOntologyExtractorAdapter(OntologyExtractorAdapter):
    """
    OpenAI-compatible Ontology Extractor supporting OpenAI (`gpt-4o-mini`, `gpt-4o`),
    Azure OpenAI, vLLM, LiteLLM, or DeepSeek (`/v1/chat/completions`).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        auth_provider: CloudAuthProvider | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.auth_provider = auth_provider
        self._http = http_client

    async def extract(self, transcript: str, tbox: CompiledTBox) -> ExtractionPayload:
        token = (
            await self.auth_provider.get_password()
            if self.auth_provider is not None
            else self.api_key
        )
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        sys_instruction = (
            GeminiOntologyExtractorAdapter.build_tbox_system_instruction(tbox)
            + "\nRespond strictly with a JSON object matching fields: "
            "`is_ephemeral_chitchat` (bool), `classification_reason` (str), "
            "`entities` (list), `facts` (list)."
        )
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": sys_instruction},
                {"role": "user", "content": transcript},
            ],
        }
        if self._http is not None:
            resp = await self._http.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
        else:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return ExtractionPayload.model_validate_json(_extract_json_object(content))


# =============================================================================
# 4. Ollama Local Neural Adapters (Embeddings & Structured Extraction)
# =============================================================================


class OllamaEmbeddingAdapter(EmbeddingAdapter):
    """
    Local Ollama Embedding Adapter (`http://127.0.0.1:11434/api/embed`) for models such as:
      - `nomic-embed-text` (768 dims)
      - `mxbai-embed-large` (1024 dims)
      - `bge-m3` (1024 dims)
      - `all-minilm` (384 dims)
    Supports optional `target_dim` alignment (e.g. `target_dim=1536`) so local models work
    seamlessly against fixed 1536-dim PostgreSQL schemas without altering DDL.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        target_dim: int | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.target_dim = target_dim
        self._http = http_client

    async def embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "input": text}
        if self._http is not None:
            resp = await self._http.post(f"{self.base_url}/api/embed", json=payload)
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{self.base_url}/api/embed", json=payload)
        resp.raise_for_status()
        data = resp.json()
        vec = (
            list(data["embeddings"][0])
            if "embeddings" in data
            else list(data["embedding"])
        )
        return _align_vector_dim(vec, self.target_dim)


class OllamaOntologyExtractorAdapter(OntologyExtractorAdapter):
    """
    Local Ollama Structured Ontology Extractor (`http://127.0.0.1:11434/api/chat`)
    using Ollama's native JSON schema enforcement (`format=ExtractionPayload.model_json_schema()`)
    for local open-weight models (`qwen2.5`, `llama3.2`, `mistral`, `gemma2`).
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://127.0.0.1:11434",
        http_client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._http = http_client

    async def extract(self, transcript: str, tbox: CompiledTBox) -> ExtractionPayload:
        sys_instruction = GeminiOntologyExtractorAdapter.build_tbox_system_instruction(tbox)
        payload = {
            "model": self.model,
            "stream": False,
            "format": ExtractionPayload.model_json_schema(),
            "options": {"temperature": 0.0},
            "messages": [
                {"role": "system", "content": sys_instruction},
                {"role": "user", "content": transcript},
            ],
        }
        if self._http is not None:
            resp = await self._http.post(f"{self.base_url}/api/chat", json=payload)
        else:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return ExtractionPayload.model_validate_json(_extract_json_object(content))


# =============================================================================
# 5. Local Sentence-Transformers / FastEmbed In-Process Adapter
# =============================================================================


class LocalSentenceTransformerEmbeddingAdapter(EmbeddingAdapter):
    """
    In-process local neural embedding adapter supporting `sentence-transformers` or `fastembed`
    (e.g., `BAAI/bge-small-en-v1.5`, `all-MiniLM-L6-v2`) with optional `target_dim` alignment.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        target_dim: int | None = None,
    ):
        self.model_name = model_name
        self.target_dim = target_dim
        self._model: Any = None
        self._backend: str = ""

    def _encode_sync(self, text: str) -> list[float]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(self.model_name)
                self._backend = "sentence_transformers"
            except ImportError:
                from fastembed import TextEmbedding  # type: ignore

                self._model = TextEmbedding(model_name=self.model_name)
                self._backend = "fastembed"

        if self._backend == "sentence_transformers":
            vec = self._model.encode(text, normalize_embeddings=True).tolist()
        else:
            vec = list(next(self._model.embed([text])))
        return _align_vector_dim([float(x) for x in vec], self.target_dim)

    async def embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._encode_sync, text)


# =============================================================================
# 6. Deterministic Offline / CI Adapters
# =============================================================================


class DeterministicHashEmbeddingAdapter(EmbeddingAdapter):
    """
    Zero-dependency deterministic n-gram feature hashing embedder for local testing, CI/CD,
    and offline development without API keys. Produces normalized unit vectors.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [
            t.strip(".,!?;:()[]\"'").lower()
            for t in text.split()
            if t.strip(".,!?;:()[]\"'")
        ]
        if not tokens:
            vec[0] = 1.0
            return vec
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class RuleBasedMockExtractorAdapter(OntologyExtractorAdapter):
    """
    Configurable mock extractor for unit tests and local simulations.
    Allows registering scripted `ExtractionPayload` responses or using rule patterns.
    """

    def __init__(self):
        self.queued_payloads: list[ExtractionPayload] = []
        self.keyword_rules: list[tuple[str, ExtractionPayload]] = []

    def enqueue(self, payload: ExtractionPayload) -> None:
        self.queued_payloads.append(payload)

    def add_keyword_rule(self, keyword: str, payload: ExtractionPayload) -> None:
        self.keyword_rules.append((keyword.lower(), payload))

    async def extract(self, transcript: str, tbox: CompiledTBox) -> ExtractionPayload:
        if self.queued_payloads:
            return self.queued_payloads.pop(0)
        lower_t = transcript.lower()
        for kw, payload in self.keyword_rules:
            if kw in lower_t:
                return payload
        return ExtractionPayload(entities=[], facts=[])
