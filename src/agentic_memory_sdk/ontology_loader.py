from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel

from .models import DomainPackSpec, RelationRule, UpperConceptType


class DomainPackLoader:
    """
    Domain-agnostic loader that allows any team on the platform to define and load
    a Layer 1 `DomainPackSpec` from:
      1. JSON / Dict definitions
      2. Python Pydantic Models (auto-introspecting classes and fields)
      3. W3C OWL / RDFS / SHACL Turtle (`.ttl`) files (via `rdflib` if installed, or fallback parser)
    """

    @staticmethod
    def from_dict(data: dict[str, Any]) -> DomainPackSpec:
        """Loads a DomainPackSpec from a raw dictionary or parsed JSON/YAML payload."""
        relations: dict[str, RelationRule] = {}
        raw_rels = data.get("relations", {})
        if isinstance(raw_rels, list):
            for r in raw_rels:
                rule = RelationRule.model_validate(r)
                relations[rule.predicate] = rule
        elif isinstance(raw_rels, dict):
            for pred, r in raw_rels.items():
                if isinstance(r, RelationRule):
                    relations[pred] = r
                else:
                    relations[pred] = RelationRule.model_validate({**r, "predicate": pred})

        disjoint = [tuple(pair) for pair in data.get("disjoint_classes", [])]
        return DomainPackSpec(
            pack_id=data["pack_id"],
            domain_name=data.get("domain_name") or data["pack_id"].split("@")[0],
            version=data.get("version", "1.0.0"),
            classes=data.get("classes", {}),
            disjoint_classes=disjoint,
            relations=relations,
        )

    @staticmethod
    def from_json_file(path: str | Path) -> DomainPackSpec:
        """Loads a DomainPackSpec from a JSON file on disk."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return DomainPackLoader.from_dict(raw)

    @staticmethod
    def from_pydantic_models(
        pack_id: str,
        entity_models: list[ tuple[Type[BaseModel], UpperConceptType | str] ],
        relations: list[RelationRule] | None = None,
        disjoint_classes: list[tuple[str, str]] | None = None,
    ) -> DomainPackSpec:
        """
        Compiles standard Python Pydantic classes directly into a governed `DomainPackSpec`
        so application teams never need to write raw OWL/SHACL files.
        """
        domain_name = pack_id.split("@")[0]
        version = pack_id.split("@")[1] if "@" in pack_id else "1.0.0"
        classes: dict[str, str] = {}
        for model_cls, parent in entity_models:
            parent_str = parent.value if isinstance(parent, UpperConceptType) else str(parent)
            classes[model_cls.__name__] = parent_str

        rel_map = {r.predicate: r for r in (relations or [])}
        return DomainPackSpec(
            pack_id=pack_id,
            domain_name=domain_name,
            version=version,
            classes=classes,
            disjoint_classes=disjoint_classes or [],
            relations=rel_map,
        )

    @staticmethod
    def from_turtle(
        pack_id: str,
        turtle_content: str,
    ) -> DomainPackSpec:
        """
        Compiles an OWL/RDFS Turtle (`.ttl`) string into a runtime `DomainPackSpec`.
        Uses `rdflib` when available, with a lightweight regex fallback for zero-dependency environments.
        """
        domain_name = pack_id.split("@")[0]
        version = pack_id.split("@")[1] if "@" in pack_id else "1.0.0"
        classes: dict[str, str] = {}
        disjoint: list[tuple[str, str]] = []
        relations: dict[str, RelationRule] = {}

        try:
            import rdflib
            from rdflib.namespace import OWL, RDF, RDFS

            g = rdflib.Graph()
            g.parse(data=turtle_content, format="turtle")

            def _local_name(uri: Any) -> str:
                s = str(uri)
                if "upper#" in s:
                    return f"p:{s.split('#')[-1]}"
                if "#" in s:
                    return s.split("#")[-1]
                return s.rsplit("/", 1)[-1]

            for cls_uri in g.subjects(RDF.type, OWL.Class):
                cls_name = _local_name(cls_uri)
                parent_name = UpperConceptType.CONCEPT.value
                for parent_uri in g.objects(cls_uri, RDFS.subClassOf):
                    parent_name = _local_name(parent_uri)
                classes[cls_name] = parent_name

                for disj_uri in g.objects(cls_uri, OWL.disjointWith):
                    disjoint.append((cls_name, _local_name(disj_uri)))

            for prop_uri in g.subjects(RDF.type, OWL.ObjectProperty):
                label_objs = list(g.objects(prop_uri, RDFS.label))
                pred_name = str(label_objs[0]) if label_objs else _local_name(prop_uri)
                dom_objs = list(g.objects(prop_uri, RDFS.domain))
                rng_objs = list(g.objects(prop_uri, RDFS.range))
                is_func = (prop_uri, RDF.type, OWL.FunctionalProperty) in g
                is_trans = (prop_uri, RDF.type, OWL.TransitiveProperty) in g
                relations[pred_name] = RelationRule(
                    predicate=pred_name,
                    domain_class=_local_name(dom_objs[0]) if dom_objs else "owl:Thing",
                    range_class=_local_name(rng_objs[0]) if rng_objs else "owl:Thing",
                    is_functional=is_func,
                    is_transitive=is_trans,
                )

            return DomainPackSpec(
                pack_id=pack_id,
                domain_name=domain_name,
                version=version,
                classes=classes,
                disjoint_classes=disjoint,
                relations=relations,
            )
        except ImportError:
            # Lightweight fallback parser for Turtle syntax when rdflib is not installed
            import re

            for match in re.finditer(
                r"(?:\w+:)?(\w+)\s+a\s+owl:Class\s*;([^.]*)\.", turtle_content, re.DOTALL
            ):
                cls_name = match.group(1)
                body = match.group(2)
                sub_match = re.search(r"rdfs:subClassOf\s+((?:p:|\w+:)?\w+)", body)
                parent = sub_match.group(1) if sub_match else UpperConceptType.CONCEPT.value
                if not parent.startswith("p:") and ":" in parent:
                    parent = parent.split(":", 1)[1]
                classes[cls_name] = parent
                for d_match in re.finditer(r"owl:disjointWith\s+([^\n;.]+)", body):
                    for target in d_match.group(1).split(","):
                        t_clean = target.strip().split(":")[-1]
                        if t_clean:
                            disjoint.append((cls_name, t_clean))

            for match in re.finditer(
                r"(?:\w+:)?(\w+)\s+a\s+([^;]+);([^.]*)\.", turtle_content, re.DOTALL
            ):
                prop_id, types_str, body = match.group(1), match.group(2), match.group(3)
                if "owl:ObjectProperty" not in types_str:
                    continue
                lbl_match = re.search(r'rdfs:label\s+"([^"]+)"', body)
                pred_name = lbl_match.group(1) if lbl_match else prop_id
                dom_match = re.search(r"rdfs:domain\s+(?:\w+:)?(\w+)", body)
                rng_match = re.search(r"rdfs:range\s+(?:\w+:)?(\w+)", body)
                relations[pred_name] = RelationRule(
                    predicate=pred_name,
                    domain_class=dom_match.group(1) if dom_match else "owl:Thing",
                    range_class=rng_match.group(1) if rng_match else "owl:Thing",
                    is_functional="owl:FunctionalProperty" in types_str,
                    is_transitive="owl:TransitiveProperty" in types_str,
                )

            return DomainPackSpec(
                pack_id=pack_id,
                domain_name=domain_name,
                version=version,
                classes=classes,
                disjoint_classes=disjoint,
                relations=relations,
            )
