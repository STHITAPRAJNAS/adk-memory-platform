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
        version = data.get("version", "1.0.0")
        raw_domain = data.get("domain_name") or data.get("domain_id") or "custom_domain"
        pack_id = data.get("pack_id") or f"{raw_domain}@{version}"
        domain_name = data.get("domain_name") or pack_id.split("@")[0]
        return DomainPackSpec(
            pack_id=pack_id,
            domain_name=domain_name,
            version=version,
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
    def from_turtle_file(pack_id: str, path: str | Path) -> DomainPackSpec:
        """Loads and compiles an OWL/RDFS Turtle (`.ttl`) file on disk into a `DomainPackSpec`."""
        turtle_content = Path(path).read_text(encoding="utf-8")
        return DomainPackLoader.from_turtle(pack_id, turtle_content)

    @staticmethod
    def from_turtle_str(pack_id: str, turtle_content: str) -> DomainPackSpec:
        """Alias for `from_turtle`."""
        return DomainPackLoader.from_turtle(pack_id, turtle_content)

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

    @staticmethod
    def from_sql_ddl(
        pack_id: str,
        sql_ddl: str,
    ) -> DomainPackSpec:
        """
        Compiles standard relational SQL DDL (`CREATE TABLE ...`, inline `REFERENCES`,
        table-level `FOREIGN KEY (...) REFERENCES ...`, and `ALTER TABLE ... ADD FOREIGN KEY`)
        into a Layer-1 `DomainPackSpec` anchored under Layer 0 Upper Ontology classes.
        """
        import re

        from .governance import PlatformOntologyRegistrar

        domain_name = pack_id.split("@")[0]
        version = pack_id.split("@")[1] if "@" in pack_id else "1.0.0"

        def _table_to_class(raw_table: str) -> str:
            # Strip quotes, schema prefix (e.g. sales.customer_orders -> customer_orders)
            clean = raw_table.strip().strip('"`[]')
            if "." in clean:
                clean = clean.split(".")[-1].strip('"`[]')
            return PlatformOntologyRegistrar.canonicalize_class_name(clean)

        classes: dict[str, str] = {}
        relations: dict[str, RelationRule] = {}
        synonyms: dict[str, list[str]] = {}
        table_fks: dict[str, list[tuple[str, str, bool]]] = {}

        # Remove SQL line comments
        cleaned_sql = re.sub(r"--[^\n]*", "", sql_ddl)

        # Match CREATE TABLE [IF NOT EXISTS] <table_name> ( <body> )
        create_pat = re.compile(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_.\"`\[\]]+)\s*\((.*?)\)\s*;",
            re.IGNORECASE | re.DOTALL,
        )
        for m in create_pat.finditer(cleaned_sql + ";"):
            raw_tbl = m.group(1)
            body = m.group(2)
            src_cls = _table_to_class(raw_tbl)
            classes[src_cls] = PlatformOntologyRegistrar.infer_upper_concept_parent(
                src_cls
            )
            raw_short = raw_tbl.split(".")[-1].strip('"`[]')
            if raw_short.lower() != src_cls.lower():
                synonyms.setdefault(src_cls, []).append(raw_short)
            table_fks.setdefault(src_cls, [])

            # 1. Table-level FOREIGN KEY (col) REFERENCES target_tbl(target_col)
            for fk_m in re.finditer(
                r"FOREIGN\s+KEY\s*\(\s*([A-Za-z0-9_\"`]+)\s*\)\s*REFERENCES\s+([A-Za-z0-9_.\"`\[\]]+)",
                body,
                re.IGNORECASE,
            ):
                col_name = fk_m.group(1).strip('"`')
                tgt_cls = _table_to_class(fk_m.group(2))
                table_fks[src_cls].append((col_name, tgt_cls, True))

            # 2. Inline column definitions with REFERENCES target_tbl
            for line in body.split(","):
                line_s = line.strip()
                if not line_s or re.match(
                    r"^(?:PRIMARY|FOREIGN|CONSTRAINT|UNIQUE|CHECK|INDEX|KEY)\b",
                    line_s,
                    re.IGNORECASE,
                ):
                    continue
                inl_m = re.match(
                    r"^([A-Za-z0-9_\"`]+)\s+.*?REFERENCES\s+([A-Za-z0-9_.\"`\[\]]+)",
                    line_s,
                    re.IGNORECASE,
                )
                if inl_m:
                    col_name = inl_m.group(1).strip('"`')
                    tgt_cls = _table_to_class(inl_m.group(2))
                    table_fks[src_cls].append((col_name, tgt_cls, True))

        # Match ALTER TABLE <src> ADD ... FOREIGN KEY (<col>) REFERENCES <tgt>
        for alt_m in re.finditer(
            r"ALTER\s+TABLE\s+(?:ONLY\s+)?([A-Za-z0-9_.\"`\[\]]+)\s+ADD\s+(?:CONSTRAINT\s+[A-Za-z0-9_\"`]+\s+)?FOREIGN\s+KEY\s*\(\s*([A-Za-z0-9_\"`]+)\s*\)\s*REFERENCES\s+([A-Za-z0-9_.\"`\[\]]+)",
            cleaned_sql,
            re.IGNORECASE,
        ):
            src_cls = _table_to_class(alt_m.group(1))
            col_name = alt_m.group(2).strip('"`')
            tgt_cls = _table_to_class(alt_m.group(3))
            classes.setdefault(
                src_cls, PlatformOntologyRegistrar.infer_upper_concept_parent(src_cls)
            )
            table_fks.setdefault(src_cls, []).append((col_name, tgt_cls, True))

        for src_cls, fk_list in table_fks.items():
            for col_name, tgt_cls, is_single_fk in fk_list:
                if tgt_cls not in classes:
                    classes[tgt_cls] = (
                        PlatformOntologyRegistrar.infer_upper_concept_parent(tgt_cls)
                    )
                stem = re.sub(r"_(?:id|uuid|key|code|fk)$", "", col_name, flags=re.IGNORECASE)
                pred_raw = f"HAS_{stem}" if stem else f"HAS_{tgt_cls}"
                pred_norm = PlatformOntologyRegistrar.canonicalize_predicate_name(
                    pred_raw
                )
                lex_func, lex_trans = PlatformOntologyRegistrar.infer_relation_flags(
                    pred_norm
                )
                relations[pred_norm] = RelationRule(
                    predicate=pred_norm,
                    domain_class=src_cls,
                    range_class=tgt_cls,
                    is_functional=is_single_fk or lex_func,
                    is_transitive=lex_trans,
                    description=f"Induced from SQL foreign key {src_cls}.{col_name} -> {tgt_cls}",
                )

            # Bridge / junction table detection: if a table has >= 2 FKs (e.g., OrderLine linking Order and StockItem),
            # also synthesize a direct multi-valued relation between the first two referenced entity classes.
            unique_targets = list(dict.fromkeys(t for _, t, _ in fk_list if t != src_cls))
            if len(unique_targets) >= 2:
                c_left, c_right = unique_targets[0], unique_targets[1]
                bridge_pred = PlatformOntologyRegistrar.canonicalize_predicate_name(
                    f"INCLUDES_{c_right}"
                )
                if bridge_pred not in relations:
                    relations[bridge_pred] = RelationRule(
                        predicate=bridge_pred,
                        domain_class=c_left,
                        range_class=c_right,
                        is_functional=False,
                        is_transitive=False,
                        description=f"Many-to-many join relation induced via bridge table {src_cls}",
                    )

        return DomainPackSpec(
            pack_id=pack_id,
            domain_name=domain_name,
            version=version,
            classes=classes,
            relations=relations,
            synonyms=synonyms,
        )

    @staticmethod
    def from_ossie_yaml(
        pack_id: str,
        ossie_content: str | dict[str, Any],
    ) -> DomainPackSpec:
        """
        Compiles an Apache Ossie / BI Semantic Model definition (YAML string, JSON string, or dict)
        containing `terms` (or `entities`), `synonyms`, `relationships` (or `joins`), and `metrics`
        (`sql_attributes`) into a Layer-1 `DomainPackSpec`.
        """
        from .governance import PlatformOntologyRegistrar

        domain_name = pack_id.split("@")[0]
        version = pack_id.split("@")[1] if "@" in pack_id else "1.0.0"

        parsed: dict[str, Any] = {}
        if isinstance(ossie_content, dict):
            parsed = ossie_content
        else:
            text = ossie_content.strip()
            if text.startswith("{"):
                parsed = json.loads(text)
            else:
                try:
                    import yaml  # type: ignore

                    parsed = yaml.safe_load(text) or {}
                except ImportError:
                    parsed = DomainPackLoader._minimal_ossie_yaml_parse(text)

        classes: dict[str, str] = {}
        relations: dict[str, RelationRule] = {}
        synonyms: dict[str, list[str]] = {}

        raw_terms = (
            parsed.get("terms")
            or parsed.get("entities")
            or parsed.get("SemanticModel", {}).get("terms")
            or []
        )
        for item in raw_terms:
            if isinstance(item, str):
                c_name = PlatformOntologyRegistrar.canonicalize_class_name(item)
                classes[c_name] = PlatformOntologyRegistrar.infer_upper_concept_parent(
                    c_name
                )
                continue
            if not isinstance(item, dict):
                continue
            raw_name = item.get("name") or item.get("term") or item.get("id") or ""
            if not raw_name:
                continue
            c_name = PlatformOntologyRegistrar.canonicalize_class_name(str(raw_name))
            parent = item.get("parent_class") or PlatformOntologyRegistrar.infer_upper_concept_parent(
                c_name
            )
            classes[c_name] = str(parent)
            syns = [str(s) for s in (item.get("synonyms") or []) if s]
            if str(raw_name) != c_name and str(raw_name) not in syns:
                syns.append(str(raw_name))
            for metric in item.get("sql_attributes") or item.get("metrics") or []:
                m_name = metric.get("name") if isinstance(metric, dict) else str(metric)
                if m_name:
                    syns.append(str(m_name))
            if syns:
                synonyms[c_name] = syns

        raw_rels = (
            parsed.get("relationships")
            or parsed.get("joins")
            or parsed.get("SemanticModel", {}).get("relationships")
            or []
        )
        for rel in raw_rels:
            if not isinstance(rel, dict):
                continue
            src_raw = rel.get("source") or rel.get("from_term") or rel.get("left") or ""
            tgt_raw = rel.get("target") or rel.get("to_term") or rel.get("right") or ""
            if not src_raw or not tgt_raw:
                continue
            src_cls = PlatformOntologyRegistrar.canonicalize_class_name(str(src_raw))
            tgt_cls = PlatformOntologyRegistrar.canonicalize_class_name(str(tgt_raw))
            for c_ref in (src_cls, tgt_cls):
                if c_ref not in classes:
                    classes[c_ref] = PlatformOntologyRegistrar.infer_upper_concept_parent(
                        c_ref
                    )
            raw_pred = (
                rel.get("name")
                or rel.get("predicate")
                or rel.get("relationship")
                or f"JOINED_TO_{tgt_cls}"
            )
            pred_norm = PlatformOntologyRegistrar.canonicalize_predicate_name(
                str(raw_pred)
            )
            card = str(rel.get("cardinality") or rel.get("type") or "").lower()
            lex_func, lex_trans = PlatformOntologyRegistrar.infer_relation_flags(
                pred_norm
            )
            is_func = (
                bool(rel["is_functional"])
                if "is_functional" in rel
                else (card in ("many_to_one", "n:1", "1:1", "one_to_one") or lex_func)
            )
            relations[pred_norm] = RelationRule(
                predicate=pred_norm,
                domain_class=src_cls,
                range_class=tgt_cls,
                is_functional=is_func,
                is_transitive=bool(rel.get("is_transitive", lex_trans)),
                description=str(rel.get("description") or f"Imported from Ossie relationship {src_cls} -> {tgt_cls}"),
            )

        return DomainPackSpec(
            pack_id=pack_id,
            domain_name=domain_name,
            version=version,
            classes=classes,
            relations=relations,
            synonyms=synonyms,
            certified=bool(parsed.get("certified", False)),
        )

    @staticmethod
    def _minimal_ossie_yaml_parse(yaml_text: str) -> dict[str, Any]:
        """Zero-dependency fallback parser for simple Ossie YAML `terms:` and `relationships:` lists."""
        import re

        out: dict[str, list[dict[str, Any]]] = {"terms": [], "relationships": []}
        section: str | None = None
        current: dict[str, Any] | None = None
        for raw_line in yaml_text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped in ("terms:", "entities:"):
                section = "terms"
                current = None
                continue
            if stripped in ("relationships:", "joins:"):
                section = "relationships"
                current = None
                continue
            if section in ("terms", "relationships"):
                if stripped.startswith("- "):
                    current = {}
                    out[section].append(current)
                    stripped = stripped[2:].strip()
                if current is not None and ":" in stripped:
                    k, v = stripped.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if v.startswith("[") and v.endswith("]"):
                        items = [
                            x.strip().strip("'\"")
                            for x in re.split(r",", v[1:-1])
                            if x.strip()
                        ]
                        current[k] = items
                    elif v.lower() in ("true", "false"):
                        current[k] = v.lower() == "true"
                    elif v:
                        current[k] = v
        return out
