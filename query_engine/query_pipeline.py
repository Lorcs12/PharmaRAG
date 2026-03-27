import re
import time
import numpy as np
from elasticsearch import Elasticsearch
from datetime import datetime

from config import CFG
from embedder import embed_document, get_model
from logger import get_logger, Timer

from .constants import (
    QUERY_TYPE_CAUSAL, QUERY_TYPE_COMPARATIVE, QUERY_TYPE_STRATEGIC, QUERY_TYPE_FACTUAL,
    CAUSAL_SIGNALS, COMPARATIVE_SIGNALS, STRATEGIC_SIGNALS, MEDICAL_CONCEPT_MAP
)
from .models import PharmQueryIntent, RetrievedNode
from .cognitive_canvas import CogCanvasArtifact

log = get_logger("query_engine", CFG.log.file, CFG.log.level)
ES  = Elasticsearch(CFG.es.host, request_timeout=CFG.es.request_timeout)
IDX = CFG.es.index

class PharmaQueryEngine:

    def __init__(self):
        get_model()

    def _decompose_semantic_intent(self, query: str) -> PharmQueryIntent:
        q_lower = query.lower()
        words   = set(q_lower.split())

        if words & CAUSAL_SIGNALS or "why" in q_lower:
            query_type = QUERY_TYPE_CAUSAL
        elif words & COMPARATIVE_SIGNALS:
            query_type = QUERY_TYPE_COMPARATIVE
        elif words & STRATEGIC_SIGNALS:
            query_type = QUERY_TYPE_STRATEGIC
        else:
            query_type = QUERY_TYPE_FACTUAL

        layout_filters    = []
        population_filter = None
        preferred_layout  = "dosing"

        for signal, (layout, pop) in MEDICAL_CONCEPT_MAP.items():
            if signal in q_lower:
                if layout not in layout_filters:
                    layout_filters.append(layout)
                if pop and not population_filter:
                    population_filter = pop

        if not layout_filters:
            layout_filters = ["dosing", "warning", "indication"]
            preferred_layout = "dosing"
        else:
            preferred_layout = layout_filters[0]

        wants_interaction = (
            query_type == QUERY_TYPE_CAUSAL
            or "interaction" in preferred_layout
            or any(s in q_lower for s in ("with", "together", "combine", "and"))
        )
        wants_dosing = "dosing" in layout_filters
        wants_mechanism = (
            "pharmacology" in layout_filters
            or query_type == QUERY_TYPE_CAUSAL
        )

        drug_names = []
        for drug in CFG.top_50_drugs:
            if drug.lower() in q_lower:
                drug_names.append(drug)
        if not drug_names:

            caps = re.findall(r'\b[A-Z][a-z]{3,}\b', query)
            drug_names = caps[:2]

        from datetime import datetime
        current_year = datetime.now().year
        label_version_gte = f"{current_year - 5}-01-01"
        label_version_lte = f"{current_year}-12-31"

        years = re.findall(r'\b(20\d{2})\b', query)
        if years:
            label_version_gte = f"{min(years)}-01-01"
            label_version_lte = f"{max(years)}-12-31"

        intent = PharmQueryIntent(
            raw_query         = query,
            query_type        = query_type,
            preferred_layout  = preferred_layout,
            layout_filters    = layout_filters,
            population_filter = population_filter,
            drug_names        = drug_names,
            label_version_gte = label_version_gte,
            label_version_lte = label_version_lte,
            wants_interaction = wants_interaction,
            wants_dosing      = wants_dosing,
            wants_mechanism   = wants_mechanism,
        )

        log.info(
            f"Step 1 — type={query_type} | layout={preferred_layout} | "
            f"population={population_filter} | drugs={drug_names}",
            extra={"query_type": query_type, "layout": preferred_layout,
                   "population": population_filter, "drugs": drug_names}
        )
        return intent

    def _execute_hybrid_retrieval(self, intent: PharmQueryIntent, q_vectors: dict) -> list[dict]:
        all_hits: dict[str, dict] = {}

        pools_to_search = [(layout, layout) for layout in intent.layout_filters[:3]]

        if intent.wants_interaction and "interaction" not in intent.layout_filters:
            pools_to_search.append(("interaction", "interaction"))

        if intent.wants_mechanism and "pharmacology" not in intent.layout_filters:
            pools_to_search.append(("mechanism", "pharmacology"))

        for pool_name, layout_type in pools_to_search:

            must_clauses: list[dict] = [
                {"term": {"layout_type": layout_type}},
            ]

            if intent.drug_names:
                drug_should = [
                    {"match": {"drug_name_generic": name}}
                    for name in intent.drug_names
                ] + [
                    {"match": {"drug_name_brand": name}}
                    for name in intent.drug_names
                ]
                must_clauses.append({"bool": {"should": drug_should, "minimum_should_match": 1}})

            if layout_type == "dosing" and intent.population_filter:
                must_clauses.append({
                    "term": {"patient_population": intent.population_filter}
                })

            if intent.label_version_gte and intent.label_version_lte:
                must_clauses.append({
                    "range": {
                        "label_version_date": {
                            "gte": intent.label_version_gte,
                            "lte": intent.label_version_lte,
                        }
                    }
                })

            es_filter = {"bool": {"must": must_clauses}}

            try:
                resp = ES.search(
                    index=IDX,
                    body={
                        "knn": {
                            "field":          "muvera_fde",
                            "query_vector":   q_vectors["muvera_fde"],
                            "k":              50,
                            "num_candidates": 500,
                            "filter":         es_filter,
                        },
                        "_source": True,
                        "size":    50,
                    }
                )
                hits = resp["hits"]["hits"]
            except Exception as e:
                log.warning(f"MUVERA failed for pool '{pool_name}': {e} — fallback")
                resp = ES.search(
                    index=IDX,
                    body={
                        "query": es_filter,
                        "knn": {
                            "field":          "content_vector",
                            "query_vector":   q_vectors["content_vector"],
                            "k":              50,
                            "num_candidates": 200,
                        },
                        "_source": True,
                        "size":    50,
                    }
                )
                hits = resp["hits"]["hits"]

            log.info(
                f"Step 3 [{pool_name}] — MUVERA retrieved {len(hits)} candidates",
                extra={"pool": pool_name, "n_candidates": len(hits)}
            )

            top_hits = self._apply_colbert_maxsim_reranking(
                q_vectors["colbert_tokens"], hits, top_k=5
            )

            for hit in top_hits:
                urn = hit["_source"].get("urn_id", "")
                if urn and urn not in all_hits:
                    all_hits[urn] = hit

            log.info(
                f"Step 4 [{pool_name}] — MaxSim top-5: "
                f"{[h['_source'].get('urn_id','')[-35:] for h in top_hits]}",
                extra={"pool": pool_name}
            )

        return list(all_hits.values())

    def _apply_colbert_maxsim_reranking(self, query_tokens: list[dict], docs: list[dict], top_k: int = 5) -> list[dict]:
        if not docs:
            return []
        if not query_tokens:
            return docs[:top_k]

        q_vecs = np.array([t["v"] for t in query_tokens], dtype=np.float32)

        for doc in docs:
            d_toks = doc["_source"].get("colbert_tokens") or []
            if not d_toks:
                doc["_maxsim"] = 0.0
                continue
            d_vecs = np.array([t["v"] for t in d_toks], dtype=np.float32)
            sim_matrix      = q_vecs @ d_vecs.T
            doc["_maxsim"]  = float(np.sum(np.max(sim_matrix, axis=1)))

        docs.sort(key=lambda x: x.get("_maxsim", 0.0), reverse=True)
        return docs[:top_k]

    def _fetch_expansion_nodes(self, urns: list[str]) -> list[dict]:
        if not urns:
            return []
        resp = ES.search(
            index=IDX,
            body={
                "query": {"terms": {"urn_id": list(urns)}},
                "size":  100,
                "_source": True,
            }
        )
        return [h["_source"] for h in resp["hits"]["hits"]]

    def _enforce_verbatim_constraints(self, node: RetrievedNode) -> RetrievedNode:
        threshold = CFG.ingestion.raglens_verbatim_threshold
        if node.raglens_risk >= threshold:
            node.verbatim_locked = True
            log.info(
                f"RAGLens LOCK: {node.urn[-45:]} "
                f"(risk={node.raglens_risk:.2f} ≥ {threshold})",
                extra={"urn": node.urn, "risk": node.raglens_risk}
            )
        return node

    def _calibrate_retrieval_confidence(self, node: RetrievedNode) -> RetrievedNode:
        if node.chunk_confidence >= 0.6:
            node.confidence_verified = True
            return node

        if node.dose_val is None:
            return node

        try:
            resp = ES.search(
                index=IDX,
                body={
                    "query": {"bool": {"must": [
                        {"term": {"rxcui": node.rxcui}},
                        {"term": {"layout_type": "dosing"}},
                        {"range": {"chunk_confidence": {"gte": 0.90}}},
                    ]}},
                    "_source": ["dose_val", "dose_unit", "urn_id"],
                    "size": 1,
                }
            )
            if resp["hits"]["hits"]:
                fact = resp["hits"]["hits"][0]["_source"]
                node.verified_dose_val   = fact.get("dose_val")
                node.confidence_verified = True
                log.info(
                    f"Calibrated: {node.urn[-35:]} → "
                    f"verified_dose={node.verified_dose_val} {fact.get('dose_unit','')}",
                    extra={"urn": node.urn, "verified": node.verified_dose_val}
                )
        except Exception as e:
            log.warning(f"Confidence calibration failed: {e}")

        return node

    def _assemble_cognitive_artifact(self, intent: PharmQueryIntent, primary_hits: list[dict], start_time: float) -> CogCanvasArtifact:
        primary_nodes: list[RetrievedNode] = []
        for hit in primary_hits:
            node = RetrievedNode.from_es_hit(
                hit["_source"],
                maxsim_score=hit.get("_maxsim", 0.0)
            )
            node = self._enforce_verbatim_constraints(node)
            node = self._calibrate_retrieval_confidence(node)
            primary_nodes.append(node)

        interaction_urns = set()
        raptor_urns      = set()
        table_urns       = set()

        for node in primary_nodes:
            if intent.wants_interaction or intent.query_type in (
                QUERY_TYPE_CAUSAL, QUERY_TYPE_COMPARATIVE
            ):
                interaction_urns.update(node.interaction_ids)

            if intent.query_type in (QUERY_TYPE_STRATEGIC, QUERY_TYPE_COMPARATIVE):
                if node.raptor_cluster:
                    raptor_urns.add(node.raptor_cluster)

            table_urns.update(node.table_ref)

        expansion_sources = self._fetch_expansion_nodes(
            list(interaction_urns | raptor_urns | table_urns)
        )
        expansion_by_urn = {s["urn_id"]: s for s in expansion_sources}

        def _make_nodes(urn_set: set) -> list[RetrievedNode]:
            nodes = []
            for urn in urn_set:
                if urn in expansion_by_urn:
                    n = RetrievedNode.from_es_hit(expansion_by_urn[urn])
                    n = self._enforce_verbatim_constraints(n)
                    n = self._calibrate_retrieval_confidence(n)
                    nodes.append(n)
            return nodes

        interaction_nodes = _make_nodes(interaction_urns)
        raptor_nodes      = _make_nodes(raptor_urns)
        table_nodes       = _make_nodes(table_urns)

        verbatim_nodes   = [n for n in primary_nodes if n.verbatim_locked]
        paraphrase_nodes = [n for n in primary_nodes if not n.verbatim_locked]

        artifact = CogCanvasArtifact(
            query            = intent.raw_query,
            intent           = intent,
            verbatim_nodes   = verbatim_nodes,
            paraphrase_nodes = paraphrase_nodes,
            causal_context   = interaction_nodes,
            macro_context    = raptor_nodes,
            table_references = table_nodes,
            total_latency_ms = (time.perf_counter() - start_time) * 1000,
        )
        artifact.conflicts = artifact._detect_conflicts()

        log.info(
            f"CogCanvas assembled: {artifact.get_artifact_summary()}",
            extra={"query": intent.raw_query,
                   "latency_ms": artifact.total_latency_ms}
        )
        return artifact

    def execute_query_pipeline(self, query: str) -> CogCanvasArtifact:
        start = time.perf_counter()

        with Timer(log, "query_pipeline", query=query[:60]):
            intent = self._decompose_semantic_intent(query)

            with Timer(log, "embed_query"):
                q_vectors = embed_document(query)

            with Timer(log, "retrieval"):
                primary_hits = self._execute_hybrid_retrieval(intent, q_vectors)

            if not primary_hits:
                log.warning("No results found — returning empty artifact")
                return CogCanvasArtifact(
                    query=query, intent=intent,
                    verbatim_nodes=[], paraphrase_nodes=[],
                    causal_context=[], macro_context=[],
                    table_references=[],
                    total_latency_ms=(time.perf_counter() - start) * 1000,
                )

            with Timer(log, "assembly"):
                artifact = self._assemble_cognitive_artifact(intent, primary_hits, start)

        return artifact