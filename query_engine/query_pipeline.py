from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from elasticsearch import Elasticsearch

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

LABEL_RECENCY_YEARS: int = 3
BM25_BOOST: float = 1.2
MUVERA_BOOST: float = 1.0
RECENCY_BOOST: float = 0.8

_POPULATION_FUZZY_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"kid(s|do|dos)?|child(ren)?|paed|pedi(atric)?|infant|neonat|baby|babies", re.I), "pediatric"),
    (re.compile(r"elder(ly)?|geriatric|older adult|senior|age\s*[≥>]=?\s*65", re.I), "elderly"),
    (re.compile(r"renal|kidney|ckd|egfr|gfr|creatinine|nephro", re.I), "renal_impairment"),
    (re.compile(r"hepat|liver|child.pugh|cirr|jaundic", re.I), "hepatic_impairment"),
    (re.compile(r"pregnan|lact|breastfeed|nursing|obstetric", re.I), "pregnancy"),
    (re.compile(r"obes|bmi\s*[>≥]\s*3", re.I), "obese"),
]


class DrugNameExtractor:
    def __init__(self, rxnorm_names: set[str]):
        self._medspacy = None
        sorted_names = sorted((name.strip().lower() for name in rxnorm_names if name.strip()), key=len, reverse=True)
        if sorted_names:
            pattern = "|".join(re.escape(name) for name in sorted_names)
            self._rx = re.compile(rf"\b({pattern})\b", re.IGNORECASE)
        else:
            self._rx = None

    def _get_medspacy(self):
        if self._medspacy is None:
            try:
                import medspacy

                self._medspacy = medspacy.load()
            except Exception as exc:
                log.warning(f"medspaCy unavailable for query-side NER: {exc}")
                self._medspacy = False
        return self._medspacy if self._medspacy is not False else None

    def extract(self, query: str) -> list[str]:
        names: list[str] = []
        if self._rx is not None:
            names = [match.group(0).lower() for match in self._rx.finditer(query)]
            if names:
                return list(dict.fromkeys(names))

        nlp = self._get_medspacy()
        if nlp is not None:
            doc = nlp(query)
            ner_names = [
                ent.text.lower()
                for ent in doc.ents
                if ent.label_ in ("DRUG", "CHEMICAL", "MEDICATION")
            ]
            if ner_names:
                return list(dict.fromkeys(ner_names))

        caps = re.findall(r"\b[A-Z][a-z]{3,}(?:-[A-Z][a-z]+)?\b", query)
        return [cap.lower() for cap in caps[:2]]


_drug_extractor: Optional[DrugNameExtractor] = None


def _get_drug_extractor() -> DrugNameExtractor:
    global _drug_extractor
    if _drug_extractor is None:
        names = set(getattr(CFG, "rxnorm_drug_names", set()))
        names |= {d.strip().lower() for d in getattr(CFG, "top_50_drugs", []) if d and d.strip()}
        
        try:
            res = ES.search(index=IDX, body={
                "size": 0,
                "aggs": {
                    "generics": {"terms": {"field": "drug_name_generic.keyword", "size": 1000}},
                    "brands": {"terms": {"field": "drug_name_brand.keyword", "size": 1000}}
                }
            })
            for bucket in res['aggregations']['generics']['buckets']:
                key = str(bucket.get('key', '')).strip().lower()
                if key:
                    names.add(key)
            for bucket in res['aggregations']['brands']['buckets']:
                key = str(bucket.get('key', '')).strip().lower()
                if key:
                    names.add(key)
        except Exception as e:
            log.error(f"Failed to sync drug dictionary with index aggregations: {e}")

        _drug_extractor = DrugNameExtractor(names)
    return _drug_extractor


def _normalize_population(query: str) -> Optional[str]:
    for pattern, normalized in _POPULATION_FUZZY_MAP:
        if pattern.search(query):
            return normalized
    return None


def _build_hybrid_dsl(
    intent: PharmQueryIntent,
    layout_type: str,
    q_muvera: list[float],
    q_content: list[float],
    k: int = 50,
    num_candidates: int = 500,
    population_filter: Optional[str] = None,
    hard_date_filter: bool = False,
) -> dict:
    del q_content

    bm25_query_text = " ".join(intent.drug_names + [intent.raw_query]).strip()
    bm25_clause = {
        "multi_match": {
            "query": bm25_query_text or intent.raw_query,
            "fields": [
                "verbatim_text^1.0",
                "drug_name_generic^3.0",
                "drug_name_brand^2.5",
            ],
            "type": "best_fields",
            "analyzer": "clinical_text",
            "boost": BM25_BOOST,
        }
    }

    filter_must: list[dict] = [
        {"term": {"layout_type": layout_type}},
    ]

    if intent.drug_names:
        drug_should = [
            {"match": {"drug_name_generic": {"query": name, "boost": 3.0}}}
            for name in intent.drug_names
        ] + [
            {"match": {"drug_name_brand": {"query": name, "boost": 2.5}}}
            for name in intent.drug_names
        ]
        filter_must.append({"bool": {"should": drug_should, "minimum_should_match": 1}})

    if population_filter:
        filter_must.append({"term": {"patient_population": population_filter}})

    recency_cutoff = (
        datetime.now().replace(month=1, day=1) - timedelta(days=LABEL_RECENCY_YEARS * 365)
    ).strftime("%Y-%m-%d")

    if hard_date_filter and intent.label_version_gte:
        filter_must.append({
            "range": {
                "label_version_date": {
                    "gte": intent.label_version_gte,
                    "lte": intent.label_version_lte,
                }
            }
        })

    knn_clause = {
        "field": "muvera_fde",
        "query_vector": q_muvera,
        "k": k,
        "num_candidates": num_candidates,
        "filter": {"bool": {"must": filter_must}},
        "boost": MUVERA_BOOST,
    }

    recency_function_score = {
        "function_score": {
            "query": {"bool": {"should": [bm25_clause]}},
            "functions": [
                {
                    "filter": {"range": {"label_version_date": {"gte": recency_cutoff}}},
                    "weight": RECENCY_BOOST,
                }
            ],
            "score_mode": "sum",
            "boost_mode": "sum",
        }
    }

    return {
        "knn": knn_clause,
        "query": recency_function_score,
        "_source": True,
        "size": k,
    }

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

        layout_filters: list[str] = []
        population_filter: Optional[str] = None

        for signal, (layout, pop) in MEDICAL_CONCEPT_MAP.items():
            if signal in q_lower:
                if layout not in layout_filters:
                    layout_filters.append(layout)
                if pop and not population_filter:
                    population_filter = pop

        fuzzy_population = _normalize_population(query)
        if fuzzy_population:
            population_filter = fuzzy_population

        if not layout_filters:
            layout_filters = ["dosing", "warning", "indication"]
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

        drug_names = _get_drug_extractor().extract(query)

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

    def _execute_hybrid_retrieval(
        self,
        intent: PharmQueryIntent,
        q_vectors: dict,
        hard_date_filter: bool = False,
    ) -> list[dict]:
        all_hits: dict[str, dict] = {}

        pools_to_search = [(layout, layout) for layout in intent.layout_filters[:4]]

        if intent.wants_interaction and "interaction" not in intent.layout_filters:
            pools_to_search.append(("interaction", "interaction"))

        if intent.wants_mechanism and "pharmacology" not in intent.layout_filters:
            pools_to_search.append(("mechanism", "pharmacology"))

        for pool_name, layout_type in pools_to_search:

            try:
                dsl = _build_hybrid_dsl(
                    intent=intent,
                    layout_type=layout_type,
                    q_muvera=q_vectors["muvera_fde"],
                    q_content=q_vectors["content_vector"],
                    population_filter=intent.population_filter if layout_type == "dosing" else None,
                    hard_date_filter=hard_date_filter,
                )
                resp = ES.search(
                    index=IDX,
                    body=dsl,
                )
                hits = resp["hits"]["hits"]
            except Exception as exc:
                log.warning(f"Hybrid retrieval failed for pool '{pool_name}': {exc} — fallback")
                resp = ES.search(
                    index=IDX,
                    body={
                        "knn": {
                            "field": "content_vector",
                            "query_vector": q_vectors["content_vector"],
                            "k": 50,
                            "num_candidates": 200,
                            "filter": {"term": {"layout_type": layout_type}},
                        },
                        "_source": True,
                        "size": 50,
                    }
                )
                hits = resp["hits"]["hits"]

            log.info(
                f"Step 3 [{pool_name}] — hybrid retrieved {len(hits)} candidates",
                extra={"pool": pool_name, "n_candidates": len(hits)}
            )

            top_hits = self._apply_colbert_maxsim_reranking(
                q_vectors["colbert_tokens"], hits, top_k=10
            )

            for hit in top_hits:
                urn = hit["_source"].get("urn_id", "")
                if urn and urn not in all_hits:
                    all_hits[urn] = hit

            log.info(
                f"Step 4 [{pool_name}] — MaxSim top-ranked: "
                f"{[h['_source'].get('urn_id','')[-35:] for h in top_hits]}",
                extra={"pool": pool_name}
            )

        return list(all_hits.values())

    def _apply_colbert_maxsim_reranking(self, query_tokens: list[dict], docs: list[dict], top_k: int = 10) -> list[dict]:
        if not docs:
            return []
        if not query_tokens:
            return docs[:top_k]

        q_vecs = np.array([t["v"] for t in query_tokens], dtype=np.float32)
        q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-9
        q_vecs = q_vecs / q_norms

        for doc in docs:
            d_toks = doc["_source"].get("colbert_tokens") or []
            if not d_toks:
                doc["_maxsim"] = 0.0
                continue
            d_vecs = np.array([t["v"] for t in d_toks], dtype=np.float32)
            d_norms = np.linalg.norm(d_vecs, axis=1, keepdims=True) + 1e-9
            d_vecs = d_vecs / d_norms
            sim_matrix = q_vecs @ d_vecs.T
            doc["_maxsim"] = float(np.mean(np.max(sim_matrix, axis=1)))

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

    def _fetch_parent_nodes(self, hits: list[dict], max_parents: int = 5) -> list[dict]:
        parent_urns: list[str] = []
        seen: set[str] = set()

        for hit in hits:
            parent_urn = hit.get("_source", {}).get("parent_urn")
            if parent_urn and parent_urn not in seen:
                parent_urns.append(parent_urn)
                seen.add(parent_urn)
            if len(parent_urns) >= max_parents:
                break

        if not parent_urns:
            return []

        try:
            resp = ES.search(
                index=IDX,
                body={
                    "query": {"terms": {"urn_id": parent_urns}},
                    "size": max_parents,
                    "_source": True,
                }
            )
            parents = resp["hits"]["hits"]
            for parent in parents:
                parent["_maxsim"] = 0.0
                parent["_is_parent_pivot"] = True
            return parents
        except Exception as exc:
            log.warning(f"Parent pivot retrieval failed: {exc}")
            return []

    def _fetch_atc_neighbors(
        self,
        atc_codes: list[str],
        layout_type: str,
        q_vectors: dict,
        top_k: int = 5,
    ) -> list[dict]:
        prefixes = list({code[:4] for code in atc_codes if code and len(code) >= 4})
        if not prefixes:
            return []

        prefix_should = [{"prefix": {"atc_code": prefix}} for prefix in prefixes]
        try:
            resp = ES.search(
                index=IDX,
                body={
                    "knn": {
                        "field": "muvera_fde",
                        "query_vector": q_vectors["muvera_fde"],
                        "k": 30,
                        "num_candidates": 200,
                        "filter": {
                            "bool": {
                                "must": [
                                    {"term": {"layout_type": layout_type}},
                                    {"bool": {"should": prefix_should, "minimum_should_match": 1}},
                                ]
                            }
                        },
                    },
                    "_source": True,
                    "size": 30,
                }
            )
            candidates = resp["hits"]["hits"]
            neighbors = self._apply_colbert_maxsim_reranking(
                q_vectors["colbert_tokens"], candidates, top_k=top_k
            )
            for neighbor in neighbors:
                neighbor["_is_atc_neighbor"] = True
            return neighbors
        except Exception as exc:
            log.warning(f"ATC neighbor retrieval failed: {exc}")
            return []

    def _hit_to_node(self, hit: dict) -> RetrievedNode:
        node = RetrievedNode.from_es_hit(
            hit.get("_source", {}),
            maxsim_score=hit.get("_maxsim", 0.0),
        )
        if hit.get("_is_parent_pivot"):
            node.__dict__["_is_parent_pivot"] = True
        if hit.get("_is_atc_neighbor"):
            node.__dict__["_is_atc_neighbor"] = True
        return node

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

        if not node.dose_values:
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
                    "_source": ["dose_values", "dose_units", "dose_val", "dose_unit", "urn_id"],
                    "size": 1,
                }
            )
            if resp["hits"]["hits"]:
                fact = resp["hits"]["hits"][0]["_source"]
                node.verified_dose_values = fact.get("dose_values") or ([] if fact.get("dose_val") is None else [fact["dose_val"]])
                verified_units = fact.get("dose_units") or ([] if fact.get("dose_unit") is None else [fact["dose_unit"]])
                node.verified_dose_units = verified_units
                node.confidence_verified = True
                log.info(
                    f"Calibrated: {node.urn[-35:]} → "
                    f"verified_doses={node.verified_dose_values} {'/'.join(verified_units)}",
                    extra={"urn": node.urn, "verified": node.verified_dose_values}
                )
        except Exception as e:
            log.warning(f"Confidence calibration failed: {e}")

        return node

    def _assemble_cognitive_artifact(self, intent: PharmQueryIntent, primary_hits: list[dict], start_time: float) -> CogCanvasArtifact:
        primary_nodes: list[RetrievedNode] = []
        for hit in primary_hits:
            node = self._hit_to_node(hit)
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
        explicit_year_filter = bool(re.search(r"\b(20\d{2})\b", query))

        with Timer(log, "query_pipeline", query=query[:60]):
            intent = self._decompose_semantic_intent(query)

            with Timer(log, "embed_query"):
                q_vectors = embed_document(query)

            with Timer(log, "retrieval"):
                primary_hits = self._execute_hybrid_retrieval(
                    intent,
                    q_vectors,
                    hard_date_filter=explicit_year_filter,
                )

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