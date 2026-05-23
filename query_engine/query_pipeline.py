from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from elasticsearch import Elasticsearch

from config import CFG
from embedder import embed_document, get_model
from logger import get_logger, Timer

from .constants import (
    QUERY_TYPE_CAUSAL, QUERY_TYPE_COMPARATIVE, QUERY_TYPE_STRATEGIC, QUERY_TYPE_FACTUAL,
    CAUSAL_SIGNALS, COMPARATIVE_SIGNALS, STRATEGIC_SIGNALS, MEDICAL_CONCEPT_MAP,
    LABEL_RECENCY_YEARS, BM25_BOOST, MUVERA_BOOST, RECENCY_BOOST,
    CLINICAL_LAYOUT_MATCH_BOOST, CLINICAL_POPULATION_MATCH_BOOST,
    CLINICAL_DOSE_METADATA_BOOST, INTERACTION_LAYOUT_OVERRIDE_BOOST,
    INTERACTION_DOSING_PENALTY,
    POPULATION_ALIASES, POPULATION_FUZZY_MAP,
    BRAND_HINT_MAP, COMBO_SPLIT_RE, BRAND_GENERIC_EXPANSIONS,
    INDICATION_QUERY_RE, DOSING_QUERY_RE, SAFETY_INTERACTION_QUERY_RE,
    INTERACTION_QUERY_RE, ADVERSE_REACTION_QUERY_RE, SUPPLY_QUERY_RE,
    BOXED_WARNING_QUERY_RE, WARNING_NUMERICS_QUERY_RE,
    DRUG_NAME_STOP_WORDS,
)
from .models import PharmQueryIntent, RetrievedNode
from .cognitive_canvas import CogCanvasArtifact

log = get_logger("query_engine", CFG.log.file, CFG.log.level)
ES  = Elasticsearch(CFG.es.host, request_timeout=CFG.es.request_timeout)
IDX = CFG.es.index


def _expand_combo_aliases(names: set[str]) -> set[str]:
    """Add ingredient-level aliases for combo drug names.

    Example:
      "sofosbuvir, velpatasvir, and voxilaprevir"
      -> "sofosbuvir", "velpatasvir", "voxilaprevir"
    """
    expanded = set(names)
    for raw in list(names):
        name = raw.strip().lower()
        if not name:
            continue
        parts = [p.strip() for p in COMBO_SPLIT_RE.split(name) if p and p.strip()]
        for part in parts:
            # Skip glue words and very short fragments.
            if part in {"and", "with"} or len(part) < 3:
                continue
            expanded.add(part)
    return expanded


def _expand_brand_aliases(query: str, extracted_names: list[str]) -> list[str]:
    expanded = {n.strip().lower() for n in extracted_names if n and n.strip()}
    q = query.lower()

    # If nothing was extracted, still allow direct brand phrase detection.
    for brand, generics in BRAND_GENERIC_EXPANSIONS.items():
        if brand in q:
            expanded.add(brand)
            expanded.update(generics)

    for name in list(expanded):
        if name in BRAND_GENERIC_EXPANSIONS:
            expanded.update(BRAND_GENERIC_EXPANSIONS[name])

    return sorted(expanded)


class DrugNameExtractor:
    def __init__(self, rxnorm_names: set[str]):
        self._medspacy = None
        sorted_names = sorted((name.strip().lower() for name in rxnorm_names if name.strip()), key=len, reverse=True)
        self._name_set = set(sorted_names)
        if sorted_names:
            pattern = "|".join(re.escape(name) for name in sorted_names)
            self._rx = re.compile(rf"\b({pattern})\b", re.IGNORECASE)
        else:
            self._rx = None

    def _repair_multiword_names(self, query: str, names: list[str]) -> list[str]:
        """Promote partial matches to known multi-word drug names when possible."""
        if not names or not self._name_set:
            return names

        query_tokens = re.findall(r"[a-z0-9]+", query.lower())
        if not query_tokens:
            return names

        repaired: list[str] = []
        for name in names:
            parts = name.split()
            best = name
            for idx in range(0, len(query_tokens) - len(parts) + 1):
                if query_tokens[idx:idx + len(parts)] != parts:
                    continue
                # Grow to a two-token suffix if a canonical dictionary phrase exists.
                for suffix_len in (1, 2):
                    end = idx + len(parts) + suffix_len
                    if end > len(query_tokens):
                        continue
                    candidate = " ".join(query_tokens[idx:end])
                    if candidate in self._name_set:
                        best = candidate
                break
            repaired.append(best)
        return list(dict.fromkeys(repaired))

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
                return self._repair_multiword_names(query, list(dict.fromkeys(names)))

        nlp = self._get_medspacy()
        if nlp is not None:
            doc = nlp(query)
            ner_names = [
                ent.text.lower()
                for ent in doc.ents
                if ent.label_ in ("DRUG", "CHEMICAL", "MEDICATION")
            ]
            if ner_names:
                normalized = [n.strip() for n in ner_names if n.strip()]
                normalized = self._repair_multiword_names(query, list(dict.fromkeys(normalized)))
                canonical = [n for n in normalized if n in self._name_set]
                return canonical or normalized

        caps = re.findall(r"\b(?:[A-Z]{4,}|[A-Z][a-z]{3,})(?:-(?:[A-Z]{2,}|[A-Z][a-z]+))?\b", query)
        fallback = [cap.lower() for cap in caps[:2] if cap.lower() not in DRUG_NAME_STOP_WORDS]
        fallback = self._repair_multiword_names(query, fallback)
        canonical = [n for n in fallback if n in self._name_set]
        return canonical or fallback


_drug_extractor: DrugNameExtractor | None = None


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

        names = _expand_combo_aliases(names)
        _drug_extractor = DrugNameExtractor(names)
    return _drug_extractor


def _detect_population(query: str) -> str | None:
    for pattern, normalized in POPULATION_FUZZY_MAP:
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
    population_filter: str | None = None,
    hard_date_filter: bool = False,
    strict_drug_filter: bool = False,
    boxed_warning_filter: bool = False,
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
        if strict_drug_filter:
            # Harder drug-centric pruning for drug-specific causal/comparative queries.
            # Keep both exact keyword and phrase match variants so aliases and
            # multi-word labels can still pass.
            drug_should = [
                {"term": {"drug_name_generic.keyword": name}}
                for name in intent.drug_names
            ] + [
                {"term": {"drug_name_brand.keyword": name}}
                for name in intent.drug_names
            ] + [
                {"match_phrase": {"drug_name_generic": name}}
                for name in intent.drug_names
            ] + [
                {"match_phrase": {"drug_name_brand": name}}
                for name in intent.drug_names
            ]
        else:
            drug_should = [
                {"match": {"drug_name_generic": {"query": name, "boost": 3.0}}}
                for name in intent.drug_names
            ] + [
                {"match": {"drug_name_brand": {"query": name, "boost": 2.5}}}
                for name in intent.drug_names
            ]
        filter_must.append({"bool": {"should": drug_should, "minimum_should_match": 1}})

    if boxed_warning_filter:
        filter_must.append({"term": {"boxed_warning": True}})

    if population_filter:
        aliases = POPULATION_ALIASES.get(population_filter, [population_filter])
        if len(aliases) == 1:
            filter_must.append({"term": {"patient_population": aliases[0]}})
        else:
            filter_must.append({"terms": {"patient_population": aliases}})

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


@dataclass(frozen=True)
class SearchBranch:
    name: str
    layout_type: str
    use_population_filter: bool = False
    use_hyde_vector: bool = False
    filter_boxed_warning: bool = False

class PharmaQueryEngine:

    def __init__(self):
        get_model()

    def _decompose_semantic_intent(self, query: str) -> PharmQueryIntent:
        q_lower = query.lower()
        words   = set(q_lower.split())
        explicit_dosing = bool(DOSING_QUERY_RE.search(q_lower))
        safety_or_interaction = bool(SAFETY_INTERACTION_QUERY_RE.search(q_lower))

        if words & CAUSAL_SIGNALS or "why" in q_lower:
            query_type = QUERY_TYPE_CAUSAL
        elif words & COMPARATIVE_SIGNALS:
            query_type = QUERY_TYPE_COMPARATIVE
        elif words & STRATEGIC_SIGNALS:
            query_type = QUERY_TYPE_STRATEGIC
        else:
            query_type = QUERY_TYPE_FACTUAL

        layout_filters: list[str] = []
        population_filter: str | None = None

        for signal, (layout, pop) in MEDICAL_CONCEPT_MAP.items():
            if signal in q_lower:
                if layout not in layout_filters:
                    layout_filters.append(layout)
                if pop and not population_filter:
                    population_filter = pop

        # Safety/interaction phrasing should not be treated as dose-first by default.
        if safety_or_interaction:
            for layout in ("warning", "interaction"):
                if layout not in layout_filters:
                    layout_filters.insert(0, layout)

        # Indication-style phrasing should route to indication-first retrieval.
        if INDICATION_QUERY_RE.search(q_lower) and "indication" not in layout_filters:
            layout_filters.insert(0, "indication")

        fuzzy_population = _detect_population(query)
        if fuzzy_population:
            population_filter = fuzzy_population

        if not layout_filters:
            if query_type == QUERY_TYPE_CAUSAL or safety_or_interaction:
                layout_filters = ["interaction", "warning", "pharmacology", "indication"]
            else:
                layout_filters = ["dosing", "warning", "indication"]

        # Contraindication-only retrieval is too narrow for many safety questions.
        if "contraindication" in layout_filters and not any(
            layout in layout_filters for layout in ("warning", "interaction")
        ):
            layout_filters.extend(["warning", "interaction"])

        # Adverse-reaction queries need the adverse_reaction pool first; the generic
        # "warning" pool (populated by the "adverse" signal above) is a fallback only.
        if ADVERSE_REACTION_QUERY_RE.search(q_lower) and "adverse_reaction" not in layout_filters:
            layout_filters.insert(0, "adverse_reaction")
            if "warning" not in layout_filters:
                layout_filters.append("warning")

        # Supply / formulation / packaging queries need the supply pool.
        if SUPPLY_QUERY_RE.search(q_lower) and "supply" not in layout_filters:
            layout_filters.insert(0, "supply")
            if "indication" not in layout_filters:
                layout_filters.append("indication")

        preferred_layout = layout_filters[0]

        explicit_interaction = bool(INTERACTION_QUERY_RE.search(q_lower))
        wants_interaction = (
            query_type == QUERY_TYPE_CAUSAL
            or "interaction" in layout_filters
            or explicit_interaction
        )
        wants_dosing = explicit_dosing or ("dosing" in layout_filters and not safety_or_interaction)
        wants_mechanism = (
            "pharmacology" in layout_filters
            or query_type == QUERY_TYPE_CAUSAL
        )

        drug_names = _expand_brand_aliases(query, _get_drug_extractor().extract(query))

        wants_boxed_warning = bool(BOXED_WARNING_QUERY_RE.search(q_lower))
        if wants_boxed_warning and "warning" not in layout_filters:
            layout_filters.insert(0, "warning")

        wants_warning_numerics = bool(WARNING_NUMERICS_QUERY_RE.search(q_lower))

        brand_hint: str | None = None
        for brand_key, brand_desc in BRAND_HINT_MAP.items():
            if brand_key in q_lower:
                brand_hint = brand_desc
                break

        # Context-based brand inference for same-molecule brands
        if not brand_hint and "liraglutide" in q_lower:
            if any(w in q_lower for w in ("type 2", "t2dm", "glycemic", "diabetes mellitus")):
                brand_hint = BRAND_HINT_MAP["victoza"]
            elif any(w in q_lower for w in ("weight", "obesity", "bmi")):
                brand_hint = BRAND_HINT_MAP["saxenda"]
        if not brand_hint and "semaglutide" in q_lower:
            if any(w in q_lower for w in ("oral", "tablet")):
                brand_hint = BRAND_HINT_MAP["rybelsus"]
            elif any(w in q_lower for w in ("weight", "obesity", "bmi")):
                brand_hint = BRAND_HINT_MAP["wegovy"]
            elif any(w in q_lower for w in ("type 2", "t2dm", "glycemic", "diabetes")):
                brand_hint = BRAND_HINT_MAP["ozempic"]

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
            wants_boxed_warning = wants_boxed_warning,
            wants_warning_numerics = wants_warning_numerics,
            brand_hint        = brand_hint,
        )

        log.info(
            f"Step 1 — type={query_type} | layout={preferred_layout} | "
            f"population={population_filter} | drugs={drug_names}",
            extra={"query_type": query_type, "layout": preferred_layout,
                   "population": population_filter, "drugs": drug_names}
        )
        return intent

    def _execute_parallel_retrieval(
        self,
        intent: PharmQueryIntent,
        q_vectors: dict,
        hard_date_filter: bool = False,
    ) -> list[dict]:
        all_hits: dict[str, dict] = {}
        branches = self._build_search_branches(intent)

        # HyDE-lite: for dosing queries with known drugs/population, blend the original
        # query embedding with a template-based "hypothetical answer" embedding.
        # This closes the vocabulary gap between a natural-language question and the
        # imperative prose in drug-label dose sections (state-of-the-art HyDE technique).
        hyde_muvera: list[float] | None = None
        if intent.wants_dosing and intent.drug_names:
            hyde_text = self._generate_hyde_text(intent)
            try:
                hyde_vecs = embed_document(hyde_text)
                orig = np.array(q_vectors["muvera_fde"], dtype=np.float32)
                hyp  = np.array(hyde_vecs["muvera_fde"], dtype=np.float32)
                blended = 0.55 * orig + 0.45 * hyp
                norm = np.linalg.norm(blended) + 1e-9
                hyde_muvera = (blended / norm).tolist()
            except Exception as exc:
                log.warning(f"HyDE embedding failed, skipping: {exc}")

        max_workers = min(max(1, len(branches)), 6)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_branch,
                    branch,
                    intent,
                    q_vectors,
                    hyde_muvera,
                    hard_date_filter,
                ): branch
                for branch in branches
            }

            for future in as_completed(futures):
                branch = futures[future]
                pool_name = branch.name
                top_hits: list[dict] = []
                try:
                    top_hits = future.result()
                except Exception as exc:
                    log.warning(f"Parallel branch failure for '{pool_name}': {exc}")

                for hit in top_hits:
                    urn = hit["_source"].get("urn_id", "")
                    if urn and urn not in all_hits:
                        all_hits[urn] = hit

                log.info(
                    f"Step 4 [{pool_name}] — Clinical top-ranked: "
                    f"{[h['_source'].get('urn_id','')[-35:] for h in top_hits]}",
                    extra={"pool": pool_name}
                )

        # Final authority-aware rerank happens once over merged context.
        return self._apply_clinical_reranking(list(all_hits.values()), intent)[:15]

    def _build_search_branches(self, intent: PharmQueryIntent) -> list[SearchBranch]:
        """Build strategy branches for parallel retrieval orchestration."""
        branches: list[SearchBranch] = []
        seen_layouts: set[str] = set()

        def add_branch(pool_name: str, layout_type: str, use_population: bool, use_hyde: bool = False) -> None:
            # Allow multiple branches for the same layout_type if population filter differs
            branch_key = f"{layout_type}_{use_population}"
            if branch_key in {f"{b.layout_type}_{b.use_population_filter}" for b in branches}:
                return
            branches.append(
                SearchBranch(
                    name=pool_name,
                    layout_type=layout_type,
                    use_population_filter=use_population,
                    use_hyde_vector=use_hyde,
                )
            )
            if use_population:
                seen_layouts.add(layout_type)

        for layout in intent.layout_filters[:6]:
            # structured_fact contains population-agnostic reference ranges — only filter dosing by population
            use_pop = (layout == "dosing")
            add_branch(layout, layout, use_population=use_pop)

        if intent.preferred_layout == "contraindication":
            add_branch("warning", "warning", use_population=False)
            add_branch("interaction", "interaction", use_population=False)

        if intent.wants_interaction:
            add_branch("interaction", "interaction", use_population=False)

        if intent.wants_mechanism:
            add_branch("mechanism", "pharmacology", use_population=False)

        if intent.wants_dosing:
            add_branch("structured_facts", "structured_fact", use_population=False, use_hyde=True)

            if intent.population_filter:
                add_branch("dosing_base", "dosing", use_population=False, use_hyde=True)

        if intent.wants_boxed_warning:
            branches.append(SearchBranch(
                name="boxed_warning",
                layout_type="warning",
                use_population_filter=False,
                filter_boxed_warning=True,
            ))

        if intent.query_type == QUERY_TYPE_STRATEGIC:
            for layout in ("warning", "contraindication", "indication"):
                add_branch(f"strategic_{layout}", layout, use_population=False)

        return branches

    def _run_branch(
        self,
        branch: SearchBranch,
        intent: PharmQueryIntent,
        q_vectors: dict,
        hyde_muvera: list[float] | None,
        hard_date_filter: bool,
    ) -> list[dict]:
        pool_name = branch.name
        layout_type = branch.layout_type
        use_population = branch.use_population_filter
        use_hyde = branch.use_hyde_vector

        try:
            q_muvera = (
                hyde_muvera if use_hyde and hyde_muvera is not None else q_vectors["muvera_fde"]
            )
            dsl = _build_hybrid_dsl(
                intent=intent,
                layout_type=layout_type,
                q_muvera=q_muvera,
                q_content=q_vectors["content_vector"],
                population_filter=(intent.population_filter if use_population else None),
                hard_date_filter=hard_date_filter,
                strict_drug_filter=bool(
                    intent.drug_names
                    and (
                        intent.query_type in (QUERY_TYPE_CAUSAL, QUERY_TYPE_COMPARATIVE)
                        or intent.drug_lockdown_exact
                    )
                ),
                boxed_warning_filter=branch.filter_boxed_warning,
            )
            resp = ES.search(index=IDX, body=dsl)
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
                },
            )
            hits = resp["hits"]["hits"]

        log.info(
            f"Step 3 [{pool_name}] — hybrid retrieved {len(hits)} candidates",
            extra={"pool": pool_name, "n_candidates": len(hits)},
        )

        # For regimen/schedule queries, retrieve more candidates before reranking
        # so that markdown tables (which may have lower MaxSim) can be boosted by LOINC priority
        query_text = (intent.raw_query or "").lower()
        is_regimen_query = any(
            signal in query_text
            for signal in ["regimen", "schedule", "titrat", "week-by-week", "day-by-day", "escalation", "initiation"]
        )
        colbert_top_k = 20 if is_regimen_query else 10

        top_hits = self._apply_colbert_maxsim_reranking(q_vectors["colbert_tokens"], hits, top_k=colbert_top_k)
        top_hits = self._apply_loinc_section_priority(top_hits, intent)
        return top_hits

    @staticmethod
    def _generate_hyde_text(intent: PharmQueryIntent) -> str:
        """Generate a template-based hypothetical answer text for HyDE embedding.

        HyDE (Hypothetical Document Embeddings) bridges the vocabulary gap between
        a natural-language question and the imperative prose of drug-label dose sections.
        We use a template instead of an LLM call to keep latency low.
        """
        drug = " and ".join(intent.drug_names) if intent.drug_names else "this drug"
        pop  = intent.population_filter.replace("_", " ") if intent.population_filter else "adult"
        raw  = intent.raw_query
        return (
            f"Dosage adjustment for {drug} in {pop} patients: "
            f"reduce initial dose based on renal function and creatinine clearance. "
            f"Recommended dose for {pop} is lower than the standard adult dose. "
            f"Administer with caution; monitor closely. "
            f"Context: {raw}"
        )

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

    @staticmethod
    def _is_markdown_table(text: str) -> bool:
        """Check if text contains a markdown table format."""
        if not text:
            return False
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        # Check if lines have pipe separators
        has_pipes = all('|' in line for line in lines[:3] if line.strip())
        # Check for header separator line (e.g., "| --- | --- |")
        has_separator = any(
            set(line.replace(' ', '').replace('-', '').replace('|', '')) == set()
            for line in lines[:5]
        )
        return has_pipes and has_separator

    @staticmethod
    def _apply_loinc_section_priority(docs: list[dict], intent: PharmQueryIntent) -> list[dict]:
        """Apply semantic-based section prioritization using LOINC codes.

        For dosing queries, prioritize nodes from 'Dosage and Administration' (34068-7)
        over general 'indication' or 'pharmacology' sections. This enforces the regulatory
        hierarchy: dosing sections contain the authoritative, actionable dose guidance.

        For regimen/schedule queries, apply an additional boost to markdown tables which
        contain the structured titration schedules.

        Args:
            docs: Retrieved hits with LOINC metadata
            intent: Query intent including query_type and wants_dosing flag

        Returns:
            Reranked docs with LOINC-based boost applied
        """
        wants_warning = intent.wants_boxed_warning or intent.wants_warning_numerics
        if not intent.wants_dosing and not wants_warning:
            return docs

        DOSAGE_LOINC = "34068-7"
        POPULATIONS_LOINC = "34082-8"
        WARNINGS_LOINC = "34071-1"
        WARN_PREC_LOINC = "43685-7"
        OVERDOSAGE_LOINC = "34088-4"

        priority_map: dict[str, float] = {}

        if intent.wants_dosing:
            priority_map.update({
                DOSAGE_LOINC: 5.0,
                POPULATIONS_LOINC: 3.5,
                "dosing": 2.5,
            })

        if wants_warning:
            priority_map.update({
                WARNINGS_LOINC: max(priority_map.get(WARNINGS_LOINC, 0), 4.0),
                WARN_PREC_LOINC: max(priority_map.get(WARN_PREC_LOINC, 0), 3.0),
                OVERDOSAGE_LOINC: max(priority_map.get(OVERDOSAGE_LOINC, 0), 2.5),
                "warning": max(priority_map.get("warning", 0), 2.0),
            })

        # Check if this is a regimen/schedule/titration query
        query_text = (intent.raw_query or "").lower()
        is_regimen_query = any(
            signal in query_text
            for signal in ["regimen", "schedule", "titrat", "week-by-week", "day-by-day", "escalation", "initiation"]
        )

        is_concentration_query = any(
            signal in query_text
            for signal in ["concentration", "level", "therapeutic range", "target range", "serum level", "monitoring frequency"]
        )

        for doc in docs:
            source = doc.get("_source", {})
            loinc_code = source.get("smpc_section_code") or source.get("section_code")
            layout_type = source.get("layout_type")
            verbatim_text = source.get("verbatim_text", "")

            boost = priority_map.get(loinc_code, 0.0) or priority_map.get(layout_type, 0.0)
            original_maxsim = doc.get("_maxsim", 0.0)

            # CRITICAL FIX: For regimen queries, boost markdown tables significantly higher
            # Tables contain the structured titration schedules that answer "week 1, week 2" queries
            # This prevents extracted prose facts from outranking the actual schedule table
            if is_regimen_query and PharmaQueryEngine._is_markdown_table(verbatim_text):
                boost = max(boost, 4.5)  # Boost tables above all other content for schedule queries

            if is_concentration_query:
                text_lower = verbatim_text.lower()
                # Boost if chunk contains both serum/concentration AND a numeric range pattern
                has_concentration_keyword = any(kw in text_lower for kw in ["serum concentration", "serum level", "target", "therapeutic"])
                has_numeric_range = bool(re.search(r"\d+\.?\d*\s*(?:to|-|–)\s*\d+\.?\d*\s*(?:meq|mcg|mg|ng)", text_lower, re.IGNORECASE))
                # Also boost tables which often contain concentration parameters
                has_table_format = PharmaQueryEngine._is_markdown_table(verbatim_text) or "goal" in text_lower

                if has_concentration_keyword and (has_numeric_range or has_table_format):
                    boost = max(boost, 4.0)  # Strong boost for concentration ranges in monitoring queries

            # Apply multiplicative boost: preserve relative ranking within priority tiers
            if boost > 0:
                doc["_loinc_boost"] = boost
                doc["_priority_score"] = original_maxsim * boost
            else:
                doc["_loinc_boost"] = 1.0
                doc["_priority_score"] = original_maxsim

        # Re-sort: first by priority_score (LOINC-adjusted), then by original maxsim
        docs.sort(
            key=lambda x: (x.get("_priority_score", 0.0), x.get("_maxsim", 0.0)),
            reverse=True
        )
        return docs

    @staticmethod
    def _apply_clinical_reranking(hits: list[dict], intent: PharmQueryIntent) -> list[dict]:
        """Boost retrieval hits using clinical authority and metadata specificity."""
        has_interaction_evidence = any(
            (hit.get("_source", {}) or {}).get("layout_type") == "interaction"
            for hit in hits
        )

        for hit in hits:
            score = float(hit.get("_priority_score", hit.get("_maxsim", 0.0)) or 0.0)
            source = hit.get("_source", {})
            layout_type = source.get("layout_type")

            if intent.preferred_layout == layout_type:
                score *= CLINICAL_LAYOUT_MATCH_BOOST

            if intent.population_filter and intent.population_filter == source.get("patient_population"):
                score *= CLINICAL_POPULATION_MATCH_BOOST

            if source.get("dose_values") or source.get("dose_val") is not None:
                score *= CLINICAL_DOSE_METADATA_BOOST

            if has_interaction_evidence and intent.wants_interaction and len(intent.drug_names) >= 2:
                if layout_type == "interaction":
                    score *= INTERACTION_LAYOUT_OVERRIDE_BOOST
                elif layout_type in {"dosing", "structured_fact"}:
                    score *= INTERACTION_DOSING_PENALTY

            if intent.wants_boxed_warning and source.get("boxed_warning"):
                score *= 2.0

            hit["_clinical_score"] = score

        hits.sort(
            key=lambda x: (x.get("_clinical_score", 0.0), x.get("_maxsim", 0.0)),
            reverse=True,
        )
        return hits

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
            maxsim_score=hit.get("_clinical_score", hit.get("_maxsim", 0.0)),
        )
        if hit.get("_is_parent_pivot"):
            node.__dict__["_is_parent_pivot"] = True
        if hit.get("_is_atc_neighbor"):
            node.__dict__["_is_atc_neighbor"] = True
        if hit.get("_cross_label"):
            node.__dict__["_cross_label"] = True
        return node

    def _enforce_verbatim_constraints(self, node: RetrievedNode) -> RetrievedNode:
        # structured_fact nodes are pre-extracted dose sentences; always verbatim-lock them.
        if node.layout_type == "structured_fact":
            node.verbatim_locked = True
        threshold = CFG.ingestion.raglens_verbatim_threshold
        if node.raglens_risk >= threshold:
            node.verbatim_locked = True
            log.info(
                f"RAGLens LOCK: {node.urn[-45:]} "
                f"(risk={node.raglens_risk:.2f} >= {threshold})",
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
                primary_hits = self._execute_parallel_retrieval(
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