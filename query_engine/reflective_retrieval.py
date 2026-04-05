from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import CFG
from embedder import embed_document
from logger import Timer, get_logger

from .constants import (
    QUERY_TYPE_CAUSAL,
    QUERY_TYPE_COMPARATIVE,
    QUERY_TYPE_FACTUAL,
    QUERY_TYPE_STRATEGIC,
)
from .cognitive_canvas import CogCanvasArtifact
from .models import PharmQueryIntent
from .query_pipeline import PharmaQueryEngine

log = get_logger("reflective_retrieval", CFG.log.file, CFG.log.level)

MAX_REFLECTION_ROUNDS: int = 3
MIN_ACCEPTABLE_HITS: int = 3
MIN_MAXSIM_FLOOR: float = 0.38
MIN_DOSE_EVIDENCE_NODES: int = 2
LABEL_RECENCY_YEARS: int = 3
POPULATION_RELAX_THRESHOLD: int = 1
NUMERIC_TITRATION_POPULATIONS: set[str] = {
    "renal_impairment",
    "hepatic_impairment",
    "pediatric",
    "elderly",
}

_FALLBACK_LAYOUT_EXPANSION: dict[str, list[str]] = {
    "dosing": ["warning", "indication", "pharmacology"],
    "contraindication": ["warning", "interaction"],
    "interaction": ["warning", "pharmacology"],
    "warning": ["indication", "pharmacology"],
    "indication": ["dosing", "warning"],
    "pharmacology": ["indication", "warning"],
    "structured_fact": ["dosing", "warning"],
}

_ABBREV_EXPANSION: dict[str, str] = {
    r"\bckd\b": "chronic kidney disease renal impairment",
    r"\bhf\b": "heart failure",
    r"\bmi\b": "myocardial infarction",
    r"\bafib\b": "atrial fibrillation",
    r"\bdm\b": "diabetes mellitus",
    r"\bhtn\b": "hypertension",
    r"\bpeds\b": "pediatric children",
    r"\bpo\b": "oral by mouth",
    r"\biv\b": "intravenous",
    r"\bsc\b": "subcutaneous",
    r"\btid\b": "three times daily",
    r"\bbid\b": "twice daily",
    r"\bqd\b": "once daily",
    r"\bprn\b": "as needed",
    r"\bmax\b": "maximum",
    r"\bmin\b": "minimum",
    r"\bped\b": "pediatric",
    r"\begfr\b": "estimated glomerular filtration rate renal function",
    r"\bgfr\b": "glomerular filtration rate",
    r"\bnsaid\b": "non-steroidal anti-inflammatory drug",
    r"\bssri\b": "selective serotonin reuptake inhibitor",
    r"\bmaoi\b": "monoamine oxidase inhibitor",
    r"\bace\b": "angiotensin converting enzyme inhibitor",
    r"\barb\b": "angiotensin receptor blocker",
}


class GateFailure(str, Enum):
    TOO_FEW_HITS = "F1"
    LOW_MAXSIM = "F2"
    NO_DOSING_NODES = "F3"
    POPULATION_TOO_NARROW = "F4"
    NO_INTERACTION_NODES = "F5"
    NO_DOSE_EVIDENCE = "F6"
    STALE_LABELS = "F7"


@dataclass
class SufficiencyReport:
    passed: bool
    round_number: int
    n_hits: int
    top_maxsim: float
    layout_coverage: dict[str, int]
    dose_evidence_nodes: int
    newest_label_year: Optional[int]
    failures: list[GateFailure]
    intent_snapshot: dict

    @property
    def failure_codes(self) -> set[GateFailure]:
        return set(self.failures)


class ReflectivePharmaQueryEngine(PharmaQueryEngine):
    def _sufficiency_gate(
        self,
        hits: list[dict],
        intent: PharmQueryIntent,
        round_number: int,
    ) -> SufficiencyReport:
        failures: list[GateFailure] = []
        layout_coverage: dict[str, int] = {}
        newest_label_year: Optional[int] = None

        for hit in hits:
            source = hit.get("_source", {})
            layout_type = source.get("layout_type", "unknown")
            layout_coverage[layout_type] = layout_coverage.get(layout_type, 0) + 1

            date_str = source.get("label_version_date", "")
            if date_str and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                except ValueError:
                    continue
                if newest_label_year is None or year > newest_label_year:
                    newest_label_year = year

        top_maxsim = max((hit.get("_maxsim", 0.0) for hit in hits), default=0.0)
        dose_evidence_nodes = sum(
            1
            for hit in hits
            if hit.get("_source", {}).get("dose_values")
            or hit.get("_source", {}).get("dose_val") is not None
        )

        if len(hits) < MIN_ACCEPTABLE_HITS:
            failures.append(GateFailure.TOO_FEW_HITS)

        if hits and top_maxsim < MIN_MAXSIM_FLOOR:
            failures.append(GateFailure.LOW_MAXSIM)

        if (
            intent.wants_dosing
            and layout_coverage.get("dosing", 0) == 0
            and layout_coverage.get("structured_fact", 0) == 0
        ):
            failures.append(GateFailure.NO_DOSING_NODES)

        requires_numeric_titration = (
            intent.wants_dosing
            and intent.population_filter in NUMERIC_TITRATION_POPULATIONS
        )

        if requires_numeric_titration and dose_evidence_nodes < MIN_DOSE_EVIDENCE_NODES:
            failures.append(GateFailure.NO_DOSE_EVIDENCE)

        if intent.population_filter and len(hits) <= POPULATION_RELAX_THRESHOLD:
            failures.append(GateFailure.POPULATION_TOO_NARROW)

        if (
            intent.wants_interaction
            and intent.query_type == QUERY_TYPE_CAUSAL
            and layout_coverage.get("interaction", 0) == 0
        ):
            failures.append(GateFailure.NO_INTERACTION_NODES)

        current_year = time.gmtime().tm_year
        if newest_label_year and (current_year - newest_label_year) > LABEL_RECENCY_YEARS:
            failures.append(GateFailure.STALE_LABELS)

        report = SufficiencyReport(
            passed=not failures,
            round_number=round_number,
            n_hits=len(hits),
            top_maxsim=top_maxsim,
            layout_coverage=layout_coverage,
            dose_evidence_nodes=dose_evidence_nodes,
            newest_label_year=newest_label_year,
            failures=failures,
            intent_snapshot={
                "query_type": intent.query_type,
                "layout_filters": list(intent.layout_filters),
                "population_filter": intent.population_filter,
                "wants_dosing": intent.wants_dosing,
                "wants_interaction": intent.wants_interaction,
                "drug_names": list(intent.drug_names),
            },
        )

        log.info(
            f"[SufficiencyGate] round={round_number} passed={report.passed} "
            f"hits={len(hits)} top_maxsim={top_maxsim:.3f} "
            f"dose_evidence={dose_evidence_nodes} newest_year={newest_label_year} "
            f"failures={[failure.value for failure in failures]}",
            extra={
                "round": round_number,
                "passed": report.passed,
                "n_hits": len(hits),
                "top_maxsim": top_maxsim,
                "dose_evidence_nodes": dose_evidence_nodes,
                "newest_label_year": newest_label_year,
                "failures": [failure.value for failure in failures],
            },
        )
        return report

    def _rewrite_intent(
        self,
        intent: PharmQueryIntent,
        report: SufficiencyReport,
        hits: list[dict],
        q_vectors: dict,
        round_number: int,
    ) -> tuple[PharmQueryIntent, list[dict], bool]:
        new_intent = copy.deepcopy(intent)
        extra_hits: list[dict] = []
        query_text_changed = False
        failure_codes = report.failure_codes

        if GateFailure.POPULATION_TOO_NARROW in failure_codes and new_intent.population_filter:
            old_population = new_intent.population_filter
            new_intent.population_filter = None
            log.info(
                f"[Rewrite R{round_number}] Dropped population filter '{old_population}'",
                extra={"rewrite": "drop_population_filter", "old": old_population, "round": round_number},
            )

        if GateFailure.NO_DOSING_NODES in failure_codes and "dosing" not in new_intent.layout_filters:
            new_intent.layout_filters.insert(0, "dosing")
            new_intent.preferred_layout = "dosing"
            new_intent.wants_dosing = True
            log.info(
                f"[Rewrite R{round_number}] Injected 'dosing' pool",
                extra={"rewrite": "inject_dosing_pool", "round": round_number},
            )

        if (
            GateFailure.NO_DOSE_EVIDENCE in failure_codes
            or GateFailure.NO_DOSING_NODES in failure_codes
        ):
            parent_hits = self._fetch_parent_nodes(hits, max_parents=5)
            extra_hits.extend(parent_hits)
            log.info(
                f"[Rewrite R{round_number}] Parent pivot added {len(parent_hits)} nodes",
                extra={"rewrite": "parent_pivot", "round": round_number, "added": len(parent_hits)},
            )

        if (
            GateFailure.NO_DOSE_EVIDENCE in failure_codes
            or GateFailure.TOO_FEW_HITS in failure_codes
        ) and round_number >= 2:
            atc_codes = [
                hit.get("_source", {}).get("atc_code", "")
                for hit in hits
                if hit.get("_source", {}).get("atc_code")
            ]
            neighbor_hits = self._fetch_atc_neighbors(
                atc_codes,
                layout_type=new_intent.preferred_layout,
                q_vectors=q_vectors,
                top_k=5,
            )
            extra_hits.extend(neighbor_hits)
            log.info(
                f"[Rewrite R{round_number}] ATC neighbor added {len(neighbor_hits)} nodes",
                extra={"rewrite": "atc_neighbor", "round": round_number, "added": len(neighbor_hits)},
            )

        if GateFailure.NO_INTERACTION_NODES in failure_codes:
            added_pools = []
            for pool in ("interaction", "pharmacology"):
                if pool not in new_intent.layout_filters:
                    new_intent.layout_filters.append(pool)
                    added_pools.append(pool)
            new_intent.wants_interaction = True
            new_intent.wants_mechanism = True
            if added_pools:
                log.info(
                    f"[Rewrite R{round_number}] Injected fallback causal pools: {added_pools}",
                    extra={"rewrite": "inject_interaction_pool", "added": added_pools, "round": round_number},
                )

        if GateFailure.TOO_FEW_HITS in failure_codes or GateFailure.LOW_MAXSIM in failure_codes:
            expanded_query = new_intent.raw_query
            for pattern, replacement in _ABBREV_EXPANSION.items():
                rewritten_query = re.sub(pattern, replacement, expanded_query, flags=re.IGNORECASE)
                if rewritten_query != expanded_query:
                    expanded_query = rewritten_query
                    query_text_changed = True

            if query_text_changed:
                new_intent.raw_query = expanded_query
                log.info(
                    f"[Rewrite R{round_number}] Expanded abbreviations in query text",
                    extra={"rewrite": "abbrev_expansion", "new_query": expanded_query[:120], "round": round_number},
                )

            primary_pool = new_intent.preferred_layout
            fallback_pools = _FALLBACK_LAYOUT_EXPANSION.get(primary_pool, ["warning"])
            added_pools = []
            for pool in fallback_pools:
                if pool not in new_intent.layout_filters:
                    new_intent.layout_filters.append(pool)
                    added_pools.append(pool)
            if added_pools:
                log.info(
                    f"[Rewrite R{round_number}] Added fallback pools: {added_pools}",
                    extra={"rewrite": "fallback_pool_expansion", "added": added_pools, "round": round_number},
                )

        if GateFailure.STALE_LABELS in failure_codes:
            new_intent.label_version_gte = "2010-01-01"
            log.info(
                f"[Rewrite R{round_number}] Relaxed date constraints for stale label recovery",
                extra={"rewrite": "relax_date_filter", "round": round_number},
            )

        return new_intent, extra_hits, query_text_changed

    def execute_query_pipeline(self, query: str) -> CogCanvasArtifact:
        start = time.perf_counter()
        explicit_year_filter = bool(re.search(r"\b(20\d{2})\b", query))

        with Timer(log, "reflective_query_pipeline", query=query[:60]):
            intent = self._decompose_semantic_intent(query)

            with Timer(log, "embed_query"):
                q_vectors = embed_document(intent.raw_query)

            all_hits: list[dict] = []
            sufficiency_reports: list[SufficiencyReport] = []

            for round_number in range(1, MAX_REFLECTION_ROUNDS + 1):
                is_final_round = round_number == MAX_REFLECTION_ROUNDS
                log.info(
                    f"[ReflectiveLoop] Starting round {round_number}/{MAX_REFLECTION_ROUNDS}",
                    extra={"round": round_number, "query": intent.raw_query[:60]},
                )

                with Timer(log, f"retrieve_round_{round_number}"):
                    round_hits = self._execute_hybrid_retrieval(
                        intent,
                        q_vectors,
                        hard_date_filter=(round_number == 1 and explicit_year_filter),
                    )

                existing_urns = {hit.get("_source", {}).get("urn_id") for hit in all_hits}
                for hit in round_hits:
                    urn = hit.get("_source", {}).get("urn_id")
                    if urn and urn not in existing_urns:
                        all_hits.append(hit)
                        existing_urns.add(urn)

                report = self._sufficiency_gate(all_hits, intent, round_number)
                sufficiency_reports.append(report)

                if report.passed:
                    log.info(
                        f"[ReflectiveLoop] Gate PASSED on round {round_number} with {len(all_hits)} hits",
                        extra={"round": round_number, "n_hits": len(all_hits)},
                    )
                    break

                if is_final_round:
                    log.warning(
                        f"[ReflectiveLoop] Gate FAILED on final round {round_number}; assembling with {len(all_hits)} hits. Failures: {report.failures}",
                        extra={"round": round_number, "n_hits": len(all_hits), "failures": report.failures},
                    )
                    break

                intent, extra_hits, text_changed = self._rewrite_intent(
                    intent,
                    report,
                    all_hits,
                    q_vectors,
                    round_number,
                )
                for hit in extra_hits:
                    urn = hit.get("_source", {}).get("urn_id")
                    if urn and urn not in existing_urns:
                        all_hits.append(hit)
                        existing_urns.add(urn)
                if text_changed:
                    log.info(
                        "[ReflectiveLoop] Query text changed; re-embedding",
                        extra={"round": round_number},
                    )
                    with Timer(log, f"re_embed_round_{round_number}"):
                        q_vectors = embed_document(intent.raw_query)

            if not all_hits:
                log.warning(
                    "No results found after all reflection rounds — returning empty artifact",
                    extra={"query": query},
                )
                return CogCanvasArtifact(
                    query=query,
                    intent=intent,
                    verbatim_nodes=[],
                    paraphrase_nodes=[],
                    causal_context=[],
                    macro_context=[],
                    table_references=[],
                    total_latency_ms=(time.perf_counter() - start) * 1000,
                )

            with Timer(log, "assembly"):
                artifact = self._assemble_cognitive_artifact(intent, all_hits, start)

            artifact.__dict__["_reflection_rounds"] = len(sufficiency_reports)
            artifact.__dict__["_sufficiency_reports"] = sufficiency_reports
            artifact.__dict__["_dose_evidence_nodes"] = sufficiency_reports[-1].dose_evidence_nodes

            log.info(
                f"[ReflectiveLoop] Complete — {len(sufficiency_reports)} round(s), {len(all_hits)} total hits, {artifact.total_latency_ms:.0f}ms",
                extra={
                    "rounds": len(sufficiency_reports),
                    "total_hits": len(all_hits),
                    "latency_ms": artifact.total_latency_ms,
                },
            )
            return artifact

    def execute(self, query: str) -> CogCanvasArtifact:
        return self.execute_query_pipeline(query)


__all__ = [
    "MAX_REFLECTION_ROUNDS",
    "MIN_ACCEPTABLE_HITS",
    "MIN_MAXSIM_FLOOR",
    "MIN_DOSE_EVIDENCE_NODES",
    "LABEL_RECENCY_YEARS",
    "POPULATION_RELAX_THRESHOLD",
    "GateFailure",
    "ReflectivePharmaQueryEngine",
    "SufficiencyReport",
    "QUERY_TYPE_FACTUAL",
    "QUERY_TYPE_CAUSAL",
    "QUERY_TYPE_COMPARATIVE",
    "QUERY_TYPE_STRATEGIC",
]