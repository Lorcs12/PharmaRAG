from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass

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

MAX_REFLECTION_ROUNDS: int = 2
MIN_ACCEPTABLE_HITS: int = 2
MIN_MAXSIM_FLOOR: float = 0.40
POPULATION_RELAX_THRESHOLD: int = 1

_FALLBACK_LAYOUT_EXPANSION: dict[str, list[str]] = {
    "dosing": ["warning", "indication"],
    "contraindication": ["warning", "interaction"],
    "interaction": ["warning", "pharmacology"],
    "warning": ["indication", "pharmacology"],
    "indication": ["dosing", "warning"],
    "pharmacology": ["indication", "warning"],
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
}


@dataclass
class SufficiencyReport:
    passed: bool
    round_number: int
    n_hits: int
    top_maxsim: float
    layout_coverage: dict[str, int]
    failures: list[str]
    intent_snapshot: dict


class ReflectivePharmaQueryEngine(PharmaQueryEngine):
    def _sufficiency_gate(
        self,
        hits: list[dict],
        intent: PharmQueryIntent,
        round_number: int,
    ) -> SufficiencyReport:
        failures: list[str] = []
        layout_coverage: dict[str, int] = {}

        for hit in hits:
            layout_type = hit.get("_source", {}).get("layout_type", "unknown")
            layout_coverage[layout_type] = layout_coverage.get(layout_type, 0) + 1

        top_maxsim = max((hit.get("_maxsim", 0.0) for hit in hits), default=0.0)

        if len(hits) < MIN_ACCEPTABLE_HITS:
            failures.append(
                f"F1:too_few_hits (got {len(hits)}, need >= {MIN_ACCEPTABLE_HITS})"
            )

        if hits and top_maxsim < MIN_MAXSIM_FLOOR:
            failures.append(
                f"F2:low_maxsim (top={top_maxsim:.3f}, floor={MIN_MAXSIM_FLOOR})"
            )

        if intent.wants_dosing and layout_coverage.get("dosing", 0) == 0:
            failures.append("F3:no_dosing_nodes (wants_dosing=True but 0 dosing hits)")

        if intent.population_filter and len(hits) <= POPULATION_RELAX_THRESHOLD:
            failures.append(
                f"F4:population_filter_too_narrow (filter='{intent.population_filter}', hits={len(hits)})"
            )

        if (
            intent.wants_interaction
            and intent.query_type == QUERY_TYPE_CAUSAL
            and layout_coverage.get("interaction", 0) == 0
        ):
            failures.append("F5:no_interaction_nodes (causal query, 0 interaction hits)")

        report = SufficiencyReport(
            passed=not failures,
            round_number=round_number,
            n_hits=len(hits),
            top_maxsim=top_maxsim,
            layout_coverage=layout_coverage,
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
            f"hits={len(hits)} top_maxsim={top_maxsim:.3f} failures={failures}",
            extra={
                "round": round_number,
                "passed": report.passed,
                "n_hits": len(hits),
                "top_maxsim": top_maxsim,
                "failures": failures,
            },
        )
        return report

    def _rewrite_intent(
        self,
        intent: PharmQueryIntent,
        report: SufficiencyReport,
    ) -> tuple[PharmQueryIntent, bool]:
        new_intent = copy.deepcopy(intent)
        query_text_changed = False
        failure_codes = {failure.split(":")[0] for failure in report.failures}

        if "F4" in failure_codes and new_intent.population_filter:
            old_population = new_intent.population_filter
            new_intent.population_filter = None
            log.info(
                f"[IntentRewrite] Dropped population filter '{old_population}' -> None",
                extra={"rewrite": "drop_population_filter", "old": old_population},
            )

        if "F3" in failure_codes and "dosing" not in new_intent.layout_filters:
            new_intent.layout_filters.insert(0, "dosing")
            new_intent.preferred_layout = "dosing"
            new_intent.wants_dosing = True
            log.info(
                "[IntentRewrite] Injected 'dosing' into layout_filters",
                extra={"rewrite": "inject_dosing_pool"},
            )

        if "F5" in failure_codes:
            added_pools = []
            for pool in ("interaction", "pharmacology"):
                if pool not in new_intent.layout_filters:
                    new_intent.layout_filters.append(pool)
                    added_pools.append(pool)
            new_intent.wants_interaction = True
            new_intent.wants_mechanism = True
            if added_pools:
                log.info(
                    f"[IntentRewrite] Injected fallback causal pools: {added_pools}",
                    extra={"rewrite": "inject_interaction_pool", "added": added_pools},
                )

        if "F1" in failure_codes or "F2" in failure_codes:
            expanded_query = new_intent.raw_query
            for pattern, replacement in _ABBREV_EXPANSION.items():
                rewritten_query = re.sub(pattern, replacement, expanded_query, flags=re.IGNORECASE)
                if rewritten_query != expanded_query:
                    expanded_query = rewritten_query
                    query_text_changed = True

            if query_text_changed:
                new_intent.raw_query = expanded_query
                log.info(
                    "[IntentRewrite] Expanded abbreviations in query text",
                    extra={"rewrite": "abbrev_expansion", "new_query": expanded_query[:120]},
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
                    f"[IntentRewrite] Added fallback pools: {added_pools}",
                    extra={"rewrite": "fallback_pool_expansion", "added": added_pools},
                )

        return new_intent, query_text_changed

    def execute_query_pipeline(self, query: str) -> CogCanvasArtifact:
        start = time.perf_counter()

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
                    round_hits = self._execute_hybrid_retrieval(intent, q_vectors)

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

                intent, text_changed = self._rewrite_intent(intent, report)
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
    "POPULATION_RELAX_THRESHOLD",
    "ReflectivePharmaQueryEngine",
    "SufficiencyReport",
    "QUERY_TYPE_FACTUAL",
    "QUERY_TYPE_CAUSAL",
    "QUERY_TYPE_COMPARATIVE",
    "QUERY_TYPE_STRATEGIC",
]