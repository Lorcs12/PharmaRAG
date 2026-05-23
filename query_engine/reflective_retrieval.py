from __future__ import annotations

import copy
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from config import CFG
from embedder import embed_document
from logger import Timer, get_logger

from .constants import (
    QUERY_TYPE_CAUSAL, QUERY_TYPE_COMPARATIVE, QUERY_TYPE_FACTUAL, QUERY_TYPE_STRATEGIC,
    LABEL_RECENCY_YEARS,
    MAX_REFLECTION_ROUNDS, MIN_ACCEPTABLE_HITS, MIN_MAXSIM_FLOOR,
    MIN_DOSE_EVIDENCE_NODES, MIN_TARGET_DRUG_HITS, SATURATION_DELTA_PCT,
    HIGH_CONF_FIXED_DOSE_MAXSIM, POPULATION_RELAX_THRESHOLD,
    NUMERIC_TITRATION_POPULATIONS,
    FALLBACK_LAYOUT_EXPANSION, ABBREV_EXPANSION,
    FIXED_DOSE_STATEMENT_RE, ABSENCE_SEEKING_RE, EXPLICIT_DOSING_INTENT_RE,
    MONITORING_FOCUSED_RE, EXPLICIT_REGIMEN_RE,
    LOINC_DIVERSITY_MATRIX, BROAD_INTENT_RE,
    SEQ_TITRATION_RE, INIT_DOSE_RE, MAX_DOSE_RE,
    NUMERIC_DOSE_SIGNAL_RE,
)
from .cognitive_canvas import CogCanvasArtifact
from .models import PharmQueryIntent
from .query_pipeline import PharmaQueryEngine

log = get_logger("reflective_retrieval", CFG.log.file, CFG.log.level)


def _f8_metadata_lockdown_enabled() -> bool:
    raw = (os.getenv("F8_METADATA_LOCKDOWN", "1") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class GateFailure(str, Enum):
    TOO_FEW_HITS = "F1"
    LOW_MAXSIM = "F2"
    NO_DOSING_NODES = "F3"
    POPULATION_TOO_NARROW = "F4"
    NO_INTERACTION_NODES = "F5"
    NO_DOSE_EVIDENCE = "F6"
    STALE_LABELS = "F7"
    LOW_TARGET_DRUG_COVERAGE = "F8"
    INCOMPLETE_DOSING_LIFECYCLE = "F9"
    LOW_LOINC_DIVERSITY = "F10"


@dataclass
class SufficiencyReport:
    passed: bool
    round_number: int
    n_hits: int
    top_maxsim: float
    layout_coverage: dict[str, int]
    dose_evidence_nodes: int
    newest_label_year: int | None
    failures: list[GateFailure]
    intent_snapshot: dict

    @property
    def failure_codes(self) -> set[GateFailure]:
        return set(self.failures)


class ReflectivePharmaQueryEngine(PharmaQueryEngine):
    @staticmethod
    def _is_fixed_dose_statement(text: str) -> bool:
        return bool(FIXED_DOSE_STATEMENT_RE.search(text or ""))

    @staticmethod
    def _extend_unique_hits(target_hits: list[dict], new_hits: list[dict]) -> None:
        existing_urns = {hit.get("_source", {}).get("urn_id") for hit in target_hits}
        for hit in new_hits:
            urn = hit.get("_source", {}).get("urn_id")
            if urn and urn not in existing_urns:
                target_hits.append(hit)
                existing_urns.add(urn)

    def _should_stop_reflection(
        self,
        *,
        round_number: int,
        is_final_round: bool,
        report: SufficiencyReport,
        all_hits: list[dict],
        intent: PharmQueryIntent,
        last_best_clinical_score: float,
        current_best_clinical_score: float,
    ) -> bool:
        if round_number == 1 and self._has_high_confidence_fixed_dose_evidence(all_hits, intent):
            log.info(
                "[ReflectiveLoop] Evidence-gated early exit: high-confidence fixed-dose evidence found in round 1",
                extra={"round": round_number, "top_maxsim": report.top_maxsim},
            )
            return True

        unresolved_critical = {
            GateFailure.NO_DOSING_NODES,
            GateFailure.NO_DOSE_EVIDENCE,
            GateFailure.INCOMPLETE_DOSING_LIFECYCLE,
            GateFailure.LOW_LOINC_DIVERSITY,
            GateFailure.LOW_TARGET_DRUG_COVERAGE,
        }
        has_unresolved_critical = bool(report.failure_codes & unresolved_critical)

        if round_number > 1 and last_best_clinical_score > 0:
            relative_gain = (current_best_clinical_score - last_best_clinical_score) / last_best_clinical_score
            if relative_gain <= SATURATION_DELTA_PCT and not has_unresolved_critical:
                log.info(
                    f"[ReflectiveLoop] Knowledge saturated on round {round_number}: "
                    f"clinical_score_gain={relative_gain:.4f} <= {SATURATION_DELTA_PCT:.4f}",
                    extra={
                        "round": round_number,
                        "relative_gain": relative_gain,
                        "last_best_clinical_score": last_best_clinical_score,
                        "current_best_clinical_score": current_best_clinical_score,
                    },
                )
                return True

        if report.passed:
            log.info(
                f"[ReflectiveLoop] Gate PASSED on round {round_number} with {len(all_hits)} hits",
                extra={"round": round_number, "n_hits": len(all_hits)},
            )
            return True

        if is_final_round:
            log.warning(
                f"[ReflectiveLoop] Gate FAILED on final round {round_number}; assembling with {len(all_hits)} hits. Failures: {report.failures}",
                extra={"round": round_number, "n_hits": len(all_hits), "failures": report.failures},
            )
            return True

        return False

    def _has_high_confidence_fixed_dose_evidence(self, hits: list[dict], intent: PharmQueryIntent) -> bool:
        if not (intent.wants_dosing and intent.query_type == QUERY_TYPE_FACTUAL and not intent.wants_interaction):
            return False

        if EXPLICIT_REGIMEN_RE.search(intent.raw_query or ""):
            return False

        best_match = 0.0
        for hit in hits:
            source = hit.get("_source", {})
            text = source.get("verbatim_text", "")
            if not self._is_fixed_dose_statement(text):
                continue
            best_match = max(best_match, float(hit.get("_maxsim", 0.0) or 0.0))
        return best_match >= HIGH_CONF_FIXED_DOSE_MAXSIM

    @staticmethod
    def _best_clinical_score(hits: list[dict]) -> float:
        return max(
            (float(h.get("_clinical_score", h.get("_maxsim", 0.0)) or 0.0) for h in hits),
            default=0.0,
        )

    @staticmethod
    def _normalize_drug_text(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip().lower())

    @staticmethod
    def _contains_numeric_dose_signal(text: str) -> bool:
        return bool(NUMERIC_DOSE_SIGNAL_RE.search(text or ""))

    @staticmethod
    def _is_absence_seeking_query(text: str) -> bool:
        return bool(ABSENCE_SEEKING_RE.search(text or ""))

    @staticmethod
    def _is_explicit_dosing_intent_query(text: str) -> bool:
        q = text or ""
        if not EXPLICIT_DOSING_INTENT_RE.search(q):
            return False
        if MONITORING_FOCUSED_RE.search(q) and not re.search(r"\b(?:dose|dosing|dosage|titrat)\b", q, re.IGNORECASE):
            return False
        return True

    def _hit_matches_target_drug(self, hit: dict, target_drugs: list[str]) -> bool:
        if not target_drugs:
            return True

        source = hit.get("_source", {})
        generic = self._normalize_drug_text(str(source.get("drug_name_generic", "")))
        brands_raw = source.get("drug_name_brand") or []
        if isinstance(brands_raw, str):
            brands = [self._normalize_drug_text(brands_raw)]
        else:
            brands = [self._normalize_drug_text(str(b)) for b in brands_raw]

        haystacks = [generic] + brands
        for target in target_drugs:
            t = self._normalize_drug_text(target)
            if not t:
                continue
            if any(t in h or h in t for h in haystacks if h):
                return True
        return False

    def _filter_to_target_drug_hits_if_sufficient(self, hits: list[dict], intent: PharmQueryIntent) -> list[dict]:
        if not intent.drug_names:
            return hits

        matching = [h for h in hits if self._hit_matches_target_drug(h, intent.drug_names)]

        if intent.population_filter and intent.wants_dosing and len(matching) >= 1:
            threshold = 1

        if len(matching) >= threshold:
            return matching

        penalized: list[dict] = []
        for h in hits:
            if not self._hit_matches_target_drug(h, intent.drug_names):
                h = {**h}
                h["_cross_label"] = True
            penalized.append(h)
        penalized.sort(
            key=lambda x: (
                not self._hit_matches_target_drug(x, intent.drug_names),
                -x.get("_maxsim", 0.0),
            )
        )
        return penalized

    @staticmethod
    def _evaluate_dosing_coverage(hits: list[dict]) -> tuple[bool, str, dict]:
        protocol_requirements: dict[str, dict] = {
            "starting_dose":        {"found": False},
            "sequential_titration": {"found": False},
            "maximum_dose":         {"found": False},
        }

        for hit in hits:
            chunk_text = (hit.get("_source", {}).get("verbatim_text", "") or "")
            if not chunk_text:
                continue
            if INIT_DOSE_RE.search(chunk_text):
                protocol_requirements["starting_dose"]["found"] = True
            if SEQ_TITRATION_RE.search(chunk_text):
                protocol_requirements["sequential_titration"]["found"] = True
            if MAX_DOSE_RE.search(chunk_text):
                protocol_requirements["maximum_dose"]["found"] = True

        missing_components = [k for k, v in protocol_requirements.items() if not v["found"]]
       
        found_count = sum(1 for v in protocol_requirements.values() if v["found"])
        is_complete = found_count >= 1

        diagnostic = (
            f"Protocol coverage: starting_dose={protocol_requirements['starting_dose']['found']}, "
            f"sequential_titration={protocol_requirements['sequential_titration']['found']}, "
            f"max_dose={protocol_requirements['maximum_dose']['found']}"
        )
        if missing_components:
            diagnostic += f" | Missing: {', '.join(missing_components)}"

        return is_complete, diagnostic, protocol_requirements

    @staticmethod
    def _evaluate_loinc_diversity(
        hits: list[dict],
        intent: PharmQueryIntent,
    ) -> tuple[bool, int, int, list[str]]:
        
        complexity = "broad" if BROAD_INTENT_RE.search(intent.raw_query or "") else "narrow"
        required = LOINC_DIVERSITY_MATRIX.get((intent.query_type, complexity), 1)

        loinc_codes: set[str] = set()
        for hit in hits:
            src = hit.get("_source", {})
            code = (
                src.get("smpc_section_code")
                or src.get("section_code")
                or src.get("loinc_code")
                or ""
            )
            if code:
                loinc_codes.add(code)

            elif src.get("layout_type") == "structured_fact":
                loinc_codes.add("structured_fact")

        distinct = len(loinc_codes)
        is_sufficient = distinct >= required
        return is_sufficient, distinct, required, sorted(loinc_codes)

    def _sufficiency_gate(
        self,
        hits: list[dict],
        intent: PharmQueryIntent,
        round_number: int,
    ) -> SufficiencyReport:
        failures: list[GateFailure] = []
        layout_coverage: dict[str, int] = {}
        newest_label_year: int | None = None

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
        numeric_dose_signal_nodes = sum(
            1
            for hit in hits
            if self._contains_numeric_dose_signal(hit.get("_source", {}).get("verbatim_text", ""))
        )
        target_drug_hits = sum(
            1
            for hit in hits
            if self._hit_matches_target_drug(hit, intent.drug_names)
        )

        if len(hits) < MIN_ACCEPTABLE_HITS:
            failures.append(GateFailure.TOO_FEW_HITS)

        if hits and top_maxsim < MIN_MAXSIM_FLOOR:
            failures.append(GateFailure.LOW_MAXSIM)

        dosing_required = (
            intent.wants_dosing
            and intent.query_type == QUERY_TYPE_FACTUAL
            and not intent.wants_interaction
        )
        explicit_dosing_intent = dosing_required and self._is_explicit_dosing_intent_query(intent.raw_query)

        if (
            explicit_dosing_intent
            and layout_coverage.get("dosing", 0) == 0
            and layout_coverage.get("structured_fact", 0) == 0
        ):
            failures.append(GateFailure.NO_DOSING_NODES)

        requires_numeric_titration = (
            explicit_dosing_intent
            and intent.population_filter in NUMERIC_TITRATION_POPULATIONS
        )

        if requires_numeric_titration and max(dose_evidence_nodes, numeric_dose_signal_nodes) < MIN_DOSE_EVIDENCE_NODES:
            failures.append(GateFailure.NO_DOSE_EVIDENCE)

        if intent.query_type == QUERY_TYPE_FACTUAL and intent.drug_names:
            required_target_hits = min(max(1, len(intent.drug_names)), MIN_TARGET_DRUG_HITS)
            if target_drug_hits < required_target_hits:
                failures.append(GateFailure.LOW_TARGET_DRUG_COVERAGE)

        if intent.population_filter and len(hits) <= POPULATION_RELAX_THRESHOLD:
            failures.append(GateFailure.POPULATION_TOO_NARROW)

        _regimen_signals = ("regimen", "schedule", "titrat", "week-by-week", "day 1", "starting dose", "initial dose", "maximum dose")
        is_regimen_query = (
            dosing_required
            and any(sig in intent.raw_query.lower() for sig in _regimen_signals)
        )
        if is_regimen_query:
            protocol_complete, protocol_diagnostic, protocol_coverage = self._evaluate_dosing_coverage(hits)
            if not protocol_complete:
                log.info(
                    f"[Dosing Lifecycle Audit] {protocol_diagnostic}",
                    extra={"coverage": protocol_coverage, "round": round_number}
                )
                failures.append(GateFailure.INCOMPLETE_DOSING_LIFECYCLE)

        loinc_ok, loinc_distinct, loinc_required, loinc_codes = self._evaluate_loinc_diversity(
            hits, intent
        )
        if not loinc_ok:
            log.info(
                f"[LOINC Diversity] Fragment Capture detected: {loinc_distinct} distinct "
                f"section codes < required {loinc_required}. Codes: {loinc_codes}",
                extra={
                    "distinct": loinc_distinct,
                    "required": loinc_required,
                    "codes": loinc_codes,
                    "round": round_number,
                },
            )
            failures.append(GateFailure.LOW_LOINC_DIVERSITY)

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
            f"dose_evidence={dose_evidence_nodes} numeric_dose_signals={numeric_dose_signal_nodes} "
            f"target_drug_hits={target_drug_hits} newest_year={newest_label_year} "
            f"failures={[failure.value for failure in failures]}",
            extra={
                "round": round_number,
                "passed": report.passed,
                "n_hits": len(hits),
                "top_maxsim": top_maxsim,
                "dose_evidence_nodes": dose_evidence_nodes,
                "numeric_dose_signal_nodes": numeric_dose_signal_nodes,
                "target_drug_hits": target_drug_hits,
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
        absence_seeking = self._is_absence_seeking_query(new_intent.raw_query)

        if GateFailure.POPULATION_TOO_NARROW in failure_codes and new_intent.population_filter:
            old_population = new_intent.population_filter
            new_intent.population_filter = None
            log.info(
                f"[Rewrite R{round_number}] Dropped population filter '{old_population}'",
                extra={"rewrite": "drop_population_filter", "old": old_population, "round": round_number},
            )

        if GateFailure.NO_DOSING_NODES in failure_codes and "dosing" not in new_intent.layout_filters:
            if absence_seeking:
                added = []
                for pool in ("warning", "indication"):
                    if pool not in new_intent.layout_filters:
                        new_intent.layout_filters.append(pool)
                        added.append(pool)
                guidance = " If dosing is not specified by the label, answer INSUFFICIENT_EVIDENCE."
                if guidance.lower() not in new_intent.raw_query.lower():
                    new_intent.raw_query = f"{new_intent.raw_query}{guidance}".strip()
                    query_text_changed = True
                log.info(
                    f"[Rewrite R{round_number}] F3 absence-safe pivot: avoided dosing expansion; added pools={added}",
                    extra={"rewrite": "f3_absence_safe_pivot", "round": round_number, "added": added},
                )
            else:
                new_intent.layout_filters.insert(0, "dosing")
                new_intent.preferred_layout = "dosing"
                new_intent.wants_dosing = True
                log.info(
                    f"[Rewrite R{round_number}] Injected 'dosing' pool",
                    extra={"rewrite": "inject_dosing_pool", "round": round_number},
                )

        should_parent_pivot = GateFailure.NO_DOSE_EVIDENCE in failure_codes
        if GateFailure.NO_DOSING_NODES in failure_codes and not absence_seeking:
            should_parent_pivot = True

        if should_parent_pivot:
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
            for pool in ("interaction", "pharmacology", "warning"):
                if pool not in new_intent.layout_filters:
                    new_intent.layout_filters.append(pool)
                    added_pools.append(pool)
            new_intent.wants_interaction = True
            new_intent.wants_mechanism = True
            if new_intent.drug_names and len(new_intent.drug_names) >= 2:
                drug_pair = " ".join(new_intent.drug_names[:2])
                if drug_pair.lower() not in new_intent.raw_query.lower():
                    new_intent.raw_query = f"{new_intent.raw_query} {drug_pair}".strip()
                    query_text_changed = True
            if added_pools:
                log.info(
                    f"[Rewrite R{round_number}] Injected fallback causal pools: {added_pools}",
                    extra={"rewrite": "inject_interaction_pool", "added": added_pools, "round": round_number},
                )

        if GateFailure.TOO_FEW_HITS in failure_codes or GateFailure.LOW_MAXSIM in failure_codes:
            expanded_query = new_intent.raw_query
            expanded_query_changed = False
            for pattern, replacement in ABBREV_EXPANSION.items():
                rewritten_query = re.sub(pattern, replacement, expanded_query, flags=re.IGNORECASE)
                if rewritten_query != expanded_query:
                    expanded_query = rewritten_query
                    expanded_query_changed = True

            if expanded_query_changed:
                new_intent.raw_query = expanded_query
                query_text_changed = True
                log.info(
                    f"[Rewrite R{round_number}] Expanded abbreviations in query text",
                    extra={"rewrite": "abbrev_expansion", "new_query": expanded_query[:120], "round": round_number},
                )

            primary_pool = new_intent.preferred_layout
            fallback_pools = FALLBACK_LAYOUT_EXPANSION.get(primary_pool, ["warning"])
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

        if GateFailure.LOW_TARGET_DRUG_COVERAGE in failure_codes and new_intent.drug_names:
            if new_intent.query_type in {QUERY_TYPE_STRATEGIC, QUERY_TYPE_COMPARATIVE}:
                log.info(
                    f"[Rewrite R{round_number}] Skipped F8 target-drug reinforcement for {new_intent.query_type} query",
                    extra={"rewrite": "skip_reinforce_target_drugs", "round": round_number},
                )
            else:
                if _f8_metadata_lockdown_enabled():
                    new_intent.drug_lockdown_exact = True
                    
                    query_text_changed = True
                    log.info(
                        f"[Rewrite R{round_number}] F8 metadata lockdown enabled: exact generic/brand match",
                        extra={"rewrite": "f8_metadata_lockdown_exact", "round": round_number},
                    )
                else:
                    log.info(
                        f"[Rewrite R{round_number}] F8 metadata lockdown disabled via env flag",
                        extra={"rewrite": "f8_metadata_lockdown_disabled", "round": round_number},
                    )

        if GateFailure.INCOMPLETE_DOSING_LIFECYCLE in failure_codes:
            drug_phrase = " ".join(new_intent.drug_names) if new_intent.drug_names else "this drug"
            titration_query = (
                f"Provide the complete step-by-step titration schedule for {drug_phrase}. "
                f"Include the starting dose, week-by-week dose increases, "
                f"and the maximum recommended dosage."
            )
            new_intent.raw_query = titration_query
            query_text_changed = True
            parent_hits = self._fetch_parent_nodes(hits, max_parents=5)
            extra_hits.extend(parent_hits)
            log.info(
                f"[Rewrite R{round_number}] F9 clinical instruction pivot: rewrote query + "
                f"parent_pivot ({len(parent_hits)} parents added)",
                extra={
                    "rewrite": "titration_clinical_instruction_pivot",
                    "new_query": titration_query,
                    "parent_hits_added": len(parent_hits),
                    "round": round_number,
                },
            )

        if GateFailure.LOW_LOINC_DIVERSITY in failure_codes:
            broad_pools = ["dosing", "indication", "warning", "interaction"]
            added_pools = []
            for pool in broad_pools:
                if pool not in new_intent.layout_filters:
                    new_intent.layout_filters.append(pool)
                    added_pools.append(pool)

            if GateFailure.INCOMPLETE_DOSING_LIFECYCLE not in failure_codes:
                drug_phrase = " ".join(new_intent.drug_names) if new_intent.drug_names else "drug"

                preserved_context = []
                orig_lower = new_intent.raw_query.lower()

                ddi_patterns = [
                    r"(taking|on|with|plus|combined with)\s+([a-z]+(?:proate|arin|pine|zole|mycin|cillin|prazole|statin|dipine))",
                    r"(concurrently|concomitant(?:ly)?)\s+(?:with\s+)?([a-z]+)",
                ]
                for pattern in ddi_patterns:
                    match = re.search(pattern, orig_lower)
                    if match:
                        preserved_context.append(match.group(0))
                        break

                time_match = re.search(r"\b(\d+-?(?:week|day|hour|month)s?)\b", orig_lower)
                if time_match:
                    preserved_context.append(time_match.group(1))

                if re.search(r"\beach\s+(?:interval|week|day|dose|time\s+point)", orig_lower):
                    preserved_context.append("each interval")
                elif "complete schedule" in orig_lower:
                    preserved_context.append("complete schedule")

                context_str = " ".join(preserved_context) if preserved_context else ""
                new_intent.raw_query = (
                    f"complete dosing and administration information "
                    f"recommended dose special populations adjustments {drug_phrase} {context_str}"
                ).strip()
                query_text_changed = True

            parent_hits = self._fetch_parent_nodes(hits, max_parents=8)
            extra_hits.extend(parent_hits)
            log.info(
                f"[Rewrite R{round_number}] F10 LOINC diversity pivot: pools={added_pools} "
                f"parent_hits={len(parent_hits)} query_rewritten={not bool(GateFailure.INCOMPLETE_DOSING_LIFECYCLE in failure_codes)}",
                extra={
                    "rewrite": "loinc_diversity_pivot",
                    "added_pools": added_pools,
                    "parent_hits_added": len(parent_hits),
                    "round": round_number,
                },
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
            _hits_before_rewrite: int = 0
            last_best_clinical_score: float = 0.0

            for round_number in range(1, MAX_REFLECTION_ROUNDS + 1):
                is_final_round = round_number == MAX_REFLECTION_ROUNDS
                log.info(
                    f"[ReflectiveLoop] Starting round {round_number}/{MAX_REFLECTION_ROUNDS}",
                    extra={"round": round_number, "query": intent.raw_query[:60]},
                )

                with Timer(log, f"retrieve_round_{round_number}"):
                    round_hits = self._execute_parallel_retrieval(
                        intent,
                        q_vectors,
                        hard_date_filter=(round_number == 1 and explicit_year_filter),
                    )
                    round_hits = self._filter_to_target_drug_hits_if_sufficient(round_hits, intent)
                self._extend_unique_hits(all_hits, round_hits)
                current_best_clinical_score = self._best_clinical_score(all_hits)

                report = self._sufficiency_gate(all_hits, intent, round_number)
                sufficiency_reports.append(report)

                if self._should_stop_reflection(
                    round_number=round_number,
                    is_final_round=is_final_round,
                    report=report,
                    all_hits=all_hits,
                    intent=intent,
                    last_best_clinical_score=last_best_clinical_score,
                    current_best_clinical_score=current_best_clinical_score,
                ):
                    break

                last_best_clinical_score = current_best_clinical_score

                _hits_before_rewrite = len(all_hits)
                intent, extra_hits, text_changed = self._rewrite_intent(
                    intent,
                    report,
                    all_hits,
                    q_vectors,
                    round_number,
                )
                self._extend_unique_hits(all_hits, extra_hits)

                if not text_changed and len(all_hits) == _hits_before_rewrite:
                    log.warning(
                        f"[ReflectiveLoop] Frozen hits on round {round_number} "
                        f"(no rewrite, no new hits); proceeding with {len(all_hits)} hits. "
                        f"Failures: {report.failures}",
                        extra={"round": round_number, "n_hits": len(all_hits), "failures": report.failures},
                    )
                    break

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
                all_hits = self._filter_to_target_drug_hits_if_sufficient(all_hits, intent)
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
    "GateFailure",
    "ReflectivePharmaQueryEngine",
    "SufficiencyReport",
]