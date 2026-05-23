import html
import re
from collections import defaultdict
from logger import get_logger
from .constants import POPULATION_EQUIVALENTS, NUMERIC_DOSE_SIGNAL_RE
from .models import RetrievedNode, CogCanvasArtifact as BaseCogCanvasArtifact, CitationAuditResult
from .utils import generate_ngrams, extract_surrounding_context
from .verbatim_extractor import VerbatimNumberExtractor, augment_answer_with_verbatim_numbers


def _x(text: object) -> str:
    return html.escape(str(text), quote=True)


log = get_logger("cognitive_canvas")

class CogCanvasArtifact(BaseCogCanvasArtifact):
    @staticmethod
    def _contains_numeric_dose_signal(text: str) -> bool:
        return bool(NUMERIC_DOSE_SIGNAL_RE.search(text or ""))

    def _detect_conflicts(self) -> list[dict]:
        conflicts = []
        dose_nodes = [
            n for n in (self.verbatim_nodes + self.paraphrase_nodes)
            if n.layout_type == "dosing" and n.dose_values
        ]
        groups: dict[tuple, list[RetrievedNode]] = defaultdict(list)
        for n in dose_nodes:
            key = (n.drug_name_generic, n.dose_route or "any",
                   n.patient_population or "general")
            groups[key].append(n)

        for key, nodes in groups.items():
            if len(nodes) < 2:
                continue
            vals = sorted({val for n in nodes for val in n.dose_values})
            if not vals:
                continue
            min_val, max_val = min(vals), max(vals)
            if min_val == 0:
                continue
            pct_diff = abs(max_val - min_val) / min_val
            if pct_diff > 0.15:
                drug, route, pop = key
                conflicts.append({
                    "type":       "dose_conflict",
                    "drug":       drug,
                    "route":      route,
                    "population": pop,
                    "values":     vals,
                    "pct_diff":   pct_diff,
                    "urns":       [n.urn for n in nodes],
                })
        return conflicts

    def _extract_high_authority_facts(
        self,
        limit: int = 20,
        nodes: list[RetrievedNode] | None = None,
    ) -> list[str]:
        facts: list[str] = []
        seen: set[str] = set()
        if nodes is None:
            nodes = self.verbatim_nodes + self.paraphrase_nodes

        for n in nodes:
            if not n.urn:
                continue

            drug = n.drug_name_generic or "unknown_drug"
            population = n.patient_population or "general"

            if n.dose_values:
                units = "/".join(str(u) for u in n.dose_units) if n.dose_units else "unspecified_units"
                doses = ", ".join(str(v) for v in n.dose_values)
                fact = f"FACT: {drug} | {population} | dose={doses} {units} | urn={n.urn}"
            else:
                snippet = re.sub(r"\s+", " ", (n.verbatim_text or "").strip())[:150]
                if not snippet:
                    continue
                fact = f"FACT: {drug} | {population} | statement={snippet}... | urn={n.urn}"

            if fact in seen:
                continue

            seen.add(fact)
            facts.append(fact)
            if len(facts) >= limit:
                break

        return facts

    def _extract_clinical_fact_triplets(self, limit: int = 24) -> list[str]:
        return self._extract_high_authority_facts(limit=min(limit, 20))

    def _render_nodes(self, nodes: list[RetrievedNode], tag: str) -> str:
        lines = [f"<{tag}>"]
        for n in nodes:
            if not n.urn:
                continue
            lines.append(
                f"  <artifact urn='{_x(n.urn)}' type='{_x(n.layout_type)}' "
                f"population='{_x(n.patient_population or 'general')}'>{_x(n.verbatim_text)}</artifact>"
            )
        lines.append(f"</{tag}>")
        return "\n".join(lines)

    def _population_incompatible(self, node_population: str) -> bool:
        intent_pop = self.intent.population_filter
        if not intent_pop or not node_population:
            return False
        
        intent_pop = intent_pop.lower().strip()
        node_pop = node_population.lower().strip()

        intent_group = POPULATION_EQUIVALENTS.get(intent_pop, {intent_pop})
        node_group = POPULATION_EQUIVALENTS.get(node_pop, {node_pop})
        
        if intent_group & node_group:
            return False
        
        incompatible_pairs = {
            "neonatal": {"pediatric", "peds", "paediatric", "pediatric_population", "adult", "geriatric", "elderly", "general"},
            "pediatric": {"adult", "geriatric", "elderly"},
            "peds": {"adult", "geriatric", "elderly"},
            "pediatric_population": {"adult", "geriatric", "elderly"},
            "elderly": {"neonatal", "pediatric", "peds", "pediatric_population"},
            "geriatric": {"neonatal", "pediatric", "peds", "pediatric_population"},
            "pregnancy": {"neonatal", "pediatric", "peds", "pediatric_population"},  # pregnancy is adult-specific
        }
        
        if intent_pop in incompatible_pairs:
            if node_pop in incompatible_pairs[intent_pop]:
                return True
        
        return False
    
    def _filter_nodes_by_population(self, nodes: list[RetrievedNode]) -> list[RetrievedNode]:
        if not self.intent.population_filter:
            return nodes

        specialized_population = (self.intent.population_filter or "").lower().strip() not in {"", "general"}

        is_renal_hepatic_query = any(pop in (self.intent.population_filter or "").lower()
                                     for pop in ["renal", "hepatic", "kidney", "liver"])

        filtered = []
        for node in nodes:
            node_population = (node.patient_population or "general").lower().strip()

            if self._population_incompatible(node_population):
                continue

            is_constraint_node = node.layout_type in {"contraindication", "warning", "interaction"}
            is_renal_hepatic_node = node_population in {"renal", "hepatic", "renal_impairment", "hepatic_impairment"}

            if (
                specialized_population
                and node_population == "general"
                and not is_constraint_node  # Don't filter contraindications/warnings
                and not is_renal_hepatic_query  # Don't filter if query is about renal/hepatic
                and (node.dose_values or self._contains_numeric_dose_signal(node.verbatim_text or ""))
            ):
                continue

            filtered.append(node)

        if len(filtered) < len(nodes):
            excluded_count = len(nodes) - len(filtered)
            log.info(
                f"[Layer 5] Population pruning: excluded {excluded_count} node(s) "
                f"incompatible with population={self.intent.population_filter}",
                extra={"excluded": excluded_count, "population": self.intent.population_filter}
            )
        return filtered

    def audit_citation_fidelity(self, llm_answer: str) -> CitationAuditResult:
        all_node_urns = {
            n.urn for n in
            self.verbatim_nodes + self.paraphrase_nodes +
            self.causal_context + self.macro_context + self.table_references
        }

        cited_urns = set(re.findall(r'\[SOURCE:\s*(urn:pharma:[^\]]+)\]', llm_answer))

        hallucinated = cited_urns - all_node_urns
        verified     = cited_urns & all_node_urns
        missing      = all_node_urns - cited_urns

        misattributed = []
        urn_to_node = {n.urn: n for n in
                       self.verbatim_nodes + self.paraphrase_nodes +
                       self.causal_context + self.macro_context}

        for urn in verified:
            node = urn_to_node.get(urn)
            if not node:
                continue
            node_grams = set(generate_ngrams(node.verbatim_text.lower(), 4))
            surrounding = extract_surrounding_context(llm_answer, urn, window=300)
            claim_grams = set(generate_ngrams(surrounding.lower(), 4))
            overlap = len(node_grams & claim_grams) / max(len(node_grams), 1)
            if overlap < 0.08 and node.verbatim_locked:
                misattributed.append((urn, "low_gram_overlap", overlap))

        dose_pattern = re.compile(
            r'\d+\.?\d*\s*(?:mg/kg|mg/m2|mcg/kg|mg|mcg|units?/kg)', re.I
        )
        orphaned_claims = []
        for m in dose_pattern.finditer(llm_answer):
            start = max(0, m.start() - 200)
            end   = min(len(llm_answer), m.end() + 200)
            nearby = llm_answer[start:end]
            if "[SOURCE:" not in nearby:
                orphaned_claims.append(m.group())

        return CitationAuditResult(
            hallucinated    = list(hallucinated),
            verified        = list(verified),
            missing         = list(missing),
            misattributed   = misattributed,
            orphaned_claims = orphaned_claims,
            audit_passed    = (len(hallucinated) == 0 and len(orphaned_claims) == 0),
        )

    _PROHIBITION_RE = re.compile(
        r"(?:not\s+recommended|not\s+established|contraindicated|not\s+indicated|"
        r"not\s+approved|safety\s+.*not\s+.*established|should\s+not\s+be\s+used|"
        r"is\s+not\s+(?:indicated|recommended)|use\s+is\s+not\s+recommended|"
        r"has\s+not\s+been\s+(?:established|studied|evaluated))",
        re.IGNORECASE,
    )

    _ABSENCE_SEEKING_QUERY_RE = re.compile(
        r"\b(?:type\s*1\s*diabetes|child\s+under\s+\d|under\s+\d\s+years?\s+of\s+age|"
        r"missed\s+.*doses?|catch-?up\s+dos|"
        r"common\s+cold|off-?label|neonat|newborn|simultaneously)\b",
        re.IGNORECASE,
    )

    def _detect_behavioral_override(self) -> str | None:
        all_nodes = (
            self.verbatim_nodes + self.paraphrase_nodes +
            self.causal_context + self.macro_context
        )
        if not all_nodes:
            return "INSUFFICIENT_EVIDENCE"

        query_lower = (self.intent.raw_query or "").lower()
        query_is_absence_seeking = bool(self._ABSENCE_SEEKING_QUERY_RE.search(query_lower))

        prohibition_nodes = []
        for node in all_nodes:
            text = node.verbatim_text or ""
            if self._PROHIBITION_RE.search(text):
                prohibition_nodes.append(node)

        for node in prohibition_nodes:
            text_lower = (node.verbatim_text or "").lower()
            if "contraindicated" in text_lower or "should not be used" in text_lower:
                if node.layout_type in ("contraindication", "warning", "indication") and query_is_absence_seeking:
                    return "CONTRAINDICATED"

            if query_is_absence_seeking and any(
                p in text_lower for p in ("not established", "not been established", "not been studied", "not been evaluated")
            ):
                if node.layout_type in ("indication", "warning", "dosing"):
                    return "CONTRAINDICATED"

        if query_is_absence_seeking:
            if prohibition_nodes:
                return "CONTRAINDICATED"
            return "INSUFFICIENT_EVIDENCE"

        return None

    def generate_constrained_prompt(self) -> str:
        verbatim_pruned = self._filter_nodes_by_population(self.verbatim_nodes)
        paraphrase_pruned = self._filter_nodes_by_population(self.paraphrase_nodes)
        causal_pruned = self._filter_nodes_by_population(self.causal_context)
        macro_pruned = self._filter_nodes_by_population(self.macro_context)
        table_pruned = self._filter_nodes_by_population(self.table_references)

        facts = self._extract_high_authority_facts(nodes=(verbatim_pruned + paraphrase_pruned))

        behavioral_override = self._detect_behavioral_override()

        prompt = [
            "<TASK>",
            "Answer the clinical question using ONLY the provided evidence.",
            "You MUST respond in JSON format with this exact structure:",
            "{",
            '  "answer": "Your detailed answer with citations [SOURCE: urn:...]",',
            '  "numeric_values": [list of ALL relevant numbers mentioned in your answer]',
            "}",
            "",
            "CRITICAL EXTRACTION RULES:",
            "1. Include EVERY numeric value in 'numeric_values' that appears in your answer",
            "2. Doses: Extract as numbers (e.g., 20 from '20 mg')",
            "3. Ranges: Extract both ends (e.g., [2.0, 3.0] from 'INR 2.0-3.0')",
            "4. Thresholds: Extract the number (e.g., 65 from 'age > 65 years')",
            "5. Fold-changes: Extract multiplier (e.g., 4.0 from '4-fold increase')",
            "",
            "TITRATION / SCHEDULE EXTRACTION RULES (for titration, escalation, 'complete schedule', 'each week/day/interval' questions):",
            "6. You MUST list EVERY intermediate titration step as a table: 'Week 1: X mg/day, Week 2: Y mg/day, ...'",
            "   Omitting ANY intermediate titration step is a CRITICAL FAILURE.",
            "   Example: if label says 'Week 1-2: 25 mg, Week 3-4: 50 mg, Week 5+: 100 mg' you MUST list ALL three steps.",
            "7. For population-specific schedules (e.g., 'with valproate'): Extract ONLY that regimen, ignore other regimens",
            "8. Exclude contextual numbers NOT part of dose schedule: '5 half-lives' → exclude 5.0, '2-fold' → exclude 2.0",
            "9. For multi-step titrations, your numeric_values MUST include the dose at EACH step, not just start and end.",
            "10. When the label specifies 'increase by X mg every Y days', compute and list each resulting dose level.",
            "",
            "Examples:",
            '{"answer": "Start 25-50 mcg in patients > 65 years [SOURCE: ...]", "numeric_values": [25, 50, 65]}',
            '{"answer": "Max 20 mg with Gemfibrozil [SOURCE: ...]", "numeric_values": [20]}',
            '{"answer": "Target INR 2.0-3.0 [SOURCE: ...]", "numeric_values": [2.0, 3.0]}',
            '{"answer": "Weeks 1-2: 25 mg every other day, Weeks 3-4: 25 mg daily, Week 5: 50 mg daily [SOURCE: ...]", "numeric_values": [25, 25, 50]}',
            "</TASK>",
        ]

        if behavioral_override:
            refusal_phrase = (
                "This use is not indicated per the label. "
                "The evidence is insufficient to provide the requested information."
            )
            prompt.extend([
                "<BEHAVIORAL_CONSTRAINT>",
                f"The retrieved evidence indicates this use case is {behavioral_override}.",
                f"Your answer MUST begin with EXACTLY: \"{refusal_phrase}\"",
                "Then quote the specific label language that prohibits or does not support this use.",
                "CRITICAL RULES FOR REFUSAL ANSWERS:",
                "- Do NOT mention ANY specific dose values (mg, mcg, mL, mEq, units) anywhere in your answer.",
                "- Do NOT provide dosing from a different population or indication as a fallback.",
                "- Do NOT suggest alternative dosing, dose adjustments, or general dosing information.",
                "- Your numeric_values array MUST be empty: []",
                '- Example: {"answer": "' + refusal_phrase + ' The label states: [quote] [SOURCE: urn:...]", "numeric_values": []}',
                "</BEHAVIORAL_CONSTRAINT>",
            ])

        if self.intent.wants_warning_numerics:
            prompt.extend([
                "<TOXICOLOGY_MONITORING_EXTRACTION>",
                "Extract ALL of the following from the evidence:",
                "- Serum level thresholds (e.g., '> 1.5 mEq/L is toxic')",
                "- Monitoring intervals (e.g., 'every 3 months', 'weekly for first 6 weeks')",
                "- Study dose multiples (e.g., '4 times the recommended human dose')",
                "- Concentration values and therapeutic windows",
                "- Incidence rates and percentages from warnings/precautions",
                "These values are CRITICAL for clinical safety — missing any is a failure.",
                "</TOXICOLOGY_MONITORING_EXTRACTION>",
            ])

        if self.intent.brand_hint:
            prompt.extend([
                "<BRAND_CONTEXT>",
                f"The query specifies: {_x(self.intent.brand_hint)}.",
                "Provide dosing ONLY for this specific product/indication.",
                "Do NOT include doses from other brands of the same molecule.",
                "</BRAND_CONTEXT>",
            ])

        prompt.extend([
            "<query_intent>",
            f"  <type>{_x(self.intent.query_type.upper())}</type>",
            f"  <target_drugs>{_x(', '.join(self.intent.drug_names) or 'not specified')}</target_drugs>",
            f"  <target_population>{_x(self.intent.population_filter or 'general')}</target_population>",
            "</query_intent>",
            "<knowledge_bottleneck>",
        ])

        if facts:
            prompt.extend(f"  <fact>{_x(f)}</fact>" for f in facts)
        else:
            prompt.append("  <fact>FACT: none | none | statement=no high-authority facts extracted | urn=none</fact>")

        prompt.extend([
            "</knowledge_bottleneck>",
            "<clinical_evidence>",
            self._render_nodes(verbatim_pruned, "verbatim_locked"),
            self._render_nodes(paraphrase_pruned, "supporting_context"),
            self._render_nodes(causal_pruned, "causal_context"),
            self._render_nodes(macro_pruned, "macro_context"),
            self._render_nodes(table_pruned, "table_references"),
            "</clinical_evidence>",
            "<reasoning_protocol>",
            "1. Read ALL verbatim_locked artifacts completely - answers may be worded differently than question",
            "2. Extract EVERY numeric value (doses, thresholds, ranges, fold-changes, percentages) that helps answer",
            "3. If exact information requested is not specified, state that clearly BUT extract related information that IS present",
            "4. For population-specific queries, prioritize population-specific guidance over general",
            "5. Quote critical instructions, warnings, dosing regimens verbatim with citations. Include section cross-references (e.g., 'See Warnings and Precautions (5.1)') as they appear in the label.",
            "6. Include context for numbers: 'eGFR < 30' not just '30', '4-fold increase' not just '4'",
            "</reasoning_protocol>",
            "<CRITICAL_FINAL_REMINDER>",
            "Before submitting your answer, verify:",
            "- Have you included EVERY dose value (mg, mcg, units) mentioned in verbatim_locked?",
            "- Have you included EVERY threshold (eGFR, INR, age, lab values) mentioned in verbatim_locked?",
            "- Have you included EVERY range (2.0-3.0, 30-45, dose ranges) mentioned in verbatim_locked?",
            "- Have you included EVERY fold-change or percentage mentioned in verbatim_locked?",
            "",
            "If verbatim_locked says 'maximum dose 20 mg', your answer MUST say '20 mg'",
            "If verbatim_locked says 'INR target 2-3', your answer MUST say '2-3' or '2.0-3.0'",
            "If verbatim_locked says 'age > 65', your answer MUST say '65'",
            "If verbatim_locked says '4-fold increase', your answer MUST say '4-fold'",
            "",
            "NEVER write descriptions like 'dose should be reduced' - ALWAYS write 'dose should be reduced to X mg'",
            "NEVER write 'increased risk in elderly' - ALWAYS write 'increased risk in patients > 65 years'",
            "NEVER write 'maintain therapeutic range' - ALWAYS write 'maintain INR 2.0-3.0'",
            "NEVER summarize a titration as 'start low and increase' - list EVERY step with EXACT doses",
            "NEVER skip intermediate titration steps - if label has 5 steps, your answer needs ALL 5 doses",
            "",
            "Your answer quality is judged by numeric completeness. Missing a single dose value = FAILURE.",
            "</CRITICAL_FINAL_REMINDER>",
            f"<user_query>{_x(self.query or self.intent.raw_query)}</user_query>",
        ])

        return "\n".join(prompt)

    def get_artifact_summary(self) -> dict:
        return {
            "query_intent": self.intent.query_type,
            "latency_ms": round(self.total_latency_ms, 2),
            "artifact_counts": {
                "verbatim_locked": len(self.verbatim_nodes),
                "paraphrasable": len(self.paraphrase_nodes),
                "causal_interactions": len(self.causal_context),
                "macro_summaries": len(self.macro_context),
                "dosing_tables": len(self.table_references)
            },
            "conflicts_detected": len(self.conflicts)
        }

    def augment_answer_with_verbatim_facts(self, llm_answer: str) -> str:
        all_evidence = (
            self.verbatim_nodes +
            self.paraphrase_nodes +
            self.causal_context +
            self.macro_context +
            self.table_references
        )

        return augment_answer_with_verbatim_numbers(llm_answer, all_evidence)