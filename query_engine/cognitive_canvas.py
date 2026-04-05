import html
import re
from collections import defaultdict
from .models import RetrievedNode, CogCanvasArtifact as BaseCogCanvasArtifact, CitationAuditResult
from .utils import generate_ngrams, extract_surrounding_context


def _x(text: object) -> str:
    return html.escape(str(text), quote=True)

class CogCanvasArtifact(BaseCogCanvasArtifact):
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

    def generate_constrained_prompt(self) -> str:
        self.conflicts = self._detect_conflicts()
        dose_evidence_nodes = getattr(self, "_dose_evidence_nodes", "?")
        reflection_rounds = getattr(self, "_reflection_rounds", "?")

        prompt = [
            "<system_directives>",
            "You are a deterministic clinical reasoning engine. You operate under ZERO-HALLUCINATION rules.",
            "1. CITATION MANDATE: Every clinical claim must end with its source URN -> [SOURCE: urn:pharma:fda_us:rxcui:...]",
            "2. VERBATIM ENFORCEMENT: Nodes marked <verbatim_locked> contain numerical or dosing data. You MUST quote exact constraints. DO NOT paraphrase doses or frequencies.",
            "3. NO EXTERNAL KNOWLEDGE: If the provided evidence cannot answer the query, output exactly: '⚠ INSUFFICIENT EVIDENCE: The provided label data does not contain enough information to answer safely.'",
            "4. CONFLICT RESOLUTION: If a <dose_conflict_warning> is present, you MUST present both values and cite both sources.",
            f"5. EVIDENCE AUDIT: {dose_evidence_nodes} node(s) in this context contain dose metadata. If this number is 0 for a dosing question, follow rule 3.",
            f"6. RETRIEVAL METADATA: {reflection_rounds} reflection round(s) were used to build this context.",
            "</system_directives>\n",
            
            "<query_intent>",
            f"<type>{_x(self.intent.query_type.upper())}</type>",
            f"<target_drugs>{_x(', '.join(self.intent.drug_names) or 'not specified')}</target_drugs>",
            f"<target_population>{_x(self.intent.population_filter or 'general')}</target_population>",
            "</query_intent>\n",
        ]

        if self.conflicts:
            prompt.append("<dose_conflict_warnings>")
            for c in self.conflicts:
                prompt.append(
                    f"  <conflict drug='{_x(c['drug'])}' route='{_x(c['route'])}' population='{_x(c['population'])}'>"
                    f"Values found: {_x(c['values'])} ({c['pct_diff']*100:.0f}% difference). "
                    f"Sources: {', '.join(_x(urn) for urn in c['urns'])}."
                    f"</conflict>"
                )
            prompt.append("</dose_conflict_warnings>\n")

        prompt.append("<clinical_evidence>")

        if self.verbatim_nodes:
            prompt.append("  <verbatim_locked_nodes type='primary_dosing_and_warnings'>")
            for n in self.verbatim_nodes:
                boxed = "true" if n.boxed_warning else "false"
                has_dose_metadata = "true" if n.dose_values else "false"
                prompt.append(
                    f"    <artifact urn='{_x(n.urn)}' type='{_x(n.layout_type)}' boxed_warning='{boxed}' "
                    f"has_dose_metadata='{has_dose_metadata}' temporal_anchor='{_x(n.label_version_date or 'unknown')}'>"
                )
                prompt.append(f"      <drug>{_x(n.drug_name_generic)}</drug>")
                prompt.append(
                    f"      <context population='{_x(n.patient_population or 'general')}' route='{_x(n.dose_route or 'unspecified')}'/>"
                )
                if n.dose_values:
                    prompt.append(
                        f"      <dose_metadata values='{_x(n.dose_values)}' units='{_x(n.dose_units)}'/>"
                    )
                prompt.append(f"      <exact_quote>{_x(n.verbatim_text)}</exact_quote>")
                prompt.append("    </artifact>")
            prompt.append("  </verbatim_locked_nodes>\n")
        else:
            prompt.append("  <!-- NO verbatim-locked dosing nodes retrieved -->\n")

        if self.paraphrase_nodes:
            prompt.append("  <supporting_nodes type='paraphrasable_context'>")
            for n in self.paraphrase_nodes:
                scope = "atc_class_level" if getattr(n, "_is_atc_neighbor", False) else "drug_specific"
                prompt.append(
                    f"    <artifact urn='{_x(n.urn)}' type='{_x(n.layout_type)}' scope='{scope}' temporal_anchor='{_x(n.label_version_date or 'unknown')}'>"
                    f"      <content>{_x(n.verbatim_text)}</content>"
                    f"    </artifact>"
                )
            prompt.append("  </supporting_nodes>\n")

        if self.causal_context:
            prompt.append("  <causal_mechanisms type='drug_interactions'>")
            for n in self.causal_context:
                prompt.append(
                    f"    <artifact urn='{_x(n.urn)}' drug='{_x(n.drug_name_generic)}'>"
                    f"      <mechanism_explanation>{_x(n.verbatim_text)}</mechanism_explanation>"
                    f"    </artifact>"
                )
            prompt.append("  </causal_mechanisms>\n")

        if self.macro_context:
            prompt.append("  <macro_context type='atc_class_summary'>")
            for n in self.macro_context:
                prompt.append(
                    f"    <artifact urn='{_x(n.urn)}' class='{_x(n.atc_code)}'>"
                    f"      <class_insight>{_x(n.verbatim_text)}</class_insight>"
                    f"    </artifact>"
                )
            prompt.append("  </macro_context>\n")

        if self.table_references:
            prompt.append("  <structured_tables>")
            for n in self.table_references:
                prompt.append(
                    f"    <artifact urn='{_x(n.urn)}'>"
                    f"      <table_data>{_x(n.verbatim_text)}</table_data>"
                    f"    </artifact>"
                )
            prompt.append("  </structured_tables>\n")

        prompt.append("</clinical_evidence>\n")

        prompt += [
            "<reasoning_protocol>",
            "Step 1: Identify the temporal anchors of the evidence and prefer the most recent label data.",
            "Step 2: Count nodes with has_dose_metadata='true'. If none exist for a dosing request, follow rule 3.",
            "Step 3: If scope='atc_class_level', state clearly that the evidence is class-level rather than drug-specific.",
            "Step 4: Map the <causal_mechanisms> to the <query_intent> for interaction or 'why' questions.",
            "Step 5: Extract exact numerical constraints from <verbatim_locked_nodes>.",
            "</reasoning_protocol>\n"
        ]

        prompt += [
            "<user_query>",
            _x(self.intent.raw_query),
            "</user_query>"
        ]

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