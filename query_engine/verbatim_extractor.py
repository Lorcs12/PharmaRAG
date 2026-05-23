from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class NumericFact:
    value: float
    context: str
    source_urn: str
    category: str


class VerbatimNumberExtractor:

    def __init__(self):
        self.dose_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|mL|units?|mEq)(?:\s*/\s*(?:kg|day|dose|hr|m2))?',
            re.IGNORECASE
        )

        self.threshold_pattern = re.compile(
            r'(?:eGFR|INR|age|ALT|AST|creatinine|K\+?|Na\+?)\s*[<>≥≤=]\s*(\d+(?:\.\d+)?)',
            re.IGNORECASE
        )

        self.range_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(?:mg|mcg|mL/min|years?)?',
            re.IGNORECASE
        )

        self.fold_change_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*-?\s*fold\s+(?:increase|decrease|higher|lower)',
            re.IGNORECASE
        )

        self.max_dose_pattern = re.compile(
            r'(?:maximum|max|not\s+to\s+exceed)\s+(?:dose\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:mg|mcg)',
            re.IGNORECASE
        )

        self.monitoring_interval_pattern = re.compile(
            r'every\s+(\d+)\s+(?:weeks?|days?|months?|hours?)',
            re.IGNORECASE
        )

        self.dose_multiple_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(?:times?|x)\s+(?:the\s+)?(?:human|recommended|clinical|therapeutic|MRHD)',
            re.IGNORECASE
        )

        self.section_ref_pattern = re.compile(
            r'(?:Section|section)s?\s+(\d+(?:\.\d+)?)',
            re.IGNORECASE
        )

    def extract_from_chunk(self, text: str, source_urn: str) -> list[NumericFact]:
        facts = []

        for match in self.dose_pattern.finditer(text):
            value = float(match.group(1))
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()

            facts.append(NumericFact(
                value=value,
                context=context,
                source_urn=source_urn,
                category="dose"
            ))

        for match in self.threshold_pattern.finditer(text):
            value = float(match.group(1))
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].strip()

            facts.append(NumericFact(
                value=value,
                context=context,
                source_urn=source_urn,
                category="threshold"
            ))

        for match in self.range_pattern.finditer(text):
            for i in [1, 2]:
                value = float(match.group(i))
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()

                facts.append(NumericFact(
                    value=value,
                    context=context,
                    source_urn=source_urn,
                    category="range"
                ))

        for match in self.fold_change_pattern.finditer(text):
            value = float(match.group(1))
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()

            facts.append(NumericFact(
                value=value,
                context=context,
                source_urn=source_urn,
                category="fold_change"
            ))

        for match in self.max_dose_pattern.finditer(text):
            value = float(match.group(1))
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            context = text[start:end].strip()

            facts.append(NumericFact(
                value=value,
                context=context,
                source_urn=source_urn,
                category="max_dose"
            ))

        for match in self.monitoring_interval_pattern.finditer(text):
            value = float(match.group(1))
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()

            facts.append(NumericFact(
                value=value,
                context=context,
                source_urn=source_urn,
                category="monitoring"
            ))

        for match in self.dose_multiple_pattern.finditer(text):
            value = float(match.group(1))
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()

            facts.append(NumericFact(
                value=value,
                context=context,
                source_urn=source_urn,
                category="dose_multiple"
            ))

        return facts

    def extract_from_evidence(self, evidence_nodes: list) -> list[NumericFact]:
        all_facts = []
        seen_contexts = set()

        for node in evidence_nodes:
            text = getattr(node, 'verbatim_text', '') or ''
            urn = getattr(node, 'urn', 'unknown')

            facts = self.extract_from_chunk(text, urn)

            for fact in facts:
                context_key = f"{fact.value}:{fact.context[:50]}"
                if context_key not in seen_contexts:
                    seen_contexts.add(context_key)
                    all_facts.append(fact)

        return all_facts

    def format_facts_for_answer(self, facts: list[NumericFact]) -> str:
        if not facts:
            return ""

        by_category: dict[str, list[NumericFact]] = {}
        for fact in facts:
            if fact.category not in by_category:
                by_category[fact.category] = []
            by_category[fact.category].append(fact)

        lines = ["\n**Numeric Details from Label:**"]

        for category, cat_facts in sorted(by_category.items()):
            cat_facts.sort(key=lambda f: f.value)

            if category == "dose":
                lines.append(f"\nDoses: {', '.join(f'{f.value} mg' for f in cat_facts[:8])}")
            elif category == "max_dose":
                for f in cat_facts[:3]:
                    lines.append(f"- Maximum dose: {f.context} [SOURCE: {f.source_urn}]")
            elif category == "threshold":
                for f in cat_facts[:5]:
                    lines.append(f"- Threshold: {f.context}")
            elif category == "range":
                ranges = set()
                for f in cat_facts:
                    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)', f.context)
                    if match:
                        ranges.add(f"{match.group(1)}-{match.group(2)}")
                if ranges:
                    lines.append(f"- Ranges: {', '.join(sorted(ranges)[:5])}")
            elif category == "fold_change":
                for f in cat_facts[:3]:
                    lines.append(f"- {f.context}")
            elif category == "monitoring":
                for f in cat_facts[:5]:
                    lines.append(f"- Monitoring: {f.context} [SOURCE: {f.source_urn}]")
            elif category == "dose_multiple":
                for f in cat_facts[:5]:
                    lines.append(f"- Dose multiple: {f.context} [SOURCE: {f.source_urn}]")

        return "\n".join(lines)


def augment_answer_with_verbatim_numbers(
    llm_answer: str,
    evidence_nodes: list,
    extractor: VerbatimNumberExtractor | None = None
) -> str:
    if extractor is None:
        extractor = VerbatimNumberExtractor()

    facts = extractor.extract_from_evidence(evidence_nodes)

    if not facts:
        return llm_answer

    numeric_section = extractor.format_facts_for_answer(facts)

    return llm_answer + numeric_section
