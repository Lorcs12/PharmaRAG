import spacy
import medspacy
from medspacy.ner import TargetRule
from typing import Optional, List, Dict, Tuple
import re
from config import CFG

class ClinicalTextAnalyzer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClinicalTextAnalyzer, cls).__new__(cls)
            cls._instance._initialize_pipeline()
        return cls._instance

    def _initialize_pipeline(self):
        self.nlp = medspacy.load()
        self.target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
        self._setup_rules()

    def _setup_rules(self):
        rules = []
        
        pop_keywords = {
            "POP_PEDIATRIC": ["pediatric", "children", "child", "infant", "neonatal", "age <"],
            "POP_ELDERLY": ["elderly", "geriatric", "older adult", "age ≥65", "age >=65"],
            "POP_RENAL": ["renal impairment", "renal failure", "kidney", "ckd", "creatinine clearance"],
            "POP_HEPATIC": ["hepatic impairment", "liver", "hepatic failure", "child-pugh"],
            "POP_PREGNANCY": ["pregnancy", "pregnant", "lactation", "breastfeed", "nursing"]
        }
        for label, terms in pop_keywords.items():
            for term in terms:
                rules.append(TargetRule(term, label))

        route_keywords = {
            "ROUTE_ORAL": ["orally", "oral", "by mouth", "po", "tablet", "capsule"],
            "ROUTE_INTRAVENOUS": ["intravenous", "iv", "i.v.", "infusion", "injection"],
            "ROUTE_SUBCUTANEOUS": ["subcutaneous", "sc", "s.c.", "subcut"],
            "ROUTE_INTRAMUSCULAR": ["intramuscular", "im", "i.m."],
            "ROUTE_INHALED": ["inhaled", "inhalation", "inhaler"],
            "ROUTE_TOPICAL": ["topical", "apply", "cream", "ointment", "patch"]
        }
        for label, terms in route_keywords.items():
            for term in terms:
                rules.append(TargetRule(term, label))

        dose_pattern = [
            {"LIKE_NUM": True},
            {"LOWER": {"IN": ["mg/kg", "mg/m2", "mcg/kg", "mg", "mcg", "g", "units/kg", "unit/kg", "meq", "ml"]}}
        ]
        rules.append(TargetRule("DOSAGE", "DOSAGE", pattern=dose_pattern))

        self.target_matcher.add(rules)

    def extract_clinical_entities(self, text: str) -> Dict:
        doc = self.nlp(text)
        
        results = {
            "route": None,
            "population": None,
            "dose_values": [],
            "dose_units": set(),
            "sentences": [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
        }

        for ent in doc.ents:
            if getattr(ent._, "is_negated", False) or getattr(ent._, "is_historical", False):
                continue

            label = ent.label_

            if label.startswith("ROUTE_"):
                results["route"] = label.replace("ROUTE_", "").lower()
            
            elif label.startswith("POP_"):
                results["population"] = label.replace("POP_", "").lower()
            
            elif label == "DOSAGE":
                try:
                    val = float(ent[0].text)
                    unit = ent[1].text.lower()
                    if val not in results["dose_values"]:
                        results["dose_values"].append(val)
                    results["dose_units"].add(unit)
                except (ValueError, IndexError):
                    pass

        results["route"] = results["route"] or "unspecified"
        results["population"] = results["population"] or "general"
        results["dose_units"] = sorted(results["dose_units"])
        
        return results

    def execute_semantic_chunking(self, text: str, max_chars: int = None, overlap_sentences: int = 1) -> List[str]:
        max_chars = max_chars or CFG.ingestion.max_chunk_chars

        if len(text) <= max_chars:
            return [text]

        protected_text = re.sub(r'(\|[^\n]+\|\n)', r'\1<TBL_BREAK>', text)

        doc = self.nlp(protected_text)
        sentences = [sent.text.replace('<TBL_BREAK>', '').strip() for sent in doc.sents if sent.text.strip()]
        sentences = self._merge_continuation_blocks(sentences)

        chunks = []
        current_chunk_sents: List[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > max_chars and current_chunk_sents:
                chunks.append(" ".join(current_chunk_sents))
                overlap_start = max(0, len(current_chunk_sents) - overlap_sentences)
                current_chunk_sents = current_chunk_sents[overlap_start:]
                current_len = sum(len(s) + 1 for s in current_chunk_sents)
            current_chunk_sents.append(sent)
            current_len += sent_len + 1

        if current_chunk_sents:
            chunks.append(" ".join(current_chunk_sents))

        return chunks

    @staticmethod
    def _merge_continuation_blocks(sentences: List[str]) -> List[str]:
        if not sentences:
            return []

        merged: List[str] = []
        for sentence in sentences:
            if merged and ClinicalTextAnalyzer._should_attach_to_previous(merged[-1], sentence):
                merged[-1] = f"{merged[-1]} \n{sentence}".strip()
            else:
                merged.append(sentence)
        return merged

    @staticmethod
    def _should_attach_to_previous(previous: str, current: str) -> bool:
        previous = previous.strip()
        current = current.strip()
        if not previous or not current:
            return False

        if previous.endswith(":"):
            return True

        return False

clinical_analyzer = ClinicalTextAnalyzer()