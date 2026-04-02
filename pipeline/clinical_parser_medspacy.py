import spacy
import medspacy
from medspacy.ner import TargetRule
from typing import Optional, List, Dict, Tuple
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
        """Processes text in a single pass to extract clean, contextualized entities."""
        doc = self.nlp(text)
        
        results = {
            "route": None,
            "population": None,
            "dose_val": None,
            "dose_unit": None,
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
            
            elif label == "DOSAGE" and results["dose_val"] is None:
                try:
                    results["dose_val"] = float(ent[0].text)
                    results["dose_unit"] = ent[1].text.lower()
                except (ValueError, IndexError):
                    pass

        results["route"] = results["route"] or "unspecified"
        results["population"] = results["population"] or "general"
        
        return results

    def execute_semantic_chunking(self, text: str, max_chars: int = None, overlap: int = None) -> List[str]:
        max_chars = max_chars or CFG.ingestion.max_chunk_chars
        overlap = overlap or CFG.ingestion.chunk_overlap

        if len(text) <= max_chars:
            return [text]

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        chunks, buf = [], ""
        for sent in sentences:
            if len(buf) + len(sent) + 1 <= max_chars:
                buf = (buf + " " + sent).strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = sent
        if buf:
            chunks.append(buf)

        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_tail = chunks[i-1][-overlap:]
                chunk = prev_tail + " " + chunk
            overlapped_chunks.append(chunk.strip())
            
        return overlapped_chunks

clinical_analyzer = ClinicalTextAnalyzer()