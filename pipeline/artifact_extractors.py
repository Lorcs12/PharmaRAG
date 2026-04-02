import re
from typing import Optional
import numpy as np
from config import CFG
from .models import RawChunk, DrugLabel
from .helpers import strip_html
from .clinical_parsers_regex import (
    parse_dose_value, parse_clinical_route, 
    parse_patient_population, execute_semantic_chunking
)
from .clinical_parser_medspacy import clinical_analyzer
from .ingestion_constants import FDA_FIELD_MAP

def generate_artifact_urn(rxcui: str, set_id: str, layout_type: str, regulatory_source: str, seq: int) -> str:
    set_prefix = set_id.replace("-", "")[:8]
    return f"urn:pharma:{regulatory_source}:{rxcui}:{set_prefix}:{layout_type}:{seq:04d}"

def generate_dosing_urn(rxcui: str, set_id: str, route: str, population: str, seq: int) -> str:
    set_prefix = set_id.replace("-", "")[:8]
    route_slug = (route or "unspecified").replace(" ", "_")
    pop_slug   = (population or "general").replace(" ", "_")
    return f"urn:pharma:fda_us:{rxcui}:{set_prefix}:dosing_fact:{route_slug}:{pop_slug}:{seq:04d}"

def extract_hierarchical_dosing_artifacts(label_data: dict, drug: DrugLabel, logger) -> list[RawChunk]:
    artifacts = []
    results = label_data.get("results", [{}])
    if not results:
        return artifacts

    result = results[0]
    seq = 0

    for field_name, (loinc_code, layout_type) in FDA_FIELD_MAP.items():
        table_field = f"{field_name}_table"

        raw_sections = result.get(table_field) or result.get(field_name)

        if not raw_sections:
            continue

        base_raglens = CFG.ingestion.raglens_by_layout.get(layout_type, 0.50)
        if drug.boxed_warning:
            base_raglens = min(1.0, base_raglens + 0.05)

        full_text = " \n".join(strip_html(s) for s in raw_sections)

        text_chunks = clinical_analyzer.execute_semantic_chunking(full_text)

        for chunk_text in text_chunks:
            if len(chunk_text) < CFG.ingestion.min_text_len:
                continue

            parent_urn = generate_artifact_urn(drug.rxcui, drug.set_id, layout_type, drug.source, seq)

            artifacts.append(RawChunk(
                urn_id=parent_urn,
                parent_urn=None,
                set_id=drug.set_id,
                rxcui=drug.rxcui,
                layout_type="dosing",
                verbatim_text=chunk_text,
                chunk_confidence=0.97,
                raglens_risk=base_raglens,
                label_version_date=drug.label_date,
                published_date=drug.published,
                drug_name_generic=drug.drug_generic,
                drug_name_brand=drug.drug_brand,
                atc_code=drug.atc_code,
                smpc_section_code="34068-7",
                boxed_warning=drug.boxed_warning,
            ))
            seq += 1

            if layout_type in ["dosing", "indication", "warning"]:
                sentences = re.split(r'(?<=[.!?])\s+', chunk_text.replace("<TBL_BREAK>", ""))

                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) < 30 or not any(c.isdigit() for c in sent):
                        continue

                    sent_data = clinical_analyzer.extract_clinical_entities(sent)

                    if not sent_data["dose_values"]:
                        continue

                    child_urn = generate_dosing_urn(drug.rxcui, drug.set_id, sent_data["route"], sent_data["population"], seq)

                    artifacts.append(RawChunk(
                        urn_id=child_urn,
                        parent_urn=parent_urn,
                        set_id=drug.set_id,
                        rxcui=drug.rxcui,
                        layout_type="structured_fact",
                        verbatim_text=sent,
                        chunk_confidence=0.97,
                        raglens_risk=base_raglens,
                        label_version_date=drug.label_date,
                        published_date=drug.published,
                        drug_name_generic=drug.drug_generic,
                        drug_name_brand=drug.drug_brand,
                        atc_code=drug.atc_code,
                        smpc_section_code=loinc_code,
                        boxed_warning=drug.boxed_warning,
                        dose_values=sent_data["dose_values"],
                        dose_units=sent_data["dose_units"],
                        dose_route=sent_data["route"],
                        patient_population=sent_data["population"],
                    ))
                    seq += 1

    logger.info(f"{drug.drug_generic}: Extracted {len(artifacts)} locked dosing artifacts")
    return artifacts