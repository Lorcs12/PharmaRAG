import re
from typing import Optional
import numpy as np
from config import CFG
from .models import RawChunk, DrugLabel
from .helpers import strip_html
from .clinical_parsers import (
    parse_dose_value, parse_clinical_route, 
    parse_patient_population, execute_semantic_chunking
)
from .ingestion_constants import FDA_FIELD_MAP

def generate_artifact_urn(rxcui: str, set_id: str, layout_type: str, regulatory_source: str, seq: int) -> str:
    set_prefix = set_id.replace("-", "")[:8]
    return f"urn:pharma:{regulatory_source}:{rxcui}:{set_prefix}:{layout_type}:{seq:04d}"

def generate_dosing_urn(rxcui: str, set_id: str, route: str, population: str, seq: int) -> str:
    set_prefix = set_id.replace("-", "")[:8]
    route_slug = (route or "unspecified").replace(" ", "_")
    pop_slug   = (population or "general").replace(" ", "_")
    return f"urn:pharma:fda_us:{rxcui}:{set_prefix}:dosing_fact:{route_slug}:{pop_slug}:{seq:04d}"

def extract_structured_dosing_artifacts(label_data: dict, drug: DrugLabel, logger) -> list[RawChunk]:
    artifacts = []
    results = label_data.get("results", [{}])
    if not results:
        return artifacts

    dosage_sections = results[0].get("dosage_and_administration", [])
    seq = 0
    for section_text in dosage_sections:
        clean_text = strip_html(section_text)
        if len(clean_text) < CFG.ingestion.min_text_len:
            continue

        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 30 or not any(c.isdigit() for c in sent):
                continue

            dose_val, dose_unit = parse_dose_value(sent)
            route               = parse_clinical_route(sent)
            population          = parse_patient_population(sent)

            urn = generate_dosing_urn(drug.rxcui, drug.set_id, route, population, seq)

            raglens_risk = CFG.ingestion.raglens_by_layout["dosing"]
            if drug.boxed_warning:
                raglens_risk = min(1.0, raglens_risk + 0.05)

            artifacts.append(RawChunk(
                urn_id=urn,
                set_id=drug.set_id,
                rxcui=drug.rxcui,
                layout_type="dosing",
                verbatim_text=sent,
                chunk_confidence=0.97,
                raglens_risk=raglens_risk,
                label_version_date=drug.label_date,
                published_date=drug.published,
                drug_name_generic=drug.drug_generic,
                drug_name_brand=drug.drug_brand,
                atc_code=drug.atc_code,
                smpc_section_code="34068-7",
                boxed_warning=drug.boxed_warning,
                dose_val=dose_val,
                dose_unit=dose_unit,
                dose_route=route,
                patient_population=population,
            ))
            seq += 1

    logger.info(f"{drug.drug_generic}: Extracted {len(artifacts)} locked dosing artifacts")
    return artifacts

def extract_semantic_prose_artifacts(label_data: dict, drug: DrugLabel, logger) -> list[RawChunk]:
    artifacts = []
    results = label_data.get("results", [{}])
    if not results:
        return artifacts
    
    result = results[0]
    seq = 0
    
    for field_name, (loinc_code, layout_type) in FDA_FIELD_MAP.items():
        sections = result.get(field_name, [])
        if not sections:
            continue

        base_raglens = CFG.ingestion.raglens_by_layout.get(layout_type, 0.50)
        if drug.boxed_warning:
            base_raglens = min(1.0, base_raglens + 0.05)

        full_text = " ".join(strip_html(s) for s in sections)
        text_chunks = execute_semantic_chunking(full_text)
        
        for chunk_text in text_chunks:
            if len(chunk_text) < CFG.ingestion.min_text_len:
                continue

            urn = generate_artifact_urn(drug.rxcui, drug.set_id, layout_type, drug.source, seq)

            artifacts.append(RawChunk(
                urn_id=urn,
                set_id=drug.set_id,
                rxcui=drug.rxcui,
                layout_type=layout_type,
                verbatim_text=chunk_text,
                chunk_confidence=0.95,
                raglens_risk=base_raglens,
                label_version_date=drug.label_date,
                published_date=drug.published,
                drug_name_generic=drug.drug_generic,
                drug_name_brand=drug.drug_brand,
                atc_code=drug.atc_code,
                smpc_section_code=loinc_code,
                boxed_warning=drug.boxed_warning,
                dose_route=parse_clinical_route(chunk_text),
                patient_population=parse_patient_population(chunk_text),
            ))
            seq += 1

    logger.info(f"{drug.drug_generic}: Extracted {len(artifacts)} semantic prose artifacts")
    return artifacts