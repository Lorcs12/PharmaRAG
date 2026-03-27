from dataclasses import dataclass
from typing import Optional

@dataclass
class RawChunk:
    urn_id:            str
    set_id:            str
    rxcui:             str
    layout_type:       str 
    verbatim_text:     str
    chunk_confidence:  float
    raglens_risk:      float
    label_version_date: Optional[str]
    published_date:    Optional[str]
    drug_name_generic: str
    drug_name_brand:   list
    atc_code:          str
    smpc_section_code: str
    boxed_warning:     bool
    regulatory_source: str = "fda_us"
    dose_val:          Optional[float] = None
    dose_unit:         Optional[str]   = None
    dose_route:        Optional[str]   = None
    patient_population: Optional[str]  = None