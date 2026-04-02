from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RetrievedNode:
    urn:               str
    layout_type:       str
    verbatim_text:     str
    label_version_date: Optional[str]
    drug_name_generic: str
    drug_name_brand:   list
    rxcui:             str
    atc_code:          str
    raglens_risk:      float
    chunk_confidence:  float
    dose_values:       list[float]
    dose_units:        list[str]
    dose_route:        Optional[str]
    patient_population: Optional[str]
    boxed_warning:     bool
    interaction_ids:   list[str]
    table_ref:         list[str]
    raptor_cluster:    Optional[str]
    smpc_section_code: str
    maxsim_score:      float = 0.0

    verbatim_locked:     bool = False
    confidence_verified: bool = False
    verified_dose_values: list[float] = field(default_factory=list)

    @classmethod
    def from_es_hit(cls, source: dict, maxsim_score: float = 0.0) -> "RetrievedNode":
        dose_values = source.get("dose_values")
        if dose_values is None and source.get("dose_val") is not None:
            dose_values = [source["dose_val"]]

        dose_units = source.get("dose_units")
        if dose_units is None and source.get("dose_unit") is not None:
            dose_units = [source["dose_unit"]]

        return cls(
            urn                = source.get("urn_id", ""),
            layout_type        = source.get("layout_type", ""),
            verbatim_text      = source.get("verbatim_text", ""),
            label_version_date = source.get("label_version_date"),
            drug_name_generic  = source.get("drug_name_generic", ""),
            drug_name_brand    = source.get("drug_name_brand") or [],
            rxcui              = source.get("rxcui", ""),
            atc_code           = source.get("atc_code", ""),
            raglens_risk       = float(source.get("raglens_risk", 0.0)),
            chunk_confidence   = float(source.get("chunk_confidence", 1.0)),
            dose_values        = dose_values or [],
            dose_units         = dose_units or [],
            dose_route         = source.get("dose_route"),
            patient_population = source.get("patient_population"),
            boxed_warning      = bool(source.get("boxed_warning", False)),
            interaction_ids    = source.get("interaction_ids") or [],
            table_ref          = source.get("table_ref") or [],
            raptor_cluster     = source.get("raptor_cluster"),
            smpc_section_code  = source.get("smpc_section_code", ""),
            maxsim_score       = maxsim_score,
        )