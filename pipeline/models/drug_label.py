from dataclasses import dataclass

@dataclass
class DrugLabel:
    """Minimal metadata for one FDA label fetch."""
    set_id:       str
    rxcui:        str
    drug_generic: str
    drug_brand:   list
    atc_code:     str
    label_date:   str           
    published:    str
    boxed_warning: bool
    source:       str = "fda_us"