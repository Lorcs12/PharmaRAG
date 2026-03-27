from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PharmQueryIntent:
    raw_query:          str
    query_type:         str
    preferred_layout:   str
    layout_filters:     list[str]
    population_filter:  Optional[str]
    drug_names:         list[str]
    label_version_gte:  Optional[str]
    label_version_lte:  Optional[str]
    wants_interaction:  bool
    wants_dosing:       bool
    wants_mechanism:    bool