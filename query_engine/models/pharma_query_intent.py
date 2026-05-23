from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class PharmQueryIntent:
    raw_query:          str
    query_type:         str
    preferred_layout:   str
    layout_filters:     list[str]
    population_filter:  str | None
    drug_names:         list[str]
    label_version_gte:  str | None
    label_version_lte:  str | None
    wants_interaction:  bool
    wants_dosing:       bool
    wants_mechanism:    bool
    drug_lockdown_exact: bool = False
    wants_boxed_warning: bool = False
    wants_warning_numerics: bool = False
    brand_hint: str | None = None