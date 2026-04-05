from typing import Optional

QUERY_TYPE_FACTUAL     = "factual"
QUERY_TYPE_CAUSAL      = "causal"
QUERY_TYPE_COMPARATIVE = "comparative" 
QUERY_TYPE_STRATEGIC   = "strategic"

CAUSAL_SIGNALS = {
    "why", "cause", "mechanism", "reason", "because", "interaction",
    "interact", "combined with", "together with", "due to",
    "serotonin syndrome", "contraindicated", "avoid"
}
COMPARATIVE_SIGNALS = {
    "compare", "vs", "versus", "difference", "alternative", "instead",
    "better", "preferred", "first-line", "second-line", "relative"
}
STRATEGIC_SIGNALS = {
    "overall", "general", "profile", "class", "all", "typical",
    "commonly", "safety", "efficacy", "overview", "summary"
}

MEDICAL_CONCEPT_MAP: dict[str, tuple[str, Optional[str]]] = {
    "dose":              ("dosing",          None),
    "dosage":            ("dosing",          None),
    "dosing":            ("dosing",          None),
    "how much":          ("dosing",          None),
    "how often":         ("dosing",          None),
    "contraindication":  ("contraindication", None),
    "contraindicated":   ("contraindication", None),
    "avoid":             ("contraindication", None),
    "interaction":       ("interaction",     None),
    "interact":          ("interaction",     None),
    "interacts":         ("interaction",     None),
    "combine":           ("interaction",     None),
    "together":          ("interaction",     None),
    "side effect":       ("warning",         None),
    "adverse":           ("warning",         None),
    "warning":           ("warning",         None),
    "safe":              ("warning",         None),
    "indication":        ("indication",      None),
    "approved for":      ("indication",      None),
    "treats":            ("indication",      None),
    "mechanism":         ("pharmacology",    None),
    "how does":          ("pharmacology",    None),
    "works by":          ("pharmacology",    None),
    "renal":             ("dosing",          "renal_impairment"),
    "kidney":            ("dosing",          "renal_impairment"),
    "hepatic":           ("dosing",          "hepatic_impairment"),
    "liver":             ("dosing",          "hepatic_impairment"),
    "pediatric":         ("dosing",          "pediatric"),
    "children":          ("dosing",          "pediatric"),
    "elderly":           ("dosing",          "elderly"),
    "pregnancy":         ("contraindication", "pregnancy"),
    "pregnant":          ("contraindication", "pregnancy"),
    "lactation":         ("contraindication", "pregnancy"),
    "breastfeed":        ("contraindication", "pregnancy"),
    "nursing":           ("contraindication", "pregnancy"),
}