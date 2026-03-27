import re

DOSE_PATTERNS = [
    re.compile(r'(\d+\.?\d*)\s*(mg/kg|mg/m2|mcg/kg|mg|mcg|g|units?/kg|mEq|mL)', re.I),
]

ROUTE_KEYWORDS = {
    "oral": ["orally", "oral", "by mouth", "po ", "tablet", "capsule"],
    "intravenous": ["intravenous", "iv ", "i.v.", "infusion", "injection"],
    "subcutaneous": ["subcutaneous", "sc ", "s.c.", "subcut"],
    "intramuscular": ["intramuscular", "im ", "i.m."],
    "inhaled":       ["inhaled", "inhalation", "inhaler"],
    "topical":       ["topical", "apply", "cream", "ointment", "patch"],
}

POPULATION_KEYWORDS = {
    "pediatric":          ["pediatric", "children", "child", "infant", "neonatal", "age <"],
    "elderly":            ["elderly", "geriatric", "older adult", "age ≥65", "age >=65"],
    "renal_impairment":   ["renal impairment", "renal failure", "kidney", "ckd", "creatinine clearance"],
    "hepatic_impairment": ["hepatic impairment", "liver", "hepatic failure", "child-pugh"],
    "pregnancy":          ["pregnan", "lactation", "breastfeed", "nursing"],
}

FDA_FIELD_MAP = {
    "contraindications":                ("34070-3", "contraindication"),
    "warnings_and_cautions":            ("43685-7", "warning"),
    "warnings":                         ("34071-1", "warning"),
    "boxed_warning":                    ("34071-1", "warning"),
    "adverse_reactions":                ("34084-4", "warning"),
    "drug_interactions":                ("34073-7", "interaction"),
    "indications_and_usage":            ("34067-9", "indication"),
    "use_in_specific_populations":      ("34082-8", "indication"),
    "clinical_pharmacology":            ("34090-1", "pharmacology"),
    "description":                      ("34089-3", "pharmacology"),
    "patient_counseling_information":   ("42229-5", "patient_info"),
}