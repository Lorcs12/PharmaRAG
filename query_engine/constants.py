from __future__ import annotations

import re

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

MEDICAL_CONCEPT_MAP: dict[str, tuple[str, str | None]] = {
    "dose":              ("dosing",          None),
    "dosage":            ("dosing",          None),
    "dosing":            ("dosing",          None),
    "how much":          ("dosing",          None),
    "how often":         ("dosing",          None),
    "serum concentration": ("dosing",        None),
    "serum level":       ("dosing",          None),
    "target concentration": ("dosing",       None),
    "target level":      ("dosing",          None),
    "therapeutic range": ("dosing",          None),
    "target range":      ("dosing",          None),
    "monitoring frequency": ("dosing",       None),
    "monitor":           ("dosing",          None),
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
    "indicated for":     ("indication",      None),
    "what conditions":   ("indication",      None),
    "used for":          ("indication",      None),
    "for treatment of":  ("indication",      None),
    "approved for":      ("indication",      None),
    "treats":            ("indication",      None),
    "mechanism":         ("pharmacology",    None),
    "how does":          ("pharmacology",    None),
    "works by":          ("pharmacology",    None),
    "renal":             ("dosing",          "renal_impairment"),
    "kidney":            ("dosing",          "renal_impairment"),
    "hepatic":           ("dosing",          "hepatic_impairment"),
    "liver":             ("dosing",          "hepatic_impairment"),
    "neonatal":          ("dosing",          "neonatal"),
    "neonate":           ("dosing",          "neonatal"),
    "newborn":           ("dosing",          "neonatal"),
    "infant":            ("dosing",          "neonatal"),
    "pediatric":         ("dosing",          "pediatric"),
    "children":          ("dosing",          "pediatric"),
    "elderly":           ("dosing",          "elderly"),
    "pregnancy":         ("contraindication", "pregnancy"),
    "pregnant":          ("contraindication", "pregnancy"),
    "lactation":         ("contraindication", "pregnancy"),
    "breastfeed":        ("contraindication", "pregnancy"),
    "nursing":           ("contraindication", "pregnancy"),
    "taper":             ("warning",         None),
    "tapering":          ("warning",         None),
    "discontinuation":   ("warning",         None),
    "discontinue":       ("warning",         None),
    "withdrawal":        ("warning",         None),
    "ARIA":              ("warning",         None),
    "surveillance":      ("warning",         None),
    "biosimilar":        ("dosing",          None),
    "formulation":       ("dosing",          None),
    "strength":          ("dosing",          None),
    "available strengths": ("dosing",        None),
    "toxicity":          ("warning",         None),
    "overdose":          ("warning",         None),
    "tumorigenicity":    ("warning",         None),
}


LABEL_RECENCY_YEARS: int = 3
BM25_BOOST: float = 1.2
MUVERA_BOOST: float = 1.0
RECENCY_BOOST: float = 0.8
STRUCTURED_FACT_BOOST: float = 1.6
CLINICAL_LAYOUT_MATCH_BOOST: float = 1.4
CLINICAL_POPULATION_MATCH_BOOST: float = 1.3
CLINICAL_DOSE_METADATA_BOOST: float = 1.2
INTERACTION_LAYOUT_OVERRIDE_BOOST: float = 1.45
INTERACTION_DOSING_PENALTY: float = 0.65


POPULATION_ALIASES: dict[str, list[str]] = {
    "renal_impairment":   ["renal_impairment", "renal"],
    "neonatal":           ["neonatal", "neonate", "newborn"],
    "pediatric":          ["pediatric", "peds", "paediatric"],
    "hepatic_impairment": ["hepatic_impairment", "hepatic"],
    "elderly":            ["elderly", "geriatric"],
    "pregnancy":          ["pregnancy", "pregnant"],
    "obese":              ["obese", "obesity"],
}

POPULATION_FUZZY_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"neonat|neonate|newborn|\b\d+\s*-?week-?old\b|\b\d+\s*-?day-?old\b", re.I), "neonatal"),
    (re.compile(r"kid(s|do|dos)?|child(ren)?|paed|pedi(atric)?|infant|baby|babies", re.I), "pediatric"),
    (re.compile(r"elder(ly)?|geriatric|older adult|senior|age\s*[>=>]=?\s*65", re.I), "elderly"),
    (re.compile(r"renal|kidney|ckd|egfr|gfr|creatinine|nephro", re.I), "renal_impairment"),
    (re.compile(r"hepat|liver|child.pugh|cirr|jaundic", re.I), "hepatic_impairment"),
    (re.compile(r"pregnan|lact|breastfeed|nursing|obstetric", re.I), "pregnancy"),
    (re.compile(r"obes|bmi\s*[>>=]\s*3", re.I), "obese"),
]

POPULATION_EQUIVALENTS: dict[str, set[str]] = {
    "neonatal": {"neonatal", "neonate", "newborn"},
    "pediatric": {"pediatric", "peds", "paediatric", "pediatric_population"},
    "elderly": {"elderly", "geriatric"},
    "renal_impairment": {"renal_impairment", "renal"},
    "hepatic_impairment": {"hepatic_impairment", "hepatic"},
    "pregnancy": {"pregnancy", "pregnant"},
}

NUMERIC_TITRATION_POPULATIONS: set[str] = {
    "renal_impairment",
    "hepatic_impairment",
    "neonatal",
    "pediatric",
    "elderly",
}

BRAND_HINT_MAP: dict[str, str] = {
    "victoza": "Victoza (T2DM, max 1.8 mg)",
    "saxenda": "Saxenda (weight management, max 3.0 mg)",
    "ozempic": "Ozempic (T2DM)",
    "wegovy": "Wegovy (weight management)",
    "rybelsus": "Rybelsus (oral semaglutide, T2DM)",
    "humira": "Humira (reference adalimumab)",
    "leqembi": "Leqembi (lecanemab)",
    "eliquis": "Eliquis (apixaban)",
    "chantix": "Chantix (varenicline)",
}

COMBO_SPLIT_RE = re.compile(r"\s*(?:,|\band\b|\bwith\b|\+|/|&)\s*", re.IGNORECASE)

BRAND_GENERIC_EXPANSIONS: dict[str, list[str]] = {
    "kazano": ["alogliptin", "metformin"],
    "actoplus met": ["pioglitazone", "metformin"],
    "invokamet": ["canagliflozin", "metformin"],
    "invokamet xr": ["canagliflozin", "metformin"],
    "glyburide-metformin hydrochloride": ["glyburide", "metformin"],
    "rybelsus": ["semaglutide"],
    "symbravo": ["meloxicam", "rizatriptan"],
    "vosevi": ["sofosbuvir", "velpatasvir", "voxilaprevir"],
    "wakix": ["pitolisant"],
    "lysodren": ["mitotane"],
    "purixan": ["mercaptopurine"],
    "venlafaxine": ["venlafaxine hcl", "venlafaxine hydrochloride"],
    "sumatriptan and naproxen sodium": ["sumatriptan", "naproxen"],
}

INDICATION_QUERY_RE = re.compile(
    r"(?:indicated\s+for|what\s+conditions?|used\s+for|for\s+treatment\s+of)",
    re.IGNORECASE,
)

DOSING_QUERY_RE = re.compile(
    r"(?:\bdose\b|\bdosage\b|\bdosing\b|\btitrate\b|\bstarting\s+dose\b|\bmaximum\s+recommended\b|\bmg\b|\bmcg\b|\begfr\b)",
    re.IGNORECASE,
)

SAFETY_INTERACTION_QUERY_RE = re.compile(
    r"(?:\brisk\b|\binteraction\b|\binteractions\b|\bcontraindication\b|\bcontraindications\b|\bwarning\b|\bwarnings\b|\bprecaution\b|\bprecautions\b|\badverse\b|\bside\s+effect\b|\bbleeding\b|\bwarfarin\b|\bavoid\b|\bconcomitant\b)",
    re.IGNORECASE,
)

INTERACTION_QUERY_RE = re.compile(
    r"(?:\binteraction\b|\binteractions\b|\binteract\b|\bconcomitant\b|\bco-?administer(?:ed|ing)?\b|\bdrug-?drug\b|\bddi\b|\bavoid(?:ed|ing)?\s+with\b|\btogether\s+with\b|\bcombine(?:d)?\s+with\b)",
    re.IGNORECASE,
)

ADVERSE_REACTION_QUERY_RE = re.compile(
    r"(?:\badverse\s+reaction|\bside\s+effects?\b|\badverse\s+event|\bdiscontinuation\s+(?:rate|percentage|percent|due)|\bstopped\s+due\s+to|\bincidence\s+of\b|\bfrequency\s+of\s+(?:adverse|side)|\bmost\s+common\s+(?:adverse|side|reaction|effect)|\btolerability|\bsafety\s+profile)",
    re.IGNORECASE,
)

SUPPLY_QUERY_RE = re.compile(
    r"(?:\bhow\s+supplied\b|\bsupplied\s+as\b|\bdosage\s+forms?\b|\bavailable\s+(?:in\s+)?(?:strength|tablet|capsule|formulation)|\bwhich\s+strengths?\b|\bwhat\s+strengths?\b|\bpackag(?:ing|ed)?\b|\bformulations?\s+available|\bmg\s+(?:tablet|capsule|extended.release|immediate.release)s?\s+(?:available|only|include))",
    re.IGNORECASE,
)

BOXED_WARNING_QUERY_RE = re.compile(
    r"(?:boxed\s+warn|black\s+box|verbatim\s+.*warn|verbatim\s+.*boxed|verbatim\s+.*contraindication|verbatim\s+.*warning)",
    re.IGNORECASE,
)

WARNING_NUMERICS_QUERY_RE = re.compile(
    r"(?:toxicit|serum\s+level|monitoring\s+(?:require|protocol|frequency)|ARIA|tumorigenicit|taper\s+schedule|discontinuation\s+.*(?:danger|schedule)|overdos|symptoms?\s+(?:are\s+)?listed)",
    re.IGNORECASE,
)

MAX_REFLECTION_ROUNDS: int = 3
MIN_ACCEPTABLE_HITS: int = 3
MIN_MAXSIM_FLOOR: float = 0.38
MIN_DOSE_EVIDENCE_NODES: int = 2
MIN_TARGET_DRUG_HITS: int = 2
SATURATION_DELTA_PCT: float = 0.02
HIGH_CONF_FIXED_DOSE_MAXSIM: float = 0.58
POPULATION_RELAX_THRESHOLD: int = 1

FALLBACK_LAYOUT_EXPANSION: dict[str, list[str]] = {
    "dosing": ["warning", "indication", "pharmacology"],
    "contraindication": ["warning", "interaction"],
    "interaction": ["warning", "pharmacology"],
    "warning": ["indication", "pharmacology"],
    "indication": ["dosing", "warning"],
    "pharmacology": ["indication", "warning"],
    "structured_fact": ["dosing", "warning"],
}

ABBREV_EXPANSION: dict[str, str] = {
    r"\bckd\b": "chronic kidney disease renal impairment",
    r"\bhf\b": "heart failure",
    r"\bmi\b": "myocardial infarction",
    r"\bafib\b": "atrial fibrillation",
    r"\bdm\b": "diabetes mellitus",
    r"\bhtn\b": "hypertension",
    r"\bpeds\b": "pediatric children",
    r"\bpo\b": "oral by mouth",
    r"\biv\b": "intravenous",
    r"\bsc\b": "subcutaneous",
    r"\btid\b": "three times daily",
    r"\bbid\b": "twice daily",
    r"\bqd\b": "once daily",
    r"\bprn\b": "as needed",
    r"\bmax\b": "maximum",
    r"\bmin\b": "minimum",
    r"\bped\b": "pediatric",
    r"\begfr\b": "estimated glomerular filtration rate renal function",
    r"\bgfr\b": "glomerular filtration rate",
    r"\bnsaid\b": "non-steroidal anti-inflammatory drug",
    r"\bssri\b": "selective serotonin reuptake inhibitor",
    r"\bmaoi\b": "monoamine oxidase inhibitor",
    r"\bace\b": "angiotensin converting enzyme inhibitor",
    r"\barb\b": "angiotensin receptor blocker",
}

FIXED_DOSE_STATEMENT_RE = re.compile(
    r"(?:recommended\s+dose\s+is|dose\s+is)\s[^.]{0,80}\d+\.?\d*\s*mg"
    r"|\d+\.?\d*\s*mg\s+(?:once|twice|three\s+times)\s+(?:daily|a\s+day)",
    re.IGNORECASE,
)

ABSENCE_SEEKING_RE = re.compile(
    r"\b(?:insufficient\s+evidence|not\s+recommended|contraindicat|"
    r"not\s+established|does\s+the\s+label\s+(?:specify|provide)|"
    r"is\s+there\s+(?:any\s+)?(?:dose|dosing)|no\s+(?:dose|dosing|data)|"
    r"off-?label|neonat|newborn|"
    r"type\s*1\s*diabetes|child\s+under\s+\d|under\s+\d\s+years?\s+of\s+age|"
    r"bmi\s+(?:greater|over|above|>)|missed\s+.*doses?|catch-?up\s+dos|"
    r"covid|sars-cov|if\s+not)\b",
    re.IGNORECASE,
)

EXPLICIT_DOSING_INTENT_RE = re.compile(
    r"\b(?:dose|dosing|dosage|titrat|starting\s+dose|initial\s+dose|"
    r"maximum\s+dose|mg/?kg|mg/day|mg\b)\b",
    re.IGNORECASE,
)

MONITORING_FOCUSED_RE = re.compile(
    r"\b(?:monitor|monitoring|serum\s+concentration|levels?|"
    r"therapeutic\s+range|clearance|renal\s+function)\b",
    re.IGNORECASE,
)

EXPLICIT_REGIMEN_RE = re.compile(
    r"\b(?:titrat|schedule|regimen|week-by-week|day\s*\d+|starting\s+dose|initial\s+dose|max(?:imum)?\s+dose)\b",
    re.IGNORECASE,
)

LOINC_DIVERSITY_MATRIX: dict[tuple[str, str], int] = {
    ("factual",    "broad"):   3,
    ("factual",    "narrow"):  1,
    ("causal",     "broad"):   2,
    ("causal",     "narrow"):  1,
    ("comparative","broad"):   2,
    ("comparative","narrow"):  1,
    ("strategic",  "broad"):   2,
    ("strategic",  "narrow"):  1,
}

BROAD_INTENT_RE = re.compile(
    r"\b(?:regimen|schedule|overview|profile|summary|guide|protocol|"
    r"all\s+dosing|complete\s+dosing|how\s+(?:should|do|does|is)|"
    r"what\s+(?:is\s+the\s+)?(?:recommended|standard|typical)\s+dosage|"
    r"tell\s+me\s+about|walk\s+me\s+through|explain\s+the\s+dosing)\b",
    re.IGNORECASE,
)

DOSE_CLAIM_RE = re.compile(
    r'\d+(?:\.\d+)?\s*(?:mg/kg|mg/m2|mcg/kg|mg|mcg|g\b|mL|units?/kg)',
    re.I,
)
XML_WRAPPER_RE = re.compile(
    r'^\s*<(?:output|response|answer)>(.*)</(?:output|response|answer)>\s*$',
    re.S | re.I,
)
INSUFF_RE = re.compile(
    r'(?:⚠\s*\*{0,2}\s*)?INSUFFICIENT\s+EVIDENCE\b',
    re.I,
)
INSUFFICIENT_EVIDENCE_EXACT = (
    "⚠ INSUFFICIENT EVIDENCE: The provided label data does not contain "
    "enough information to answer safely."
)
INSUFF_SENTENCE_RE = re.compile(
    r'(?:⚠\s*\*{0,2}\s*)?INSUFFICIENT\s+EVIDENCE\*{0,2}:?[^.\n]*\.?',
    re.I,
)

DRUG_NAME_STOP_WORDS: set[str] = {
    "what", "which", "does", "when", "where", "how", "who", "why",
    "this", "that", "these", "those", "with", "from", "they", "them",
    "their", "have", "been", "will", "should", "would", "could",
    "please", "note", "also", "drug", "dose", "oral", "take",
    "provide", "describe", "detail", "outline", "quote", "retrieve",
    "identify", "explain", "compare", "list", "state", "tell", "summarize",
}

SEQ_TITRATION_RE = re.compile(
    r"(?:"
    r"weeks?\s*\d+.{0,200}?weeks?\s*\d+"
    r"|days?\s*\d+.{0,200}?days?\s*\d+"
    r"|week(?:s)?\s+(?:daily\s+)?dose.{0,300}?\d+\s*(?:mg|mcg).{0,80}?\d+\s*(?:mg|mcg)"
    r"|week(?:s)?\s+\d+.{0,80}?\d+\s*(?:mg|mcg).{0,80}?\d+\s*(?:mg|mcg)"
    r"|initiat.{0,150}?(?:increase.{0,60}?\d+\s*(?:mg|mcg)|after\s+\d+\s+days?.{0,80}?(?:maximum|increase))"
    r"|starting.{0,150}?(?:titrat|escalat|increase|adjust)"
    r"|days?\s*\d+[-–]\d+.{0,100}?days?\s*\d+[-–]\d+"
    r")",
    re.IGNORECASE | re.DOTALL,
)
INIT_DOSE_RE = re.compile(
    r"(?:initiat|starting|initial|begin).{0,80}\d+\.?\d*\s*(?:mg|mcg)",
    re.IGNORECASE,
)
MAX_DOSE_RE = re.compile(
    r"(?:"
    r"maximum\s+(?:recommended\s+)?(?:dosage|dose)"
    r"|max(?:imum)?\s+(?:dose|dosage)"
    r"|highest\s+(?:recommended\s+)?(?:dose|dosage)"
    r"|(?:recommended|maintenance)\s+(?:dosage|dose)(?:\s+is|\s+of)?"
    r"|not\s+to\s+exceed"
    r"|up\s+to\s+\d+\.?\d*\s*(?:mg|mcg)"
    r")"
    r".{0,80}\d+\.?\d*\s*(?:mg|mcg)",
    re.IGNORECASE,
)

NUMERIC_DOSE_SIGNAL_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?|mL)(?:\s*/\s*(?:kg|day|dose|hr|h))?\b",
    re.IGNORECASE,
)
