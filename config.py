from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class ElasticsearchConfig:
    host:  str = "http://localhost:9200"
    index: str = "pharma_knowledge_v3"

    bulk_chunk_size: int = 400

    max_retries:     int = 3
    request_timeout: int = 60


@dataclass
class EmbeddingConfig:
    content_model: str = "BAAI/bge-base-en-v1.5"
    colbert_model:  str = "colbert-ir/colbertv2.0"

    batch_size: int = 64

    colbert_dim:        int = 128
    muvera_dim:         int = 1024
    max_colbert_tokens: int = 48


@dataclass
class IngestionConfig:
    drug_limit: Optional[int] = 200

    min_text_len: int = 80
    smpc_section_map: Dict[str, str] = field(default_factory=lambda: {
        "34068-7": "dosing",          # DOSAGE AND ADMINISTRATION
        "34069-5": "dosing",          # HOW SUPPLIED
        "43678-2": "dosing",          # DOSAGE FORMS AND STRENGTHS

        # Contraindications — VERBATIM LOCKED (legal safety statement)
        "34070-3": "contraindication",

        # Warnings and Precautions — HIGH RISK
        "43685-7": "warning",         # WARNINGS AND PRECAUTIONS
        "34071-1": "warning",         # WARNINGS (boxed)
        "42232-9": "warning",         # PRECAUTIONS
        "34084-4": "warning",         # ADVERSE REACTIONS

        # Drug interactions — HIGH RISK (determines safety of combination)
        "34073-7": "interaction",

        # Indications — HIGH RISK (what the drug is approved for)
        "34067-9": "indication",
        "34082-8": "indication",      # USE IN SPECIFIC POPULATIONS

        # Pharmacology — can paraphrase mechanism
        "34089-3": "pharmacology",    # DESCRIPTION
        "34090-1": "pharmacology",    # CLINICAL PHARMACOLOGY
        "34091-9": "pharmacology",    # ANIMAL PHARMACOLOGY/TOXICOLOGY
        "34092-7": "pharmacology",    # CLINICAL STUDIES

        # Patient information — plain language version
        "42229-5": "patient_info",    # PATIENT COUNSELING INFORMATION
        "59845-8": "patient_info",    # PATIENT PACKAGE INSERT

        # Administrative
        "34093-5": "references",
        "51945-4": "label_metadata",
    })

    # EMA SmPC section numbers → layout_type
    # QRD template section numbers (EU-harmonized)
    ema_section_map: Dict[str, str] = field(default_factory=lambda: {
        "4.1": "indication",          # THERAPEUTIC INDICATIONS
        "4.2": "dosing",              # POSOLOGY AND METHOD OF ADMINISTRATION
        "4.3": "contraindication",    # CONTRAINDICATIONS
        "4.4": "warning",             # SPECIAL WARNINGS AND PRECAUTIONS
        "4.5": "interaction",         # INTERACTIONS WITH OTHER MEDICINAL PRODUCTS
        "4.6": "warning",             # FERTILITY, PREGNANCY AND LACTATION
        "4.7": "warning",             # EFFECTS ON ABILITY TO DRIVE
        "4.8": "warning",             # UNDESIRABLE EFFECTS (adverse reactions)
        "4.9": "warning",             # OVERDOSE
        "5.1": "pharmacology",        # PHARMACODYNAMIC PROPERTIES
        "5.2": "pharmacology",        # PHARMACOKINETIC PROPERTIES
        "5.3": "pharmacology",        # PRECLINICAL SAFETY DATA
        "6.1": "dosing",              # LIST OF EXCIPIENTS
        "6.6": "dosing",              # SPECIAL PRECAUTIONS FOR DISPOSAL
    })

    raglens_by_layout: Dict[str, float] = field(default_factory=lambda: {
        "dosing":          0.95,   # A dose is a number. Never paraphrase.
        "contraindication": 0.90,  # Legal safety statement.
        "interaction":     0.85,   # Drug combination safety.
        "warning":         0.80,   # Regulatory warning language.
        "indication":      0.75,   # Approved use — verbatim for compliance.
        "patient_info":    0.65,   # Plain language — some flexibility.
        "pharmacology":    0.45,   # Mechanism — can paraphrase.
        "references":      0.20,
        "label_metadata":  0.10,
        "cluster_summary": 0.30,   # RAPTOR summary — our words.
    })

    raglens_verbatim_threshold: float = 0.70

    max_chunk_chars: int = 1000
    chunk_overlap:   int = 120

    raptor_threshold:   float = 0.65
    raptor_min_cluster: int   = 3


@dataclass
class APIConfig:
    openfda_base:   str   = "https://api.fda.gov/drug/label.json"
    dailymed_base:  str   = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
    openfda_limit:  int   = 100
    request_delay:  float = 0.34   
    max_retries:    int   = 3
    retry_delay:    float = 5.0

    rxnorm_base: str = "https://rxnav.nlm.nih.gov/REST"

    ema_base: str = "https://www.ema.europa.eu/en/medicines/download-medicine-data"


@dataclass
class CheckpointConfig:
    dir:        str = "./checkpoints"
    save_every: int = 50  

@dataclass
class LogConfig:
    level:        str  = "INFO"
    file:         str  = "./logs/pharma_pipeline.jsonl"
    progress_bar: bool = True


@dataclass
class PharmaConfig:
    es:          ElasticsearchConfig = field(default_factory=ElasticsearchConfig)
    embedding:   EmbeddingConfig     = field(default_factory=EmbeddingConfig)
    ingestion:   IngestionConfig     = field(default_factory=IngestionConfig)
    api:         APIConfig           = field(default_factory=APIConfig)
    checkpoint:  CheckpointConfig    = field(default_factory=CheckpointConfig)
    log:         LogConfig           = field(default_factory=LogConfig)

    top_50_drugs: List[str] = field(default_factory=lambda: [
        # Statins (cardiovascular)
        "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
        # ACE inhibitors / ARBs (hypertension)
        "lisinopril", "amlodipine", "metoprolol", "carvedilol", "losartan",
        # Diabetes
        "metformin", "glipizide", "insulin glargine", "sitagliptin",
        "empagliflozin", "semaglutide",
        # Anticoagulants
        "warfarin", "apixaban", "rivaroxaban", "clopidogrel",
        # Antibiotics
        "amoxicillin", "azithromycin", "doxycycline", "ciprofloxacin",
        "trimethoprim-sulfamethoxazole",
        # Respiratory
        "albuterol", "fluticasone", "montelukast", "tiotropium",
        # Mental health / neurology
        "sertraline", "escitalopram", "fluoxetine", "bupropion",
        "duloxetine", "gabapentin", "pregabalin", "quetiapine",
        # Pain / inflammation
        "ibuprofen", "naproxen", "acetaminophen", "tramadol",
        "celecoxib", "prednisone",
        # GI
        "omeprazole", "pantoprazole", "ondansetron",
        # Other common
        "levothyroxine", "allopurinol", "furosemide",
    ])


CFG = PharmaConfig()