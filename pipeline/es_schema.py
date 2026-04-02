from elasticsearch import Elasticsearch
from logger import get_logger

log = get_logger("es_schema")

INDEX_NAME = "pharma_knowledge_v2"

INDEX_MAPPING = {
    "settings": {
        "index": {
            "number_of_shards":   1,
            "number_of_replicas": 0,
            "similarity": {
                "clinical_bm25": {
                    "type": "BM25",
                    "b":    0.60,
                    "k1":   1.2,
                }
            },
        },
        "analysis": {
            "filter": {
                "medical_synonym": {
                    "type": "synonym_graph",
                    "synonyms": [
                        "mi, myocardial infarction, heart attack",
                        "ckd, chronic kidney disease, renal impairment, renal failure",
                        "htn, hypertension, high blood pressure",
                        "dm, diabetes mellitus, diabetes",
                        "gi, gastrointestinal",
                        "cns, central nervous system",
                        "cv, cardiovascular",
                        "ae, adverse event, side effect, adverse reaction",
                        "bb, beta blocker, beta-blocker",
                        "ace, ace inhibitor, angiotensin converting enzyme inhibitor",
                        "arb, angiotensin receptor blocker",
                        "ssri, selective serotonin reuptake inhibitor",
                        "maoi, monoamine oxidase inhibitor",
                        "nsaid, non-steroidal anti-inflammatory drug",
                        "po, oral, by mouth",
                        "iv, intravenous, intravenously",
                        "sc, subcutaneous, subcutaneously",
                        "qd, once daily, once a day",
                        "bid, twice daily, twice a day",
                        "tid, three times daily",
                        "qid, four times daily",
                        "prn, as needed",
                    ]
                }
            },
            "analyzer": {
                "clinical_text": {
                    "type":      "custom",
                    "tokenizer": "standard",
                    "filter":    ["lowercase", "medical_synonym"],
                },
                "drug_name": {
                    "type":      "custom",
                    "tokenizer": "standard",
                    "filter":    ["lowercase", "asciifolding"],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "urn_id": {"type": "keyword"},
            "set_id": {"type": "keyword"},
            "layout_type": {"type": "keyword"},
            "verbatim_text": {
                "type":          "text",
                "analyzer":      "clinical_text",
                "index_options": "offsets",
                "fields": {
                    "exact": {"type": "keyword", "ignore_above": 4096},
                },
            },

            "label_version_date": {
                "type":   "date",
                "format": "strict_date_optional_time",
            },

            "published_date": {
                "type":   "date",
                "format": "strict_date_optional_time",
            },

            "drug_name_generic": {
                "type":     "text",
                "analyzer": "drug_name",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256},
                },
            },

            "drug_name_brand": {
                "type":     "text",
                "analyzer": "drug_name",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256},
                },
            },

            "rxcui": {"type": "keyword"},
            "atc_code": {"type": "keyword"},
            "regulatory_source": {"type": "keyword"},
            "smpc_section_code": {"type": "keyword"},
            "dose_val": {"type": "double"},
            "dose_unit": {"type": "keyword"},
            "dose_route": {"type": "keyword"},
            "patient_population": {"type": "keyword"},
            "boxed_warning": {"type": "boolean"},

            "content_vector": {
                "type":       "dense_vector",
                "dims":       768,
                "index":      True,
                "similarity": "cosine",
            },

            "muvera_fde": {
                "type":       "dense_vector",
                "dims":       1024,
                "index":      True,
                "similarity": "dot_product",
            },


            "colbert_tokens": {
                "type": "nested",
                "properties": {
                    "v": {
                        "type":  "dense_vector",
                        "dims":  128,
                        "index": False,
                    }
                },
            },

            "raglens_risk": {"type": "float"},
            "chunk_confidence": {"type": "float"},
            "raptor_cluster": {"type": "keyword"},
            "interaction_ids": {"type": "keyword"},
            "table_ref": {"type": "keyword"},
        }
    },
}


def create_index(
    es: Elasticsearch,
    index_name: str = INDEX_NAME,
    force: bool = False,
) -> str:
    if es.indices.exists(index=index_name):
        if not force:
            log.info(
                f"Index '{index_name}' already exists — skipping. "
                "Pass force=True to recreate.",
                extra={"index": index_name},
            )
            return index_name
        log.warning(f"force=True — deleting '{index_name}'", extra={"index": index_name})
        es.indices.delete(index=index_name)

    resp = es.indices.create(index=index_name, body=INDEX_MAPPING)
    log.info(f"Created index '{resp['index']}'", extra={"index": resp["index"]})
    return resp["index"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="http://localhost:9200")
    parser.add_argument("--index", default=INDEX_NAME)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    es = Elasticsearch(args.host)
    if not es.ping():
        print(f"Cannot reach Elasticsearch at {args.host}")
        raise SystemExit(1)
    create_index(es, index_name=args.index, force=args.force)