from typing import Optional
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers

from config import CFG
from checkpoint import Checkpoint
from embedder import embed_batch, build_colbert_tokens, build_muvera_fde
from .es_schema import create_index, INDEX_NAME
from logger import get_logger, Timer
from .clients.daily_med_cl import DailyMedClient
from .clients.rx_norm_cl import RxNormClient
from .models import RawChunk, DrugLabel, IndexDoc

from .artifact_extractors import (
    extract_hierarchical_dosing_artifacts
)

log = get_logger("pipeline", CFG.log.file, CFG.log.level)

class PharmaIngestionPipeline:
    def __init__(self, drug_set: str = "top50"):
        self.drug_set = drug_set
        self.ckpt     = Checkpoint(drug_set, CFG.checkpoint.dir)
        self.dm       = DailyMedClient()
        self.rxnorm   = RxNormClient()
        self.es       = Elasticsearch(CFG.es.host)

        if not self.es.ping():
            raise RuntimeError(f"Cannot reach Elasticsearch at {CFG.es.host}")
        create_index(self.es, INDEX_NAME, force=False)

    def _resolve_clinical_entity(self, drug_name: str) -> Optional[tuple[DrugLabel, dict]]:
        log.info(f"Resolving Entity: {drug_name}")
        rxcui = self.rxnorm.get_rxcui(drug_name)
        if not rxcui:
            return None
        
        atc_code = self.rxnorm.get_atc_code(rxcui) or "ZZZZ"
        label_resp = self.dm.get_label_by_generic_name(drug_name)
        
        if not label_resp or not label_resp.get("results"):
            label_resp = self.dm.search_drug(drug_name)
        if not label_resp or not label_resp.get("results"):
            return None

        result = label_resp["results"][0]
        openfda = result.get("openfda", {})
        
        eff_time = result.get("effective_time", "")
        label_date = f"{eff_time[:4]}-{eff_time[4:6]}-{eff_time[6:8]}" if len(eff_time) == 8 else None

        drug_label = DrugLabel(
            set_id=result.get("set_id", f"unknown-{rxcui}"),
            rxcui=rxcui,
            drug_generic=(openfda.get("generic_name", [drug_name])[0]).lower(),
            drug_brand=[b.title() for b in openfda.get("brand_name", [])],
            atc_code=atc_code,
            label_date=label_date,
            published=label_date,
            boxed_warning=bool(result.get("boxed_warning")),
        )
        return drug_label, label_resp

    def _generate_tripartite_embeddings(self, artifacts: list[RawChunk]) -> list[IndexDoc]:
        if not artifacts:
            return []

        texts = [c.verbatim_text for c in artifacts]
        with Timer(log, "tripartite_embedding_batch", n=len(texts)):
            content_vecs = embed_batch(texts)

        docs = []
        for artifact, cvec in zip(artifacts, content_vecs):
            colbert_toks = build_colbert_tokens(artifact.verbatim_text)
            muvera_fde   = build_muvera_fde(colbert_toks)

            source = {
                "urn_id":             artifact.urn_id,
                "set_id":             artifact.set_id,
                "rxcui":              artifact.rxcui,
                "layout_type":        artifact.layout_type,
                "verbatim_text":      artifact.verbatim_text,
                "chunk_confidence":   artifact.chunk_confidence,
                "raglens_risk":       artifact.raglens_risk,
                "drug_name_generic":  artifact.drug_name_generic,
                "drug_name_brand":    artifact.drug_name_brand,
                "atc_code":           artifact.atc_code,
                "smpc_section_code":  artifact.smpc_section_code,
                "boxed_warning":      artifact.boxed_warning,
                "content_vector":     cvec,
                "colbert_tokens":     colbert_toks,
                "muvera_fde":         muvera_fde,
            }
            for field in ["label_version_date", "dose_values", "dose_units", "dose_route", "patient_population", "parent_urn"]:
                value = getattr(artifact, field, None)
                if value is None:
                    continue
                if isinstance(value, list) and not value:
                    continue
                source[field] = value

            docs.append(IndexDoc(source=source))
        return docs

    def _execute_drug_ingestion_protocol(self, drug_name: str) -> tuple[int, Optional[DrugLabel]]:
        resolution = self._resolve_clinical_entity(drug_name)
        if not resolution:
            return 0, None
        
        drug, label_data = resolution

        if self.ckpt.is_set_id_done(drug.set_id):
            return 0, drug

        all_artifacts = extract_hierarchical_dosing_artifacts(label_data, drug, log)
        
        if not all_artifacts:
            return 0, drug

        docs = self._generate_tripartite_embeddings(all_artifacts)
        
        actions = [{"_index": INDEX_NAME, "_id": d.source["urn_id"], "_source": d.source} for d in docs]
        success, _ = helpers.bulk(self.es, actions, raise_on_error=False)
        
        self.ckpt.mark_set_id_done(drug.set_id, drug.drug_generic, success)
        return success, drug

    def execute_batch_ingestion(self, drug_names: list[str]) -> dict:
        total_indexed = 0
        failed_drugs = []
        log.info(f"Initiating PharmaRAG Ingestion Protocol | Target: {len(drug_names)} entities")

        for drug in tqdm(drug_names, desc="Ingesting Clinical Artifacts"):
            try:
                n_docs, _ = self._execute_drug_ingestion_protocol(drug)
                total_indexed += n_docs
            except Exception as e:
                log.error(f"Ingestion Failure for {drug}: {e}")
                failed_drugs.append(drug)

        log.info(f"Protocol Complete: {total_indexed} artifacts indexed.")
        return {"total": total_indexed, "failed": failed_drugs}