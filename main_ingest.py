import argparse
from config import CFG
from pipeline import PharmaIngestionPipeline

def main():
    parser = argparse.ArgumentParser(description="PharmaRAG — Cognitive Artifact Ingestion Pipeline")
    parser.add_argument("--drug", type=str, help="Single drug generic name to ingest")
    parser.add_argument("--set", type=str, default="top50", help="Named clinical set to ingest")
    parser.add_argument("--limit", type=int, help="Limit to first N drugs")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Force full re-ingestion")
    args = parser.parse_args()

    drug_names = [args.drug] if args.drug else CFG.top_50_drugs
    drug_set   = args.drug.replace(" ", "_") if args.drug else args.set
    if args.limit:
        drug_names = drug_names[:args.limit]

    pipeline = PharmaIngestionPipeline(drug_set=drug_set)
    if args.reset_checkpoint:
        pipeline.ckpt.reset()

    result = pipeline.execute_batch_ingestion(drug_names)
    print(f"\nIngestion Complete — {result['total']} cognitive artifacts successfully indexed.")

if __name__ == "__main__":
    main()