import argparse

from query_engine import ReflectivePharmaQueryEngine, generate_llm_answer

TEST_QUERIES = [
    "What is the standard dose of apixaban?",
]

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single query through ReflectivePharmaQueryEngine.")
    parser.add_argument(
        "query",
        nargs="?",
        default=TEST_QUERIES[0],
        help="Clinical question to run.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["google_ai_studio", "azure_openai"],
        default="google_ai_studio",
        help="LLM provider to use for final answer generation.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    query = args.query

    print(f"\n{'═'*72}")
    print(f"  PharmaRAG · Advanced Clinical Query Engine")
    print(f"  Query : {query}")
    print(f"{'═'*72}\n")

    engine = ReflectivePharmaQueryEngine()
    artifact = engine.execute_query_pipeline(query)

    print(f"\n{'═'*72}")
    print("  COGNITIVE CANVAS SUMMARY")
    print(f"{'═'*72}")
    print(f"  {artifact.get_artifact_summary()}")

    reflection_rounds = getattr(artifact, "_reflection_rounds", None)
    sufficiency_reports = getattr(artifact, "_sufficiency_reports", None)
    if reflection_rounds is not None:
        print(f"\n{'═'*72}")
        print("  REFLECTIVE RETRIEVAL")
        print(f"{'═'*72}")
        print(f"  rounds: {reflection_rounds}")
        if sufficiency_reports:
            for report in sufficiency_reports:
                status = "PASS" if report.passed else "FAIL"
                failures = ", ".join(report.failures) if report.failures else "none"
                print(
                    f"  round {report.round_number}: {status} | hits={report.n_hits} | "
                    f"top_maxsim={report.top_maxsim:.3f} | failures={failures}"
                )

    print(f"\n{'═'*72}")
    print("  CONSTRAINED LLM PROMPT")
    print(f"{'═'*72}")
    prompt = artifact.generate_constrained_prompt()
    print(prompt)

    print(f"\n{'═'*72}")
    print("LLM ANSWER")
    print(f"{'═'*72}")
    try:
        answer = generate_llm_answer(prompt, provider=args.provider)
        print(answer)
    except Exception as exc:
        print(f"  ERROR: {exc}")