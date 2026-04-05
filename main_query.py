import sys
from query_engine import ReflectivePharmaQueryEngine, generate_google_ai_studio_answer

TEST_QUERIES = [
    "What is the standard dose of apixaban for DVT treatment in adults",
]

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else TEST_QUERIES[0]

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
        answer = generate_google_ai_studio_answer(prompt)
        print(answer)
    except Exception as exc:
        print(f"  ERROR: {exc}")