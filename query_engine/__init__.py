from .query_pipeline import PharmaQueryEngine
from .reflective_retrieval import ReflectivePharmaQueryEngine, SufficiencyReport
from .google_ai_studio import GoogleAIStudioConnector, generate_google_ai_studio_answer
from .azure_openai import AzureOpenAIConnector, generate_azure_openai_answer


def generate_llm_answer(prompt: str, provider: str | None = None, artifact=None, use_structured_output: bool = True) -> str:
	"""
	Generate LLM answer with structured JSON output for number extraction.

	Args:
		prompt: The LLM prompt
		provider: LLM provider ("google_ai_studio" or "azure_openai")
		artifact: Optional CogCanvasArtifact (not used with structured output)
		use_structured_output: If True, use JSON mode and parse response

	Returns:
		LLM answer with numbers formatted inline
	"""
	import json
	from config import CFG

	selected = provider or CFG.llm.provider

	if use_structured_output and selected == "azure_openai":
		# Use JSON mode
		connector = AzureOpenAIConnector()
		json_response = connector.generate(prompt, use_json_mode=True)

		try:
			data = json.loads(json_response)
			answer = data.get("answer", "")
			numeric_values = data.get("numeric_values", [])

			# Format answer with numbers clearly listed
			if numeric_values:
				# Ensure numbers are unique and sorted
				unique_nums = sorted(set(float(n) for n in numeric_values))
				nums_str = ", ".join(str(n) for n in unique_nums)
				answer = f"{answer}\n\n**Numeric values from answer:** {nums_str}"

			return answer

		except json.JSONDecodeError:
			# Fallback if JSON parsing fails
			return json_response

	# Fallback to non-structured mode
	if selected == "google_ai_studio":
		answer = generate_google_ai_studio_answer(prompt)
	elif selected == "azure_openai":
		answer = generate_azure_openai_answer(prompt)
	else:
		raise ValueError(
			f"Unknown LLM provider '{selected}'. Use 'google_ai_studio' or 'azure_openai'."
		)

	return answer