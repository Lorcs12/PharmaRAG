import os
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

from config import CFG
from logger import Timer, get_logger
from .utils import normalize_llm_output

log = get_logger("azure_openai", CFG.log.file, CFG.log.level)



class AzureOpenAIConnector:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("AZURE_OPENAI_MODEL", CFG.llm_azure.model)
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", CFG.llm_azure.api_version)
        self._endpoint = self._load_endpoint()
        self._api_key = self._load_api_key()
        self._client = AzureOpenAI(
            api_key=self._api_key,
            api_version=self.api_version,
            azure_endpoint=self._endpoint,
        )

    def _load_endpoint(self) -> str:
        endpoint = os.getenv(CFG.llm_azure.endpoint_env) or os.getenv(CFG.llm_azure.endpoint_env.upper())
        if not endpoint:
            raise RuntimeError(
                f"Missing Azure OpenAI endpoint. Add {CFG.llm_azure.endpoint_env} to your .env file."
            )
        return endpoint

    def _load_api_key(self) -> str:
        api_key = os.getenv(CFG.llm_azure.api_key_env) or os.getenv(CFG.llm_azure.api_key_env.upper())
        if not api_key:
            raise RuntimeError(
                f"Missing Azure OpenAI API key. Add {CFG.llm_azure.api_key_env} to your .env file."
            )
        return api_key

    def generate(self, prompt: str, use_json_mode: bool = False) -> str:
        with Timer(log, "azure_openai_generate", model=self.model):
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }

            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self._client.chat.completions.create(**kwargs)

        text = self._extract_text(response)

        if use_json_mode:
            return text

        return self._normalize_constrained_output(text)

    @staticmethod
    def _extract_text(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("Azure OpenAI returned no choices.")

        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", None)
        if not text:
            raise RuntimeError("Azure OpenAI returned an empty response.")
        return str(text).strip()

    @staticmethod
    def _normalize_constrained_output(text: str) -> str:
        return normalize_llm_output(text)


def generate_azure_openai_answer(prompt: str, model: str | None = None) -> str:
    connector = AzureOpenAIConnector(model=model)
    return connector.generate(prompt)
