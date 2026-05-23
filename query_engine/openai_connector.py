"""OpenAI API connector for platform.openai.com"""
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from config import CFG
from logger import Timer, get_logger
from .utils import normalize_llm_output

log = get_logger("openai", CFG.log.file, CFG.log.level)


class OpenAIConnector:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self._api_key = self._load_api_key()
        self._client = OpenAI(api_key=self._api_key)

    def _load_api_key(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
        if not api_key:
            raise RuntimeError(
                "Missing OpenAI API key. Add OPENAI_API_KEY or openai_api_key to your .env file."
            )
        return api_key

    def generate(self, prompt: str) -> str:
        with Timer(log, "openai_generate", model=self.model):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
        text = self._extract_text(response)
        return self._normalize_constrained_output(text)

    @staticmethod
    def _extract_text(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("OpenAI returned no choices.")

        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", None)
        if not text:
            raise RuntimeError("OpenAI returned an empty response.")
        return str(text).strip()

    @staticmethod
    def _normalize_constrained_output(text: str) -> str:
        return normalize_llm_output(text)


def generate_openai_answer(prompt: str, model: str | None = None) -> str:
    connector = OpenAIConnector(model=model)
    return connector.generate(prompt)
