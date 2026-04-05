import os
from typing import Any

from config import CFG
from logger import get_logger, Timer
from google import genai


from dotenv import load_dotenv

load_dotenv()

log = get_logger("google_ai_studio", CFG.log.file, CFG.log.level)


class GoogleAIStudioConnector:
    def __init__(self, model: str | None = None):
        self.model = model or CFG.llm.model
        self._api_key = self._load_api_key()

    def _load_api_key(self) -> str:
        api_key = os.getenv(CFG.llm.api_key_env) or os.getenv(CFG.llm.api_key_env.upper())
        if not api_key:
            raise RuntimeError(
                f"Missing Google AI Studio API key. Add {CFG.llm.api_key_env} to your .env file."
            )
        return api_key

    def generate(self, prompt: str) -> str:
        with Timer(log, "google_ai_studio_generate", model=self.model):
            client = genai.Client(api_key=self._api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

        text = self._extract_text(response)
        if not text:
            raise RuntimeError("Google AI Studio returned an empty response.")
        return text

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)
        return "\n".join(parts).strip()


def generate_google_ai_studio_answer(prompt: str, model: str | None = None) -> str:
    connector = GoogleAIStudioConnector(model=model)
    return connector.generate(prompt)
