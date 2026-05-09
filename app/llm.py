"""Multi-provider LLM client: Ollama (local/bonus), OpenAI, Anthropic."""

import logging

import requests

from app.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — strict grounding, no hallucination, no medical advice
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a professional healthcare AI assistant for MedAssist Healthcare System.

STRICT RULES — follow without exception:
1. Answer ONLY using the CONTEXT provided below. Do not use any outside knowledge.
2. If the answer is NOT clearly in the context, respond with exactly:
   "I could not find this information in the provided documents."
3. Never diagnose medical conditions or prescribe treatments.
4. Always recommend consulting a qualified healthcare professional for personal medical decisions.
5. Keep responses concise, accurate, and professional.
6. Do not speculate, infer, or guess beyond what the context explicitly states.

CONTEXT FROM HEALTHCARE DOCUMENTS:
{context}
"""

NOT_FOUND = "I could not find this information in the provided documents."


class LLMService:
    """Synchronous LLM service supporting Ollama, OpenAI, and Anthropic."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def answer_with_context(self, question: str, context: str) -> str:
        """Generate a grounded answer using the retrieved context."""
        if not context.strip():
            return NOT_FOUND

        system = SYSTEM_PROMPT.format(context=context)
        provider = self.settings.llm_provider.lower()

        logger.info("Calling LLM provider=%s", provider)
        if provider == "ollama":
            return self._ask_ollama(system, question)
        if provider == "openai":
            return self._ask_openai(system, question)
        if provider == "anthropic":
            return self._ask_anthropic(system, question)
        if provider == "mock":
            return self._ask_mock(context)
        raise ValueError(f"Unsupported llm_provider: {provider!r}. Use ollama|openai|anthropic|mock.")

    # ------------------------------------------------------------------
    # Ollama — local on-premise LLM (no API key needed; bonus points)
    # ------------------------------------------------------------------

    def _ask_ollama(self, system: str, user: str) -> str:
        url = f"{self.settings.ollama_base_url}/api/generate"
        # Minimal, direct prompt — small models (1b/3b) respond better to
        # "read this text and answer" framing than complex rule lists.
        prompt = (
            "Read the healthcare document excerpts below and answer the question.\n"
            "Use ONLY information from the excerpts. Do not add outside knowledge.\n"
            "If the answer is not in the excerpts, say: "
            "\"I could not find this information in the provided documents.\"\n"
            "Never give a medical diagnosis or prescribe medication.\n\n"
            f"--- EXCERPTS ---\n{system.split('CONTEXT FROM HEALTHCARE DOCUMENTS:')[-1].strip()}\n"
            f"--- END EXCERPTS ---\n\n"
            f"Question: {user}\n"
            "Answer:"
        )
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.settings.llm_temperature,
                "num_predict": self.settings.llm_max_tokens,  # max output tokens
                "num_ctx": self.settings.llm_num_ctx,         # context window (prompt + output)
                "stop": ["Question:", "---", "\n\nQuestion"],  # prevent run-on generation
            },
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "").strip()

        # Log if model stopped early (truncation indicator)
        stop_reason = data.get("done_reason", "")
        if stop_reason == "length":
            logger.warning("Ollama response hit num_predict limit — consider increasing llm_max_tokens")

        return text or NOT_FOUND

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    def _ask_openai(self, system: str, user: str) -> str:
        from openai import OpenAI

        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")

        client = OpenAI(api_key=self.settings.openai_api_key, base_url=self.settings.openai_base_url)
        resp = client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        return text.strip() or NOT_FOUND

    # ------------------------------------------------------------------
    # Anthropic / Claude
    # ------------------------------------------------------------------

    def _ask_anthropic(self, system: str, user: str) -> str:
        import anthropic

        if not self.settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in .env")

        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        resp = client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=self.settings.llm_max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text if resp.content else ""
        return text.strip() or NOT_FOUND

    # ------------------------------------------------------------------
    # Mock — returns first 3 sentences from context (no LLM needed)
    # Use LLM_PROVIDER=mock in .env for local demo without any API key
    # ------------------------------------------------------------------

    def _ask_mock(self, context: str) -> str:
        if not context.strip():
            return NOT_FOUND
        # Extract first 3 sentences from the top retrieved chunk as the answer
        sentences = [s.strip() for s in context.replace("\n", " ").split(". ") if s.strip()]
        summary = ". ".join(sentences[:3])
        if summary and not summary.endswith("."):
            summary += "."
        return summary or NOT_FOUND
