import logging
import os
import re
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

SUMMARY_FALLBACK_TEXT = "Summary unavailable due to a processing error."
RELEVANCE_FALLBACK_SCORE = 5
ENGLISH_FALLBACK_SCORE = 5.0

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_LLM_NO_PROXY_HOSTS = (
    ".googleapis.com",
    ".google.com",
    "generativelanguage.googleapis.com",
    "api.groq.com",
)


def _looks_like_dead_loopback_proxy(proxy_url: str | None) -> bool:
    if not proxy_url:
        return False

    try:
        parsed = urlparse(proxy_url)
    except ValueError:
        return False

    return parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port == 9


def _disable_dead_loopback_proxy() -> None:
    proxies = {
        key: value
        for key, value in ((key, os.getenv(key)) for key in _PROXY_ENV_KEYS)
        if value
    }
    if not proxies or not any(_looks_like_dead_loopback_proxy(value) for value in proxies.values()):
        return

    for key in _PROXY_ENV_KEYS:
        os.environ.pop(key, None)

    no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
    no_proxy_items = [item.strip() for item in no_proxy.split(",") if item.strip()]
    for host in _LLM_NO_PROXY_HOSTS:
        if host not in no_proxy_items:
            no_proxy_items.append(host)

    if no_proxy_items:
        joined = ",".join(no_proxy_items)
        os.environ["NO_PROXY"] = joined
        os.environ["no_proxy"] = joined

    logger.warning("Disabled dead loopback proxy settings before LLM initialization.")


class Gemini:
    """Compatibility wrapper for the repo's existing Gemini import path.

    The active provider is selected from environment variables:
    - `LLM_PROVIDER=groq|gemini|auto`
    - `GROQ_API_KEY` enables Groq via its OpenAI-compatible endpoint.
    - `GEMINI_API_KEY` enables Google Gemini.
    """

    def __init__(self):
        load_dotenv()
        _disable_dead_loopback_proxy()

        self.provider = self._resolve_provider()
        self.model = None
        self.session = None
        self.groq_api_key = None
        self.groq_base_url = (os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
        self.groq_model = os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
        self.gemini_model = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite"
        self.request_timeout = self._resolve_timeout()

        try:
            if self.provider == "groq":
                self._init_groq()
            elif self.provider == "gemini":
                self._init_gemini()
            else:
                logger.warning(
                    "No LLM provider configured. Set GROQ_API_KEY or GEMINI_API_KEY."
                )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error(
                "LLM initialization failed for provider '%s': %s",
                self.provider or "none",
                exc,
            )
            self.provider = None

    def _resolve_provider(self) -> str | None:
        requested = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()
        has_groq = bool(os.getenv("GROQ_API_KEY"))
        has_gemini = bool(os.getenv("GEMINI_API_KEY"))

        if requested == "groq":
            if has_groq:
                return "groq"
            logger.warning("LLM_PROVIDER=groq but GROQ_API_KEY is missing.")
            return "gemini" if has_gemini else None

        if requested == "gemini":
            if has_gemini:
                return "gemini"
            logger.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is missing.")
            return "groq" if has_groq else None

        if has_groq:
            return "groq"
        if has_gemini:
            return "gemini"
        return None

    def _resolve_timeout(self) -> float:
        if self.provider == "groq":
            value = os.getenv("GROQ_TIMEOUT_SECONDS") or os.getenv("LLM_TIMEOUT_SECONDS")
        else:
            value = os.getenv("GEMINI_TIMEOUT_SECONDS") or os.getenv("LLM_TIMEOUT_SECONDS")
        return float(value or "30")

    def _init_groq(self) -> None:
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing.")

        self.session = requests.Session()
        self.session.trust_env = False
        logger.info("LLM provider initialized: groq (%s)", self.groq_model)

    def _init_gemini(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing.")
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - only triggered in broken envs
            raise RuntimeError("google-generativeai is not installed.") from exc

        if genai is None:  # pragma: no cover - defensive
            raise RuntimeError("google-generativeai is not installed.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            self.gemini_model,
            generation_config={
                "temperature": 0,
                "top_k": 1,
            },
        )
        logger.info("LLM provider initialized: gemini (%s)", self.gemini_model)

    def _chat_completion(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        if self.provider == "groq":
            return self._groq_chat_completion(system_prompt, user_prompt, max_tokens)
        if self.provider == "gemini":
            return self._gemini_chat_completion(system_prompt, user_prompt)
        raise RuntimeError("No active LLM provider is configured.")

    @staticmethod
    def _format_llm_warning(exc: Exception) -> str:
        text = str(exc).lower()
        if (
            "quota" in text
            or "rate limit" in text
            or "429" in text
            or "resourceexhausted" in text
            or "too many requests" in text
        ):
            return "LLM quota or rate limit reached."
        if "timeout" in text:
            return "LLM request timed out."
        if "unauthorized" in text or "401" in text or "403" in text:
            return "LLM authentication or provider access failed."
        return "LLM request failed."

    def _groq_chat_completion(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        assert self.session is not None
        assert self.groq_api_key is not None

        response = self.session.post(
            f"{self.groq_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.groq_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
            },
            timeout=self.request_timeout,
        )
        response.raise_for_status()

        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("Groq returned no choices.")

        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            return "\n".join(parts).strip()
        return str(content).strip()

    def _gemini_chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        assert self.model is not None
        prompt = (
            f"System instructions:\n{system_prompt}\n\n"
            f"User request:\n{user_prompt}"
        )
        response = self.model.generate_content(
            prompt,
            request_options={"timeout": self.request_timeout},
        )
        return response.text.strip()

    def summarize_result(self, text: str) -> dict[str, object]:
        system_prompt = (
            "You are an AI assistant specializing in summarizing job interview "
            "responses. Produce a concise, professional summary in one natural "
            "paragraph. Keep the key ideas and remove filler."
        )
        user_prompt = (
            "Candidate response:\n"
            f"\"\"\"{text}\"\"\"\n\n"
            "Return only the summary paragraph."
        )

        try:
            return {
                "value": self._chat_completion(system_prompt, user_prompt, max_tokens=180),
                "degraded": False,
                "warning": None,
            }
        except Exception as exc:
            logger.error(
                "%s summarize failed: %s",
                (self.provider or "llm").upper(),
                exc,
            )
            return {
                "value": SUMMARY_FALLBACK_TEXT,
                "degraded": True,
                "warning": self._format_llm_warning(exc),
            }

    def summarize(self, text: str) -> str:
        return str(self.summarize_result(text)["value"])

    def english_score(self, text: str) -> float:
        system_prompt = (
            "You assess spoken English in job-interview transcripts. Score the "
            "candidate's English from 0 to 10 based on grammar, clarity, "
            "vocabulary, coherence, and fluency that can be inferred from the "
            "transcript alone. Return only one number. Use decimals when useful."
        )
        user_prompt = (
            "Interview transcript:\n"
            f"\"\"\"{text}\"\"\"\n\n"
            "Return only one number from 0 to 10."
        )

        try:
            raw = self._chat_completion(system_prompt, user_prompt, max_tokens=20)
            match = re.search(r"\d+(?:\.\d+)?", raw)
            if match:
                score = float(match.group())
                return max(0.0, min(10.0, score))
            logger.warning(
                "%s english_score returned no numeric value in: '%s'. Using fallback 5.",
                (self.provider or "llm").upper(),
                raw,
            )
            return ENGLISH_FALLBACK_SCORE
        except Exception as exc:
            logger.error(
                "%s english_score failed: %s. Using fallback 5.",
                (self.provider or "llm").upper(),
                exc,
            )
            return ENGLISH_FALLBACK_SCORE

    def relevance_check_result(self, text: str, question: str) -> dict[str, object]:
        system_prompt = (
            "You are an AI interviewer assistant. Score how relevant the "
            "candidate's answer is to the interview question on a scale from 0 "
            "to 10. Return only a single integer."
        )
        user_prompt = (
            f"Interview question:\n\"{question}\"\n\n"
            f"Candidate answer:\n\"{text}\"\n\n"
            "Return only one integer from 0 to 10."
        )

        try:
            raw = self._chat_completion(system_prompt, user_prompt, max_tokens=20)
            match = re.search(r"\d+", raw)
            if match:
                score = int(match.group())
                return {
                    "value": max(0, min(10, score)),
                    "degraded": False,
                    "warning": None,
                }
            logger.warning(
                "%s relevance_check returned no integer in: '%s'. Using fallback 5.",
                (self.provider or "llm").upper(),
                raw,
            )
            return {
                "value": RELEVANCE_FALLBACK_SCORE,
                "degraded": True,
                "warning": "LLM returned an invalid relevance score.",
            }
        except Exception as exc:
            logger.error(
                "%s relevance_check failed: %s. Using fallback 5.",
                (self.provider or "llm").upper(),
                exc,
            )
            return {
                "value": RELEVANCE_FALLBACK_SCORE,
                "degraded": True,
                "warning": self._format_llm_warning(exc),
            }

    def relevance_check(self, text: str, question: str) -> int:
        return int(self.relevance_check_result(text, question)["value"])
