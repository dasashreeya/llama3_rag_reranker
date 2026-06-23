"""Answer generator LLM.

Backends, selected by ``generator.backend`` (default ``auto``):
  - openai:  gpt-4o (faithful to the paper) when OPENAI_API_KEY is set
  - ollama:  a local Ollama model, clearly flagged as an approximate substitute
  - stub:    deterministic extractive answer (no LLM) so the pipeline and smoke
             test run end to end with no API key and no Ollama; flagged approximate

``auto`` prefers openai, then ollama, then stub.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass

_PROMPT = (
    "Answer the question using only the context below. "
    "Be concise. If the context is insufficient, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
)


@dataclass
class Generator:
    backend: str
    approximate: bool

    def generate(self, query: str, contexts: list[str]) -> str:  # pragma: no cover
        raise NotImplementedError


class StubGenerator(Generator):
    """Extractive, no-LLM fallback. NOT faithful — configure a real backend for results."""

    def __init__(self):
        super().__init__(backend="stub", approximate=True)

    def generate(self, query: str, contexts: list[str]) -> str:
        top = contexts[0] if contexts else ""
        return f"[stub answer — no LLM configured] Based on the retrieved context: {top}".strip()


class OpenAIGenerator(Generator):
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 400):
        super().__init__(backend="openai", approximate=False)
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def generate(self, query: str, contexts: list[str]) -> str:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        prompt = _PROMPT.format(context="\n\n".join(contexts), query=query)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()


class OllamaGenerator(Generator):
    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        # Approximate: a local Ollama model is not the paper's gpt-4o generator.
        super().__init__(backend="ollama", approximate=True)
        self.model = model
        self.host = host

    def generate(self, query: str, contexts: list[str]) -> str:
        import json

        prompt = _PROMPT.format(context="\n\n".join(contexts), query=query)
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["response"].strip()


def _ollama_available(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def get_generator(config) -> Generator:
    backend = config.get("generator.backend", "auto")
    ollama_host = config.get("generator.ollama_host", "http://localhost:11434")
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if backend == "auto":
        if has_openai:
            backend = "openai"
        elif _ollama_available(ollama_host):
            backend = "ollama"
        else:
            backend = "stub"

    if backend == "openai":
        if not has_openai:
            raise RuntimeError("generator.backend=openai but OPENAI_API_KEY is not set")
        return OpenAIGenerator(
            config.get("generator.openai_model", "gpt-4o"),
            config.get("generator.max_tokens", 400),
        )
    if backend == "ollama":
        return OllamaGenerator(config.get("generator.ollama_model", "llama3.2"), ollama_host)
    return StubGenerator()
