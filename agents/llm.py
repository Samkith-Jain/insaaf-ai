"""
Optional LLM backend for agents.

The agents run fully offline by default (rule-based). When a backend is
supplied, it is used to *re-classify* items the rules already located, and
any failure (no key, no network, bad JSON) silently falls back to the rule
result. The LLM therefore can never remove grounding: it only chooses labels
for sentences that already exist in the source text.

Only stdlib is used (urllib), so no extra dependency is needed.
"""
import json
import os
import urllib.request
from typing import Protocol


class LLMBackend(Protocol):
    def complete_json(self, system: str, user: str) -> object:
        """Return parsed JSON (list/dict) or raise on any failure."""


class AnthropicBackend:
    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: int = 60):
        self.model = model or os.environ.get("INSAAF_LLM_MODEL", "claude-sonnet-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

    def complete_json(self, system: str, user: str) -> object:
        body = json.dumps({
            "model": self.model,
            "max_tokens": 4000,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("\n") + 1:] if text.lower().startswith("json") else text
        return json.loads(text)
