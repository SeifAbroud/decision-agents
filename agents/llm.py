"""Accès à l'API Claude (Anthropic).

Modèle par défaut : Claude Haiku 5.5 (le moins cher), payé par les crédits API
mensuels du plan Claude Team. Si aucune clé n'est configurée, le système passe en
mode hors-ligne (règles déterministes) pour que le dashboard reste utilisable.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

DEFAULT_MODEL = "claude-haiku-5-5"


def model_name() -> str:
    return os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)


def has_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    log: list = field(default_factory=list)

    def add(self, agent: str, u):
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.calls += 1
        self.log.append({"agent": agent, "in": u.input_tokens, "out": u.output_tokens})

    def cost_usd(self) -> float:
        # Tarifs Haiku 5.5 (prompt < 100k tokens) : 0.10 $ / 0.50 $ par million de tokens
        return round(self.input_tokens * 0.10 / 1e6 + self.output_tokens * 0.50 / 1e6, 5)


_client = None


def client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()  # lit ANTHROPIC_API_KEY
    return _client


def ask(system: str, user: str, usage: Usage | None = None, agent: str = "", max_tokens: int = 2000) -> str:
    resp = client().messages.create(
        model=model_name(), max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    if usage is not None:
        usage.add(agent, resp.usage)
    return "".join(b.text for b in resp.content if b.type == "text")


def parse_json(text: str):
    """Extrait le premier bloc JSON (objet ou liste) d'une réponse du modèle."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    raw = m.group(1) if m else text
    start = min([i for i in (raw.find("["), raw.find("{")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("Pas de JSON dans la réponse")
    raw = raw[start:]
    for end in range(len(raw), 0, -1):
        if raw[end - 1] in "]}":
            try:
                return json.loads(raw[:end])
            except json.JSONDecodeError:
                continue
    raise ValueError("JSON invalide dans la réponse")
