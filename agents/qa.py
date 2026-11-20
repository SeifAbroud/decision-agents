"""Agent « Ask the data » : Claude choisit lui-même les outils (tool use), chaque
appel devient une preuve, puis la réponse est vérifiée chiffre par chiffre."""
from __future__ import annotations

import json
import re

from . import llm, rag
from .tools import TOOL_SCHEMAS, EvidenceLog
from .verify import check_numbers

SYSTEM = ("Tu es l'assistant analytique de la plateforme d'aide à la décision Favorita (supermarchés, Équateur). "
          "Pour répondre, appelle les outils : ils sont la seule source de chiffres autorisée. Ne calcule pas toi-même "
          "de nouveaux chiffres. Réponds en français, de façon concise (5 à 10 lignes), écris les nombres sans séparateur "
          "de milliers avec un point décimal, et cite les preuves [E1], [E2] renvoyées après chaque appel. "
          "Si la question sort du périmètre des données, dis-le.")


def answer(question: str, history: list[dict] | None = None, max_steps: int = 6) -> dict:
    ev = EvidenceLog()
    usage = llm.Usage()
    kb = rag.context(question, 2)
    msgs = list(history or []) + [{"role": "user", "content": f"{question}\n\n(Contexte métier utile :\n{kb})"}]
    for _ in range(max_steps):
        resp = llm.client().messages.create(model=llm.model_name(), max_tokens=1200, system=SYSTEM,
                                            tools=TOOL_SCHEMAS, messages=msgs)
        usage.add("qa", resp.usage)
        msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in resp.content]})
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            break
        results = []
        for b in calls:
            item = ev.call(b.name, **(b.input or {}))
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": f"[{item['id']}] " + json.dumps(item["result"], ensure_ascii=False)})
        msgs.append({"role": "user", "content": results})
    text = "".join(b.text for b in resp.content if b.type == "text")
    cited = sorted(set(re.findall(r"\bE\d+\b", text)), key=lambda x: int(x[1:]))
    check = check_numbers(text, cited or [e["id"] for e in ev.items], ev)
    return {"answer": text, "evidence": ev.items, "check": check, "cost_usd": usage.cost_usd(),
            "tokens": usage.input_tokens + usage.output_tokens}
