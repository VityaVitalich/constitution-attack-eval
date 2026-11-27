"""Step 2: one compliance label per reply.

Default judge: DeepSeek V4.1 Flash on OpenRouter (``deepseek/deepseek-v4.1-flash-1017``),
providers pinned to Morph then Pearl Research with no fallbacks, reasoning off,
temperature 0, 400 output tokens. The prompt was validated with this model only;
another judge model gives scores that are not comparable.

Labels are appended to ``judgements.jsonl`` as they arrive, so an interrupted
run resumes and only missing labels are asked again.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from .scoring import judge_prompt, parse_label

DEFAULT_JUDGE_MODEL = "deepseek/deepseek-v4.1-flash-1017"
DEFAULT_JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PROVIDER_ORDER = ("Morph", "Pearl Research")
JUDGE_MAX_TOKENS = 400


def extra_body(model: str, base_url: str, attempt: int = 0) -> dict:
    """Provider pinning for the default judge on OpenRouter; the order rotates
    per retry so a failing provider is demoted. Nothing for other judges."""
    if model != DEFAULT_JUDGE_MODEL or "openrouter.ai" not in base_url:
        return {}
    order = list(DEFAULT_PROVIDER_ORDER)
    shift = attempt % len(order)
    return {"provider": {"order": order[shift:] + order[:shift], "allow_fallbacks": False},
            "reasoning": {"enabled": False}}


def load_judgements(path: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    if path.exists():
        for l in open(path, encoding="utf-8"):
            if l.strip():
                o = json.loads(l)
                out[(o["item_id"], int(o["sample_idx"]))] = o
    return out


def judge_all(items: Mapping[str, dict], tenets: Mapping[str, dict], gens: Sequence[dict], out_path: Path, *,
              model: str = DEFAULT_JUDGE_MODEL, base_url: str = DEFAULT_JUDGE_BASE_URL, api_key: str | None = None,
              concurrency: int = 64, max_retries: int = 8) -> dict[tuple[str, int], dict]:
    """Judge every reply that has no label in ``out_path`` yet. Returns all labels.
    A reply whose label cannot be obtained after the retries stays unjudged."""
    from openai import AsyncOpenAI

    done = load_judgements(out_path)
    todo = [(g["item_id"], i, r) for g in gens for i, r in enumerate(g["responses"]) if (g["item_id"], i) not in done]
    if not todo:
        return done
    key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("no judge API key: set OPENROUTER_API_KEY (or pass --judge-api-key)")

    async def run() -> None:
        client = AsyncOpenAI(base_url=base_url, api_key=key, max_retries=0)
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        failed = 0

        async def one(item_id: str, idx: int, reply: str, f) -> None:
            nonlocal failed
            prompt = judge_prompt(items[item_id], tenets, reply)
            last = ""
            async with sem:
                for attempt in range(max_retries + 1):
                    try:
                        c = await client.chat.completions.create(
                            model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
                            max_tokens=JUDGE_MAX_TOKENS, extra_body=extra_body(model, base_url, attempt))
                        last = c.choices[0].message.content or ""
                        parsed = parse_label(last)
                    except Exception as e:  # network or provider error: retry
                        last, parsed = f"{type(e).__name__}: {e}", None
                    if parsed is not None:
                        rec = {"item_id": item_id, "sample_idx": idx, "label": parsed["label"],
                               "reason": parsed["reason"], "raw": last}
                        async with lock:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            f.flush()
                        done[(item_id, idx)] = rec
                        return
                    if attempt < max_retries:
                        await asyncio.sleep(min(30.0, 0.5 * 2 ** attempt))
            failed += 1
            print(f"unjudged after {max_retries + 1} attempts: {item_id} sample {idx}: {last[:160]!r}")

        try:
            with open(out_path, "a", encoding="utf-8") as f:
                await asyncio.gather(*[one(a, b, c, f) for a, b, c in todo])
        finally:
            await client.close()
        print(f"judged {len(todo) - failed} replies, {failed} left unjudged")

    asyncio.run(run())
    return done
