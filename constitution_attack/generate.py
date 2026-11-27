"""Step 1: k replies per prompt. Single user turn, no system prompt, the model's
own chat template.

Two backends:
  vllm    load the model in this process with vLLM (one GPU job, no server)
  openai  any OpenAI-compatible chat endpoint (``vllm serve``, a hosted API, ...)

Output: ``generations.jsonl``, one row per prompt:
{"item_id", "responses": [k strings], "finish_reasons": [k strings]}
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Sequence

_BANNED = re.compile(r"^<charter_\d+\.\d+>$")


def banned_token_bias(tok) -> dict[int, float] | None:
    """Section-marker tokens (``<charter_X.Y>``) that some of our fine-tuned
    checkpoints carry in the vocabulary must never be generated. Models without
    such tokens get no bias."""
    ids = [i for t, i in tok.get_vocab().items() if _BANNED.match(t)]
    return {i: -100.0 for i in ids} or None


def _end_of_turn_ids(model: str, tok) -> list[int]:
    """The tokenizer's eos id plus the eos ids of the model's generation_config."""
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    try:
        from vllm.transformers_utils.config import try_get_generation_config
        gc = try_get_generation_config(str(model), trust_remote_code=False)
    except Exception:
        gc = None
    eos = getattr(gc, "eos_token_id", None) if gc is not None else None
    if eos is not None:
        ids |= {int(eos)} if isinstance(eos, int) else {int(i) for i in eos}
    return sorted(ids)


def generate_vllm(items: Sequence[dict], *, model: str, k: int, temperature: float, top_p: float,
                  max_tokens: int, seed: int | None, max_model_len: int, dtype: str,
                  tensor_parallel_size: int) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype=dtype, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len,
              gpu_memory_utilization=0.90, seed=seed)
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                     add_generation_prompt=True) for it in items]
    # The rendered text already carries the template's special tokens.
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    print(f"example rendered model input:\n{texts[0]!r}")
    room = max_model_len - max_tokens
    too_long = [it["item_id"] for it, x in zip(items, ids) if len(x) > room]
    if too_long:
        raise ValueError(f"{len(too_long)} prompts exceed max_model_len - max_tokens = {room}: {too_long[:5]}")
    # End-of-turn ids are passed as explicit stop ids. With n > 1, vLLM 0.9's
    # offline engine applied the generation_config eos ids (e.g. <|im_end|>) only
    # to the parent request, so every sampled reply ran to max_tokens.
    stop_ids = _end_of_turn_ids(model, tok)
    sp = SamplingParams(n=k, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed,
                        logit_bias=banned_token_bias(tok), stop_token_ids=stop_ids)
    if hasattr(sp, "_all_stop_token_ids"):
        sp._all_stop_token_ids.update(stop_ids)
    print(f"stop token ids {stop_ids} ({[tok.convert_ids_to_tokens(i) for i in stop_ids]})")
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp)
    rows = []
    for it, o in zip(items, outs):
        if len(o.outputs) != k:
            raise RuntimeError(f"{it['item_id']}: {len(o.outputs)} samples, expected {k}")
        rows.append({"item_id": it["item_id"], "responses": [c.text for c in o.outputs],
                     "finish_reasons": [c.finish_reason for c in o.outputs]})
    return rows


def generate_openai(items: Sequence[dict], *, model: str, base_url: str, api_key: str | None, k: int,
                    temperature: float, top_p: float, max_tokens: int, seed: int | None,
                    concurrency: int, max_retries: int = 5) -> list[dict]:
    """k separate requests per prompt (n=1 each, seed + sample index), so the
    result does not depend on how a server handles n > 1."""
    from openai import AsyncOpenAI

    async def run() -> list[dict]:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
                             max_retries=max_retries)
        sem = asyncio.Semaphore(concurrency)

        async def one(it: dict, i: int) -> tuple[str, str | None]:
            kw = {} if seed is None else {"seed": seed + i}
            async with sem:
                c = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": it["prompt"]}],
                    temperature=temperature, top_p=top_p, max_tokens=max_tokens, **kw)
            ch = c.choices[0]
            return ch.message.content or "", ch.finish_reason

        async def row(it: dict) -> dict:
            outs = await asyncio.gather(*[one(it, i) for i in range(k)])
            return {"item_id": it["item_id"], "responses": [o[0] for o in outs],
                    "finish_reasons": [o[1] for o in outs]}

        try:
            return list(await asyncio.gather(*[row(it) for it in items]))
        finally:
            await client.close()

    return asyncio.run(run())
