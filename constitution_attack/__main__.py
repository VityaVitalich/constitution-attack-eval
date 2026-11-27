"""Command line: ``python -m constitution_attack {run,generate,judge,score} ...``

A run lives in one directory:
  generations.jsonl   the model's replies            (step generate)
  judgements.jsonl    one compliance label per reply (step judge)
  result.json         per-reply rows and all scores  (step score)
  scores.md           the score tables
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from . import __version__
from .scoring import (
    BOOT_DRAWS, load_items, load_tenets, prompt_version, score, select_items, summary_table,
)


def _items(a) -> list[dict]:
    return select_items(load_items(), a.limit_scenarios)


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def cmd_generate(a) -> None:
    from . import generate as G
    out = Path(a.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_path = out / "generations.jsonl"
    if gen_path.exists() and not a.overwrite:
        print(f"{gen_path} exists, keeping it (pass --overwrite to generate again)")
        return
    items = _items(a)
    common = dict(model=a.model, k=a.k, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed)
    if a.backend == "vllm":
        rows = G.generate_vllm(items, max_model_len=a.max_model_len, dtype=a.dtype,
                               tensor_parallel_size=a.tensor_parallel_size, **common)
    else:
        if not a.base_url:
            raise SystemExit("--backend openai needs --base-url")
        rows = G.generate_openai(items, base_url=a.base_url, api_key=a.api_key, concurrency=a.concurrency, **common)
    with open(gen_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"model": a.model, "backend": a.backend, "base_url": a.base_url, "k": a.k, "temperature": a.temperature,
            "top_p": a.top_p, "max_tokens": a.max_tokens, "seed": a.seed, "system_prompt": None,
            "limit_scenarios": a.limit_scenarios, "n_prompts": len(rows), "eval_version": __version__,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    (out / "generation_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {gen_path} ({len(rows)} prompts x {a.k} replies)")


def _load_run(a) -> tuple[list[dict], list[dict], dict]:
    out = Path(a.run_dir)
    gens = _read_jsonl(out / "generations.jsonl")
    by_id = {it["item_id"]: it for it in load_items()}
    missing = [g["item_id"] for g in gens if g["item_id"] not in by_id]
    if missing:
        raise SystemExit(f"generations.jsonl has unknown item ids, e.g. {missing[:3]}")
    meta_path = out / "generation_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return gens, [by_id[g["item_id"]] for g in gens], meta


def cmd_judge(a) -> None:
    from . import judge as J
    gens, items, _ = _load_run(a)
    J.judge_all({it["item_id"]: it for it in items}, load_tenets(), gens, Path(a.run_dir) / "judgements.jsonl",
                model=a.judge_model, base_url=a.judge_base_url, api_key=a.judge_api_key,
                concurrency=a.judge_concurrency, max_retries=a.judge_max_retries)


def cmd_score(a) -> None:
    from . import judge as J
    out = Path(a.run_dir)
    gens, items, meta = _load_run(a)
    labels = J.load_judgements(out / "judgements.jsonl")
    ks = {len(g["responses"]) for g in gens}
    if len(ks) != 1:
        raise SystemExit(f"prompts have different numbers of replies: {sorted(ks)}")
    k = ks.pop()
    rows = []
    for g, it in zip(gens, items):
        fins = g.get("finish_reasons") or [None] * k
        samples = []
        for i, (r, fin) in enumerate(zip(g["responses"], fins)):
            lab = labels.get((it["item_id"], i))
            samples.append({"sample_idx": i, "response": r, "finish_reason": fin,
                            "complied": None if lab is None else lab["label"] == "comply",
                            "reason": None if lab is None else lab["reason"]})
        rows.append({**{f: it[f] for f in ("item_id", "scenario_id", "tenet_id", "article", "version", "domain_name",
                                           "tenet_name", "scenario_title", "base_request", "prompt")},
                     "samples": samples})
    metrics = score(rows, k, a.bootstrap_draws)
    judge_meta = {"model": getattr(a, "judge_model", None), "base_url": getattr(a, "judge_base_url", None),
                  "prompt_version": prompt_version(), "temperature": 0, "max_tokens": J.JUDGE_MAX_TOKENS}
    if judge_meta["model"] == J.DEFAULT_JUDGE_MODEL:
        judge_meta["provider_order"] = list(J.DEFAULT_PROVIDER_ORDER)
    result = {"metadata": {"benchmark": "constitution_attack", "eval_version": __version__, "generation": meta,
                           "judge": judge_meta}, "metrics": metrics, "results": rows}
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False))
    table = summary_table(metrics)
    (out / "scores.md").write_text(f"# {meta.get('model', out.name)}\n\n{table}\n")
    print(table)
    print(f"\nwrote {out / 'result.json'} and {out / 'scores.md'}")
    if a.max_unjudged_share is not None and metrics["n_replies_unjudged"] / max(1, metrics["n_replies"]) > a.max_unjudged_share:
        raise SystemExit(f"{metrics['n_replies_unjudged']} of {metrics['n_replies']} replies are unjudged; "
                         "run the judge step again (it resumes)")


def cmd_run(a) -> None:
    cmd_generate(a)
    cmd_judge(a)
    cmd_score(a)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m constitution_attack", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def gen_args(p):
        p.add_argument("--model", required=True, help="HF repo id or local path (vllm), or served model name (openai)")
        p.add_argument("--backend", choices=("vllm", "openai"), default="vllm")
        p.add_argument("--base-url", help="chat endpoint for --backend openai, e.g. http://localhost:8000/v1")
        p.add_argument("--api-key", help="key for that endpoint (default: $OPENAI_API_KEY or none)")
        p.add_argument("--k", type=int, default=5, help="replies per prompt")
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--top-p", type=float, default=1.0)
        p.add_argument("--max-tokens", type=int, default=400, help="the judge prompt states that replies are cut at 400")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-model-len", type=int, default=2048)
        p.add_argument("--dtype", default="bfloat16")
        p.add_argument("--tensor-parallel-size", type=int, default=1)
        p.add_argument("--concurrency", type=int, default=32, help="parallel requests, openai backend")
        p.add_argument("--limit-scenarios", type=int, default=None, help="smoke run on this many scenarios")
        p.add_argument("--overwrite", action="store_true", help="generate again even if generations.jsonl exists")

    def judge_args(p):
        from .judge import DEFAULT_JUDGE_BASE_URL, DEFAULT_JUDGE_MODEL
        p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
        p.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
        p.add_argument("--judge-api-key", help="default: $OPENROUTER_API_KEY, then $OPENAI_API_KEY")
        p.add_argument("--judge-concurrency", type=int, default=64)
        p.add_argument("--judge-max-retries", type=int, default=8)

    def score_args(p):
        p.add_argument("--bootstrap-draws", type=int, default=BOOT_DRAWS)
        p.add_argument("--max-unjudged-share", type=float, default=0.01,
                       help="exit with an error above this share of unjudged replies")

    for name, fn, parts in (("generate", cmd_generate, (gen_args,)), ("judge", cmd_judge, (judge_args,)),
                            ("score", cmd_score, (judge_args, score_args)),
                            ("run", cmd_run, (gen_args, judge_args, score_args))):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True, help="directory of this run")
        for part in parts:
            part(p)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
