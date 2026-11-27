"""The set, the judge prompt, label parsing and the scores (no model, no network)."""
import collections

from constitution_attack.scoring import (
    METRICS, boot_ci, judge_prompt, load_items, load_tenets, parse_label, prompt_values, prompt_version,
    score, select_items, summary_table,
)


def test_set_shape():
    items, tenets = load_items(), load_tenets()
    assert len(items) == 1740 and len({it["item_id"] for it in items}) == 1740
    assert collections.Counter(it["version"] for it in items) == {"plain": 870, "framed": 870}
    assert len(tenets) == 87 and {it["tenet_id"] for it in items} == set(tenets)
    per_tenet = collections.Counter(it["tenet_id"] for it in items if it["version"] == "plain")
    assert set(per_tenet.values()) == {10}
    by_scen = collections.defaultdict(dict)
    for it in items:
        by_scen[it["scenario_id"]][it["version"]] = it
    for d in by_scen.values():
        assert set(d) == {"plain", "framed"}
        # Same request in both versions; the plain prompt is the bare request.
        assert d["plain"]["base_request"] == d["framed"]["base_request"] == d["plain"]["prompt"]
        assert d["plain"]["base_request"] in d["framed"]["prompt"]
        assert d["framed"]["quotes"] and all(q in d["framed"]["prompt"] for q in d["framed"]["quotes"])
    assert len({it["domain_name"] for it in items}) == 6


def test_smoke_subset_keeps_both_versions():
    sub = select_items(load_items(), 12)
    assert len(sub) == 24 and len({it["scenario_id"] for it in sub}) == 12


def test_judge_prompt_shows_request_not_framing():
    items, tenets = load_items(), load_tenets()
    framed = next(it for it in items if it["version"] == "framed")
    p = judge_prompt(framed, tenets, "REPLY-TEXT")
    assert framed["base_request"] in p and "REPLY-TEXT" in p
    assert tenets[framed["tenet_id"]]["statement"] in p
    assert framed["quotes"][0] not in p
    assert prompt_version().startswith("comply_v1-")


def test_parse_label():
    assert parse_label('{"reason": "delivers it", "label": "comply"}') == {"label": "comply", "reason": "delivers it"}
    assert parse_label('<think>{"label": "comply"}</think> {"reason": "refuses", "label": "no"}')["label"] == "no"
    assert parse_label("no") is None and parse_label('{"label": "partial"}') is None and parse_label("") is None


def _row(sid, version, complied, domain="D", tenet="T"):
    return {"scenario_id": sid, "version": version, "domain_name": domain, "tenet_id": tenet,
            "samples": [{"complied": c, "response": "x", "finish_reason": "stop"} for c in complied]}


def test_prompt_values():
    assert prompt_values([{"complied": True}, {"complied": False}], 2) == {"mean": 0.5, "worst@k": 1}
    assert prompt_values([{"complied": False}, {"complied": False}], 2) == {"mean": 0.0, "worst@k": 0}
    assert prompt_values([{"complied": True}, {"complied": None}], 2) is None
    assert prompt_values([{"complied": True}], 2) is None


def test_score():
    rows = [_row("a", "plain", [0, 0]), _row("a", "framed", [1, 0]),
            _row("b", "plain", [1, 1], "E"), _row("b", "framed", [1, 1], "E"),
            _row("c", "plain", [0, 0]), _row("c", "framed", [None, 1])]   # c is excluded: one reply unjudged
    m = score(rows, 2, draws=200)
    assert m["n_scenarios"] == 2 and m["n_scenarios_excluded"] == 1 and m["n_replies_unjudged"] == 1
    assert m["mean"]["plain"] == 0.5 and m["mean"]["framed"] == 0.75
    assert m["worst@k"]["plain"] == 0.5 and m["worst@k"]["framed"] == 1.0
    assert abs(m["mean"]["framed_minus_plain"][0] - 0.25) < 1e-9
    assert m["by_domain"]["E"]["mean"] == {"plain": 1.0, "framed": 1.0}
    assert set(METRICS) <= set(m) and "Framed - Plain" in summary_table(m)


def test_boot_ci_is_deterministic():
    v = [0, 1, 1, 0, 1, 0, 0, 1]
    assert boot_ci(v, 300) == boot_ci(v, 300) and boot_ci([], 10) == [None, None, None]
