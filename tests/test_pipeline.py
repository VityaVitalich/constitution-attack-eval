"""generate (openai backend) -> judge -> score against a local fake chat server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from constitution_attack.__main__ import main
from constitution_attack.judge import DEFAULT_JUDGE_MODEL, extra_body

CALLS = {"target": 0, "judge": 0, "judge_fail_once": set()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][0]["content"]
        assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"   # no system prompt
        if body["model"] == "fake-judge":
            CALLS["judge"] += 1
            reply = user.split("REPLY:\n", 1)[1]
            if "FLAKY" in reply and reply not in CALLS["judge_fail_once"]:
                CALLS["judge_fail_once"].add(reply)
                text = "no"                       # unparseable: the client must retry
            else:
                text = json.dumps({"reason": "test", "label": "comply" if "SURE" in reply else "no"})
        else:
            CALLS["target"] += 1
            framed = "Value Constitution" in user or len(user) > 600
            text = ("SURE here it is" if framed and body.get("seed", 0) % 2 == 0 else "I can't help with that") + \
                   (" FLAKY" if body.get("seed") == 1 else "")
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
               "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_end_to_end(server, tmp_path):
    run = tmp_path / "run"
    args = ["--run-dir", str(run), "--model", "fake-model", "--backend", "openai", "--base-url", server,
            "--k", "3", "--limit-scenarios", "6", "--judge-model", "fake-judge", "--judge-base-url", server,
            "--judge-api-key", "x", "--bootstrap-draws", "100"]
    main(["run", *args])
    gens = [json.loads(l) for l in open(run / "generations.jsonl")]
    assert len(gens) == 12 and all(len(g["responses"]) == 3 for g in gens) and CALLS["target"] == 36
    res = json.loads((run / "result.json").read_text())
    m = res["metrics"]
    assert m["n_scenarios"] == 6 and m["n_replies"] == 36 and m["n_replies_unjudged"] == 0
    assert m["mean"]["plain"] == 0 and m["worst@k"]["plain"] == 0
    assert m["worst@k"]["framed"] == 1 and abs(m["mean"]["framed"] - 2 / 3) < 1e-9
    assert res["metadata"]["generation"]["system_prompt"] is None
    assert (run / "scores.md").exists()
    # Second run: generations kept, nothing judged again.
    n_t, n_j = CALLS["target"], CALLS["judge"]
    main(["run", *args])
    assert (CALLS["target"], CALLS["judge"]) == (n_t, n_j)


def test_default_judge_routing():
    b0 = extra_body(DEFAULT_JUDGE_MODEL, "https://openrouter.ai/api/v1", 0)
    b1 = extra_body(DEFAULT_JUDGE_MODEL, "https://openrouter.ai/api/v1", 1)
    assert b0["provider"] == {"order": ["Morph", "Pearl Research"], "allow_fallbacks": False}
    assert b1["provider"]["order"] == ["Pearl Research", "Morph"] and b0["reasoning"] == {"enabled": False}
    assert extra_body("other/model", "https://openrouter.ai/api/v1") == {}
    assert extra_body(DEFAULT_JUDGE_MODEL, "http://localhost:8000/v1") == {}
