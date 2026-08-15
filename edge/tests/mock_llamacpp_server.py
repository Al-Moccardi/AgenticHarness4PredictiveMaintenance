#!/usr/bin/env python3
"""Mock of llama.cpp's llama-server for OFFLINE pipeline testing only.
Serves POST /v1/chat/completions and GET /health with the same response shape
(choices/usage/timings) as the real server, generating a plausible sectioned
interpretation that names two distinct sensors from the prompt. Never use its
output for the paper — it exists so `--backend llamacpp` can be integration-
tested without a GPU or model file.

    python3 tests/mock_llamacpp_server.py --port 8089
"""
import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        user = next((m["content"] for m in body.get("messages", [])
                     if m.get("role") == "user"), "")
        feats = []
        for f in re.findall(r"([A-Za-z]\w{0,7})\s*(?:<=|>)", user):
            if f not in feats:
                feats.append(f)
        a = feats[0] if feats else "T50"
        b = feats[1] if len(feats) > 1 else "Nc"
        content = (
            f"**Anomaly Interpretation:** The isolation rule combines {a} "
            f"and {b} outside the regime envelope, indicating an efficiency "
            f"deviation.\n**Cause:** Likely degradation in the subsystem "
            f"monitored by {a}, consistent with the {b} condition.\n"
            f"**Impact:** Reduced performance and higher fuel consumption "
            f"if unaddressed.\n**Anomalous Trend:** The unit's counters and "
            f"prior events suggest a developing pattern.\n"
            f"**Expected Future Failures:** Moderate risk over the coming "
            f"cycles. (gravity score: 3)\n**Recommendation:** Inspect the "
            f"{a}-related subsystem at the next window.")
        p_n = max(len(user) // 4, 1)
        g_n = max(len(content) // 4, 1)
        p_ms, g_ms = p_n / 900.0 * 1000, g_n / 20.0 * 1000
        time.sleep(0.05)
        self._send({
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": content}}],
            "usage": {"prompt_tokens": p_n, "completion_tokens": g_n,
                      "total_tokens": p_n + g_n},
            "timings": {"prompt_n": p_n, "prompt_ms": p_ms,
                        "prompt_per_second": 900.0,
                        "predicted_n": g_n, "predicted_ms": g_ms,
                        "predicted_per_second": 20.0},
            "model": body.get("model", "mock")})

    def _send(self, obj):
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8089)
    a = ap.parse_args()
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()
