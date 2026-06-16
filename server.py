"""RunPredict - deployable server (frontend + AI relay in one process)

Serves index.html (the app) and proxies AI calls to Google Gemini's free
tier, keeping the Gemini key server-side only (set via env var on the host).
"""
import json
import os
import sys
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
DEFAULT_MODEL = "gemini-2.5-flash"

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
PORT = int(os.environ.get("PORT", 8787))

if not API_KEY:
    print("[ERROR] GEMINI_API_KEY environment variable is not set")


def call_gemini(text, max_tokens):
    url = GEMINI_ENDPOINT.format(model=MODEL, key=API_KEY)
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens or 1000,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates: " + json.dumps(body)[:300])
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/messages":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req_body = json.loads(raw.decode("utf-8"))
            messages = req_body.get("messages", [])
            user_text = ""
            for m in messages:
                if m.get("role") == "user":
                    c = m.get("content")
                    user_text += c if isinstance(c, str) else json.dumps(c)
            max_tokens = req_body.get("max_tokens", 1000)
            text_out = call_gemini(user_text, max_tokens)
            resp_body = json.dumps({"content": [{"type": "text", "text": text_out}]}).encode("utf-8")
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            err_detail = e.read().decode("utf-8", "ignore")[:500]
            print(f"[RELAY] Gemini HTTPError {e.code}: {err_detail}")
            self.send_response(502)
            self._cors()
            self.end_headers()
        except Exception as e:
            print(f"[RELAY] Error: {e}")
            self.send_response(500)
            self._cors()
            self.end_headers()

    def log_message(self, fmt, *args):
        print("[SERVER] " + (fmt % args))


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[SERVER] RunPredict running on port {PORT} (model={MODEL})")
    server.serve_forever()


if __name__ == "__main__":
    main()
