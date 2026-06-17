"""AI relay - Vercel Python serverless function

Receives requests in Anthropic Messages API shape (as sent by the frontend),
translates to Gemini, and returns a response in the same Anthropic shape.
"""
import json
import os
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)


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
    with urllib.request.urlopen(req, timeout=8) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError("No candidates: " + json.dumps(body)[:200])
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
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
            resp_body = json.dumps(
                {"content": [{"type": "text", "text": text_out}]}
            ).encode("utf-8")
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            print(f"Gemini HTTPError {e.code}: {e.read().decode('utf-8','ignore')[:200]}")
            self.send_response(502)
            self._cors()
            self.end_headers()
        except Exception as e:
            print(f"Error: {e}")
            self.send_response(500)
            self._cors()
            self.end_headers()

    def log_message(self, fmt, *args):
        print(fmt % args)
