"""Zero-dependency web service: static pages + JSON API.

Start with: python -m csgopred serve --port 8080
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..db import connect
from . import service

STATIC = Path(__file__).parent / "static"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # Keep quiet
        pass

    def _json(self, payload, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._file(STATIC / "index.html")
            if u.path == "/en":
                return self._file(STATIC / "index_en.html")
            if u.path.startswith("/static/"):
                return self._file(STATIC / u.path[len("/static/"):])
            if u.path == "/api/teams":
                conn = connect()
                return self._json(service.rankings(conn, int(qs.get("top", 30))))
            if u.path == "/api/predict":
                if not qs.get("a") or not qs.get("b"):
                    return self._json({"error": "需要 a 和 b 两个队名"}, 400)
                conn = connect()
                return self._json(
                    service.predict(
                        conn, qs["a"], qs["b"], qs.get("format", "bo3"),
                        qs.get("date") or None, qs.get("event", ""),
                        float(qs.get("margin", 0.06)),
                        float(qs["bankroll"]) if qs.get("bankroll") else None,
                        float(qs["odds_a"]) if qs.get("odds_a") else None,
                        float(qs["odds_b"]) if qs.get("odds_b") else None,
                    )
                )
            if u.path == "/api/stats":
                conn = connect()
                return self._json({**service.dataset_stats(conn), "model": service.model_info()})
            if u.path == "/api/news":
                conn = connect()
                return self._json(service.recent_news(conn, int(qs.get("limit", 20))))
            return self._json({"error": "not found"}, 404)
        except Exception as exc:  # Return the exception to the frontend to ease debugging
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def run(port: int = 8080, host: str = "127.0.0.1") -> None:
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"服务已启动: http://{host}:{port}")
    srv.serve_forever()
