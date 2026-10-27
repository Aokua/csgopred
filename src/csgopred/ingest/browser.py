"""Minimal CDP browser wrapper (headless Edge/Chrome).

Purpose: handle sites that require JavaScript execution (5EPlay's Aliyun WAF challenge, HLTV's Cloudflare).
Depends only on websocket-client + the standard library; no playwright.

Usage:
    with Browser() as b:
        html = b.html("https://www.5eplay.com/")
        cookies = b.cookies()
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import websocket

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def find_browser() -> str:
    for p in EDGE_CANDIDATES:
        if Path(p).exists():
            return p
    raise RuntimeError("找不到 Edge/Chrome 可执行文件")


def _free_port(base: int) -> int:
    import socket
    for p in range(base, base + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("找不到空闲端口")


class Browser:
    def __init__(self, port: int = 9333, headless: bool = True):
        self.port = _free_port(port)
        self.headless = headless
        self.proc: subprocess.Popen | None = None
        self.profile = tempfile.mkdtemp(prefix="csgopred-cdp-")
        self.ws: websocket.WebSocket | None = None
        self._id = 0
        self._events: list[dict] = []

    # ---------------- Lifecycle ----------------
    def __enter__(self) -> "Browser":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        args = [
            find_browser(),
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={self.profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-gpu",
            "--disable-extensions",
            "--disable-popup-blocking",
        ]
        if self.headless:
            args.append("--headless=new")
        args.append("about:blank")
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/version", timeout=1).read()
                return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("CDP 端口未就绪")

    def close(self) -> None:
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    # ---------------- Pages ----------------
    def new_page(self, url: str) -> None:
        target = self._http("/json/new?" + url, method="PUT")
        ws_url = target["webSocketDebuggerUrl"]
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self.send("Page.enable")
        self.send("Runtime.enable")
        self.send("Network.enable")

    def _http(self, path: str, method: str = "GET") -> Any:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method=method)
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))

    def send(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        assert self.ws is not None
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.ws.settimeout(max(0.1, deadline - time.time()))
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"CDP {method}: {msg['error']}")
                return msg.get("result", {})
            if "method" in msg:
                self._events.append(msg)
        raise TimeoutError(f"CDP {method} 超时")

    def eval(self, expression: str, timeout: float = 30) -> Any:
        res = self.send(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
            timeout=timeout,
        )
        return res.get("result", {}).get("value")

    def goto(self, url: str, wait: float = 2.0, ready_check: str | None = None,
             max_wait: float = 25.0) -> str:
        self.send("Page.navigate", {"url": url})
        deadline = time.time() + max_wait
        time.sleep(wait)
        while time.time() < deadline:
            html = self.eval("document.documentElement.outerHTML") or ""
            if ready_check is None or ready_check in html:
                return html
            time.sleep(1.0)
        return self.eval("document.documentElement.outerHTML") or ""

    def requests_log(self) -> list[str]:
        """Network request URLs captured in this session (deduplicated)."""
        out, seen = [], set()
        for ev in self._events:
            if ev.get("method") == "Network.requestWillBeSent":
                u = ev["params"]["request"]["url"]
                if u not in seen:
                    seen.add(u)
                    out.append(u)
        return out

    def scroll(self, times: int = 5, pause: float = 1.5) -> None:
        import time as _t
        for _ in range(times):
            self.eval("window.scrollTo(0, document.body.scrollHeight)")
            _t.sleep(pause)

    def screenshot(self, path: str, full_page: bool = True) -> str:
        """Screenshot of the current page (PNG)."""
        import base64

        params: dict = {"format": "png"}
        if full_page:
            metrics = self.send("Page.getLayoutMetrics")
            size = metrics.get("cssContentSize") or metrics.get("contentSize") or {}
            w = int(size.get("width", 1280))
            h = int(size.get("height", 800))
            params.update({"captureBeyondViewport": True,
                           "clip": {"x": 0, "y": 0, "width": w, "height": h, "scale": 1}})
        res = self.send("Page.captureScreenshot", params, timeout=30)
        with open(path, "wb") as f:
            f.write(base64.b64decode(res["data"]))
        return path

    def cookies(self) -> list[dict]:
        res = self.send("Network.getAllCookies")
        return res.get("cookies", [])
