"""Tiny local server so the dashboard can refresh itself.

Uses nothing but the standard library — no Flask, no Node, no hosting, no
account. It binds to 127.0.0.1 only, so it is reachable from your own browser
and from nowhere else.

    python3 -m nflproj.cli serve

Then open http://localhost:8765 and press Refresh. The button runs the same
three commands the cron job runs: refresh data, regrade completed games,
reproject the next slate (which re-fetches weather and injuries), then
rebuilds the page.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime

STATE = {"status": "idle", "log": [], "started": None, "finished": None}
LOCK = threading.Lock()


def _steps(slate: str | None, season: int | None, week: int | None) -> list[list[str]]:
    py = sys.executable or "python3"
    project = [py, "-m", "nflproj.cli", "project", "--dashboard", "dashboard.html"]
    if slate and slate != "next":
        project += ["--slate", slate]
        if season and week:
            project += ["--season", str(season), "--week", str(week)]
    else:
        project += ["--next"]
    return [
        [py, "-m", "nflproj.cli", "refresh"],
        [py, "-m", "nflproj.cli", "grade", "--output", "accuracy.json"],
        project,
    ]


def _run_job(slate=None, season=None, week=None) -> None:
    with LOCK:
        STATE.update(status="running", log=[], started=datetime.now().isoformat(),
                     finished=None)

    def log(line: str):
        with LOCK:
            STATE["log"].append(line)
            STATE["log"] = STATE["log"][-200:]

    try:
        for cmd in _steps(slate, season, week):
            log(f"$ {' '.join(cmd[2:])}")
            proc = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    log(line)
            if proc.wait() != 0:
                raise RuntimeError(f"step failed: {' '.join(cmd[2:])}")
        log("done")
        with LOCK:
            STATE.update(status="done", finished=datetime.now().isoformat())
    except Exception as e:
        log(f"ERROR {e}")
        with LOCK:
            STATE.update(status="error", finished=datetime.now().isoformat())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the terminal quiet

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/dashboard.html"):
            try:
                with open("dashboard.html", "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                return self._send(404, b"dashboard.html not built yet - run a projection first",
                                  "text/plain")
        if path == "/api/status":
            with LOCK:
                body = json.dumps(STATE).encode()
            return self._send(200, body, "application/json")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path.split("?")[0] != "/api/refresh":
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length") or 0)
        payload = {}
        if length:
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                payload = {}
        with LOCK:
            if STATE["status"] == "running":
                return self._send(409, json.dumps({"status": "running"}).encode(),
                                  "application/json")
        threading.Thread(target=_run_job, kwargs={
            "slate": payload.get("slate"),
            "season": payload.get("season"),
            "week": payload.get("week"),
        }, daemon=True).start()
        return self._send(202, json.dumps({"status": "started"}).encode(), "application/json")


def serve(port: int = 8765, open_browser: bool = True) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"nflproj dashboard: {url}")
    print("press Refresh in the page to update data, weather, injuries and projections")
    print("ctrl-C to stop")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        srv.shutdown()
