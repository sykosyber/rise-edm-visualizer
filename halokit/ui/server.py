"""halokit local web UI: a small stdlib HTTP server + one static page.

Renders run as `python -m halokit ...` subprocesses (the same code path as the
CLI), so the UI never produces anything the command line couldn't reproduce:
every finished job has a manifest and a copyable CLI command.

Workspace layout:
  uploads/   files added through the UI (content-addressed names)
  jobs/      <id>.json job files + <id>.status.json
  outputs/   rendered videos/stills + their .job.json manifests
  previews/  quick preview stills
"""
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from .. import __version__, media
from ..config import Job

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))       # folder containing the halokit package
JOB_FIELDS = {f.name for f in fields(Job)}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".wma"}


class Workspace:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        for d in ("uploads", "jobs", "outputs", "previews"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        self.lock = threading.Lock()
        self.jobs = {}                      # id -> status dict
        self.procs = {}                     # id -> Popen
        for fn in sorted(os.listdir(os.path.join(self.root, "jobs"))):
            if fn.endswith(".status.json"):
                with open(os.path.join(self.root, "jobs", fn), encoding="utf-8") as f:
                    st = json.load(f)
                if st.get("state") in ("queued", "running"):
                    st["state"] = "interrupted"
                self.jobs[st["id"]] = st

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def rel(self, p):
        return os.path.relpath(p, self.root).replace("\\", "/")

    def resolve(self, rel):
        """Workspace-relative path -> absolute, refusing anything outside the workspace."""
        p = os.path.abspath(os.path.join(self.root, rel))
        if os.path.commonpath([p, self.root]) != self.root:
            raise PermissionError(rel)
        return p

    # ---- inputs --------------------------------------------------------------------------
    def store(self, name, data):
        name = re.sub(r"[^\w.\- ]+", "_", os.path.basename(name)).strip() or "file"
        digest = hashlib.sha256(data).hexdigest()
        dest = self.path("uploads", f"{digest[:12]}-{name}")
        if not os.path.exists(dest):
            with open(dest, "wb") as f:
                f.write(data)
        return self.describe(dest)

    def import_paths(self, paths):
        """Copy local files (or every image in a folder) into the workspace."""
        out = []
        for p in paths:
            p = os.path.expanduser(p.strip().strip('"'))
            if os.path.isdir(p):
                files, _ = media.discover_images([p])
            elif os.path.isfile(p):
                files = [p]
            else:
                raise FileNotFoundError(p)
            for f in files:
                with open(f, "rb") as fh:
                    out.append(self.store(os.path.basename(f), fh.read()))
        return out

    def describe(self, p):
        ext = os.path.splitext(p)[1].lower()
        kind = "image" if ext in media.IMAGE_EXTS else "audio" if ext in AUDIO_EXTS else "other"
        return {"path": self.rel(p), "name": os.path.basename(p)[13:], "kind": kind,
                "url": "/files/" + self.rel(p), "bytes": os.path.getsize(p)}

    def uploads(self):
        d = self.path("uploads")
        return [self.describe(os.path.join(d, f)) for f in sorted(os.listdir(d))]

    # ---- jobs ------------------------------------------------------------------------------------
    def make_job(self, spec, kind, output):
        data = {k: v for k, v in spec.items() if k in JOB_FIELDS}
        data["kind"] = kind
        data["audio"] = self.resolve(data.get("audio", ""))
        data["images"] = [self.resolve(p) for p in data.get("images", [])]
        data["output"] = output
        return Job.from_dict(data).validate()

    def start(self, spec):
        jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        name = re.sub(r"[^\w\-]+", "-", spec.get("name") or "") or \
            os.path.splitext(os.path.basename(spec.get("audio", "render")))[0][13:] or "render"
        kind = spec.get("kind", "video")
        ext = ".png" if kind == "still" else ".mp4"
        output = self.path("outputs", f"{name}-{jid}{ext}")
        job = self.make_job(spec, kind, output)
        job_file = self.path("jobs", f"{jid}.json")
        job.save(job_file)
        st = {"id": jid, "name": name, "kind": kind, "state": "queued", "done": 0, "total": 0,
              "elapsed": 0, "created": time.time(), "output": self.rel(output),
              "manifest": self.rel(os.path.splitext(output)[0] + ".job.json"),
              "job_file": self.rel(job_file), "log": [], "error": None,
              "command": cli_command(job_file)}
        with self.lock:
            self.jobs[jid] = st
        self._save(st)
        threading.Thread(target=self._run, args=(jid, job_file, kind), daemon=True).start()
        return st

    def _run(self, jid, job_file, kind):
        cmd = [sys.executable, "-m", "halokit", "render" if kind == "video" else "still",
               "--config", job_file, "--progress", "json"]
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=_env())
        with self.lock:
            self.procs[jid] = proc
            self.jobs[jid]["state"] = "running"
        for line in proc.stdout:
            line = line.strip()
            try:
                ev = json.loads(line)
            except ValueError:
                ev = {"event": "log", "message": line}
            with self.lock:
                st = self.jobs[jid]
                if ev.get("event") == "progress":
                    st.update(done=ev["done"], total=ev["total"], elapsed=ev["elapsed"])
                elif ev.get("event") == "log" and ev.get("message"):
                    st["log"] = (st["log"] + [ev["message"]])[-20:]
                elif ev.get("event") == "error":
                    st["error"] = ev["message"]
                elif ev.get("event") == "done":
                    st["result"] = ev.get("result")
        code = proc.wait()
        with self.lock:
            st = self.jobs[jid]
            if st["state"] == "cancelled":
                pass
            elif code == 0 and os.path.exists(self.resolve(st["output"])):
                st["state"] = "done"
            else:
                st["state"] = "failed"
                st["error"] = st["error"] or "\n".join(st["log"][-5:]) or f"exit code {code}"
            self.procs.pop(jid, None)
        self._save(st)

    def cancel(self, jid):
        with self.lock:
            proc = self.procs.get(jid)
            if jid in self.jobs:
                self.jobs[jid]["state"] = "cancelled"
        if proc:
            _kill_tree(proc)
        return True

    def _save(self, st):
        with open(self.path("jobs", f"{st['id']}.status.json"), "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2)

    def preview(self, spec, width, height):
        """Render one still synchronously (small) and return its URL."""
        pid = uuid.uuid4().hex[:10]
        out = self.path("previews", f"{pid}.png")
        spec = dict(spec, width=width, height=height)
        job = self.make_job(spec, "still", out)
        job_file = self.path("previews", f"{pid}.json")
        job.save(job_file)
        res = subprocess.run([sys.executable, "-m", "halokit", "still", "--config", job_file,
                              "--progress", "json"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                             timeout=300, env=_env())
        if res.returncode != 0 or not os.path.exists(out):
            msgs = [json.loads(l).get("message", "") for l in res.stdout.splitlines() if l.startswith("{")]
            raise RuntimeError("; ".join(m for m in msgs if m) or res.stderr[-400:])
        with open(os.path.splitext(out)[0] + ".job.json", encoding="utf-8") as f:
            at = json.load(f)["result"]["first_frame_time"]
        return {"url": "/files/" + self.rel(out), "at": at}


def cli_command(job_file):
    return f'python -m halokit render --config "{os.path.abspath(job_file)}"'


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _kill_tree(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


class Handler(BaseHTTPRequestHandler):
    ws: Workspace = None
    server_version = f"halokit/{__version__}"

    def log_message(self, fmt, *args):
        pass

    # ---- helpers ---------------------------------------------------------------------------
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def fail(self, e, code=400):
        self.send_json({"error": str(e) or e.__class__.__name__}, code)

    # ---- routes --------------------------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                return self.send_file(os.path.join(HERE, "index.html"))
            if url.path == "/api/state":
                with self.ws.lock:
                    jobs = sorted((dict(j) for j in self.ws.jobs.values()), key=lambda j: -j["created"])
                return self.send_json({"version": __version__, "defaults": asdict(Job()),
                                       "uploads": self.ws.uploads(), "jobs": jobs,
                                       "workspace": self.ws.root})
            if url.path.startswith("/files/"):
                return self.send_file(self.ws.resolve(unquote(url.path[len("/files/"):])))
            self.send_error(404)
        except PermissionError:
            self.send_error(403)
        except FileNotFoundError:
            self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/api/upload":
                return self.send_json(self.ws.store(q.get("name", ["file"])[0], self.body()))
            if url.path == "/api/import":
                return self.send_json(self.ws.import_paths(json.loads(self.body())["paths"]))
            if url.path == "/api/render":
                return self.send_json(self.ws.start(json.loads(self.body())))
            if url.path == "/api/preview":
                spec = json.loads(self.body())
                w, h = int(spec.get("width", 1920)), int(spec.get("height", 1080))
                s = min(1.0, 960 / max(w, h))
                pw, ph = max(64, int(w * s) // 2 * 2), max(64, int(h * s) // 2 * 2)
                return self.send_json(self.ws.preview(spec, pw, ph))
            m = re.fullmatch(r"/api/jobs/([\w\-]+)/cancel", url.path)
            if m:
                return self.send_json({"ok": self.ws.cancel(m.group(1))})
            self.send_error(404)
        except (ValueError, KeyError, FileNotFoundError, PermissionError, RuntimeError,
                subprocess.TimeoutExpired) as e:
            self.fail(e)

    def send_file(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng or "")
        if m and size:
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1 if size else 0))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass


def serve(host="127.0.0.1", port=8765, workspace="halokit-workspace", open_browser=False):
    Handler.ws = Workspace(workspace)
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"halokit UI on {url}  (workspace: {Handler.ws.root})  Ctrl+C to stop", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in list(Handler.ws.procs.values()):
            _kill_tree(proc)
        httpd.server_close()
