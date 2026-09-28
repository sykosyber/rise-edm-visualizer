"""Render orchestration: prepare -> parallel frames -> encode -> manifest; verify."""
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import ENGINE_VERSION, __version__, media
from .analysis import SIM_HZ, Timeline, loudest_time, simulate
from .config import Job, manifest_path_for
from .scene import Renderer

CHECKPOINT_EVERY = 150


@dataclass
class Prepared:
    job: Job
    images: list
    duplicates: list
    samples: np.ndarray
    timeline: Timeline
    font: str
    t0: float          # first frame time (s)
    count: int         # frames to render
    audio_seconds: float


def prepare(job, log=None):
    job.validate()
    images, dupes = media.discover_images(job.images)
    for d, keep in dupes:
        _log(log, f"skipping duplicate image {os.path.basename(d)} (same bytes as {os.path.basename(keep)})")
    job.images = images                      # the manifest records the exact resolved list
    job.audio = os.path.abspath(job.audio)
    samples = media.decode_audio(job.audio)
    secs = len(samples) / media.SAMPLE_RATE
    if job.start >= secs:
        raise ValueError(f"start {job.start}s is past the end of the audio ({secs:.2f}s)")
    tl = simulate(samples, media.SAMPLE_RATE, job, len(images))
    font = media.find_font(job.font) if job.show_brand else None
    if job.kind == "still":
        if job.still_at is None:
            end = None if job.duration is None else job.start + job.duration
            t0 = loudest_time(tl, job.start, end)
        else:
            t0 = job.still_at
        count = 1
    else:
        dur = min(job.duration or secs, secs - job.start)
        t0, count = job.start, max(1, int(math.floor(dur * job.fps + 1e-6)))
    _log(log, f"{len(images)} image(s), {secs:.1f}s audio, {tl.beats} beats, "
              f"{tl.changes} image changes, {count} frame(s)")
    return Prepared(job, images, dupes, samples, tl, font, t0, count, secs)


# ---- parallel frame production ---------------------------------------------
_W = {}


def _init_worker(job_dict, images, font, tl, samples):
    _W["r"] = Renderer(Job.from_dict(job_dict), images, font)
    _W["tl"], _W["s"] = tl, samples


def _render_frame(t):
    return _W["r"].frame(_W["tl"], t, _W["s"]).tobytes()


def default_workers():
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def iter_frames(prep, count=None, workers=None):
    """Yield raw RGB frames in order. Output is independent of `workers`."""
    count = prep.count if count is None else min(count, prep.count)
    times = [prep.t0 + i / prep.job.fps for i in range(count)]
    workers = default_workers() if workers is None else max(1, workers)
    args = (prep.job.to_dict(), prep.images, prep.font, prep.timeline, prep.samples)
    if workers == 1 or count < 8:
        _init_worker(*args)
        for t in times:
            yield _render_frame(t)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=args) as pool:
        yield from pool.imap(_render_frame, times, chunksize=2)


class FrameHasher:
    def __init__(self):
        self.h = hashlib.sha256()
        self.n = 0
        self.checkpoints = {}

    def update(self, b):
        self.h.update(b)
        self.n += 1
        if self.n % CHECKPOINT_EVERY == 0:
            self.checkpoints[str(self.n)] = self.h.hexdigest()

    def final(self):
        return self.h.hexdigest()


# ---- encode ------------------------------------------------------------------
def encoder_args(job, start, seconds):
    return (["-ss", f"{start:.6f}", "-t", f"{seconds:.6f}", "-i", job.audio,
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "libx264", "-preset", job.preset, "-crf", str(job.crf), "-tune", "film",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", job.audio_bitrate, "-shortest",
             "-map_metadata", "-1"]
            + (["-metadata", f"title={job.title}"] if job.title else [])
            + ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
               "-movflags", "+faststart"])


def render(job, workers=None, progress=None, log=None):
    """Render a Job (video or still). Returns the manifest dict (also written to disk)."""
    started = time.time()
    prep = prepare(job, log)
    out = os.path.abspath(job.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    stem, ext = os.path.splitext(out)
    partial = f"{stem}.partial{ext}"
    hasher = FrameHasher()

    if job.kind == "still":
        frame = next(iter_frames(prep, workers=1))
        hasher.update(frame)
        Image.frombytes("RGB", (job.width, job.height), frame).save(partial, format="PNG")
        if progress:
            progress(1, 1, time.time() - started)
    else:
        seconds = prep.count / job.fps
        cmd = ([media.ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{job.width}x{job.height}", "-r", str(job.fps), "-i", "-"]
               + encoder_args(job, job.start, seconds) + ["-f", "mp4", partial])
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i, frame in enumerate(iter_frames(prep, workers=workers)):
                enc.stdin.write(frame)
                hasher.update(frame)
                if progress:
                    progress(i + 1, prep.count, time.time() - started)
            enc.stdin.close()
            err = enc.stderr.read().decode(errors="replace")
            enc.stderr.close()
            if enc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed: {err}")
        except BaseException:
            enc.kill()
            enc.wait()
            if os.path.exists(partial):
                os.remove(partial)
            raise
    os.replace(partial, out)

    manifest = {
        "halokit": __version__,
        "engine": ENGINE_VERSION,
        "job": job.to_dict(relative_to=os.path.dirname(out)),
        "inputs": _inputs(prep),
        "env": environment(prep.font),
        "result": {
            "kind": job.kind,
            "first_frame_time": round(prep.t0, 6),
            "frames": prep.count,
            "frame_digest": hasher.final(),
            "checkpoints": hasher.checkpoints,
            "output_sha256": media.sha256_file(out),
            "beats": prep.timeline.beats,
            "image_changes": prep.timeline.changes,
            "render_seconds": round(time.time() - started, 2),
        },
    }
    mpath = manifest_path_for(out)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    _log(log, f"wrote {out}\nwrote {mpath}")
    return manifest


def verify(manifest_path, frames=None, workers=None, log=None):
    """Re-render and compare against a manifest. Returns (ok, report lines)."""
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    job = Job.from_dict(m, base_dir=os.path.dirname(os.path.abspath(manifest_path)))
    lines, ok = [], True
    if m.get("engine") != ENGINE_VERSION:
        lines.append(f"WARN engine {m.get('engine')} != installed {ENGINE_VERSION}; pixels may differ")
    rec = m["inputs"]
    for item in [rec["audio"]] + rec["images"]:
        p = _resolve(item["path"], manifest_path)
        if not os.path.isfile(p):
            lines.append(f"FAIL missing input {p}")
            ok = False
        elif media.sha256_file(p) != item["sha256"]:
            lines.append(f"FAIL input changed {p}")
            ok = False
    if rec.get("font") and job.font is None:
        cur = media.find_font(None)
        if not cur or media.sha256_file(cur) != rec["font"]["sha256"]:
            lines.append("WARN brand font differs from the recorded one; brand pixels may differ")
    if not ok:
        return False, lines

    prep = prepare(job, log)
    res = m["result"]
    if job.kind == "still" or frames is None or frames >= res["frames"]:
        n, want = res["frames"], res["frame_digest"]
    else:
        cps = sorted(int(c) for c in res["checkpoints"])
        eligible = [c for c in cps if c >= frames] or cps
        if not eligible:
            n, want = res["frames"], res["frame_digest"]
        else:
            n = eligible[0]
            want = res["checkpoints"][str(n)]
    hasher = FrameHasher()
    for frame in iter_frames(prep, count=n, workers=workers):
        hasher.update(frame)
    got = hasher.final()
    if got == want:
        lines.append(f"OK   {n} frame(s) reproduce bit-exactly ({got[:16]}...)")
    else:
        lines.append(f"FAIL frame digest over {n} frame(s): {got[:16]} != {want[:16]}")
        ok = False
    return ok, lines


# ---- helpers -------------------------------------------------------------------
def _inputs(prep):
    base = os.path.dirname(os.path.abspath(prep.job.output))
    from .config import _rel
    d = {"audio": {"path": _rel(prep.job.audio, base), "sha256": media.sha256_file(prep.job.audio),
                   "seconds": round(prep.audio_seconds, 3)},
         "images": [{"path": _rel(p, base), "sha256": media.sha256_file(p)} for p in prep.images]}
    if prep.font:
        d["font"] = {"path": prep.font, "sha256": media.sha256_file(prep.font)}
    return d


def _resolve(p, manifest_path):
    return p if os.path.isabs(p) else os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(manifest_path)), p))


def environment(font=None):
    import PIL
    import scipy
    try:
        import imageio_ffmpeg
        iff = imageio_ffmpeg.__version__
    except ImportError:
        iff = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__,
            "imageio_ffmpeg": iff, "ffmpeg": media.ffmpeg_version(), "sim_hz": SIM_HZ}


def _log(log, msg):
    if log:
        log(msg)
