"""halokit command line.

  python -m halokit render SONG.mp3 IMG [IMG|DIR|GLOB ...] -o out.mp4 [options]
  python -m halokit still  SONG.mp3 IMG ... -o out.png [--at 62.5]
  python -m halokit render --config out.job.json        # reproduce a render
  python -m halokit run jobs/*.json                     # render job files (video or still)
  python -m halokit verify out.job.json [--frames 300]  # prove it reproduces
  python -m halokit ui                                  # local web interface
"""
import argparse
import json
import os
import sys
import time

from . import __version__
from .config import Job

# CLI flag -> Job field for options shared by render/still
_FIELDS = {
    "fps": "fps", "seed": "seed", "start": "start", "duration": "duration", "halo": "halo",
    "order": "image_order", "beats_per_change": "beats_per_change", "min_hold": "min_hold",
    "fallback_min": "fallback_min", "fallback_max": "fallback_max", "crossfade": "crossfade",
    "grain": "grain", "brand": "brand", "subbrand": "subbrand", "font": "font", "title": "title",
    "crf": "crf", "preset": "preset", "audio_bitrate": "audio_bitrate",
}


def _size(s):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 1920x1080")


def _job_options(p, still):
    p.add_argument("audio", nargs="?", help="music file (anything ffmpeg reads)")
    p.add_argument("images", nargs="*", help="image files, directories or glob patterns")
    p.add_argument("-o", "--output", help="output path (default: <audio name>.mp4/.png)")
    p.add_argument("--config", help="start from a job or manifest JSON; flags override it")
    g = p.add_argument_group("look")
    g.add_argument("--size", type=_size, help="WxH, default 1920x1080")
    g.add_argument("--fps", type=int)
    g.add_argument("--seed", type=int, help="all randomness derives from this (default 0)")
    g.add_argument("--halo", type=float, help="halo intensity, default 1.0 (app: .65/1/1.35)")
    g.add_argument("--order", choices=["shuffle", "sequence"], help="image order")
    g.add_argument("--beats-per-change", type=int, help="change image every N beats (0=off)")
    g.add_argument("--min-hold", type=float, help="min seconds between beat changes")
    g.add_argument("--fallback-min", type=float, help="quiet-passage change after [min,max] s")
    g.add_argument("--fallback-max", type=float)
    g.add_argument("--crossfade", type=float, help="seconds")
    g.add_argument("--grain", type=float, help="film grain amount, 0 disables")
    g.add_argument("--brand", help="brand line (default RISE RADIO)")
    g.add_argument("--subbrand", help="second brand line")
    g.add_argument("--no-brand", action="store_true", help="hide brand text and meter")
    g.add_argument("--font", help="TTF/OTF for the brand text")
    t = p.add_argument_group("timing")
    t.add_argument("--start", type=float, help="seconds into the audio")
    t.add_argument("--duration", type=float, help="seconds to render")
    if still:
        t.add_argument("--at", help="time in seconds, or 'peak' for the loudest moment (default)")
    else:
        e = p.add_argument_group("encoding")
        e.add_argument("--title", help="metadata title")
        e.add_argument("--crf", type=int, help="x264 quality, lower = better (default 20)")
        e.add_argument("--preset", help="x264 preset (default medium)")
        e.add_argument("--audio-bitrate")
        e.add_argument("--workers", type=int, help="render processes (default: cores-1, max 8)")
    p.add_argument("--save-job", metavar="PATH", help="also write the resolved job JSON here")
    p.add_argument("--dry-run", action="store_true", help="resolve and save the job, don't render")
    p.add_argument("--progress", choices=["text", "json", "none"], default="text")


def build_job(a, kind):
    job = Job.load(a.config) if a.config else Job()
    job.kind = kind
    if a.audio:
        job.audio = a.audio
    if a.images:
        job.images = a.images
    for flag, field in _FIELDS.items():
        v = getattr(a, flag, None)
        if v is not None:
            setattr(job, field, v)
    if a.size:
        job.width, job.height = a.size
    if a.no_brand:
        job.show_brand = False
    if kind == "still":
        if a.at is not None:
            job.still_at = None if a.at == "peak" else float(a.at)
    ext = ".png" if kind == "still" else ".mp4"
    if a.output:
        job.output = a.output
    elif not job.output:
        job.output = os.path.splitext(os.path.basename(job.audio or "halokit"))[0] + ext
    elif os.path.splitext(job.output)[1].lower() != ext:   # e.g. still from a video job
        job.output = os.path.splitext(job.output)[0] + ext
    job.output = os.path.abspath(job.output)
    return job.validate()


def _progress_printer(mode):
    if mode == "none":
        return None
    last = [0.0]

    def text(done, total, elapsed):
        now = time.time()
        if done != total and now - last[0] < 1.0:
            return
        last[0] = now
        rate = done / elapsed if elapsed else 0
        eta = (total - done) / rate if rate else 0
        sys.stderr.write(f"\r  {done}/{total} frames  {100 * done / total:5.1f}%  "
                         f"{rate:5.1f} fps  eta {eta:5.0f}s ")
        if done == total:
            sys.stderr.write("\n")
        sys.stderr.flush()

    def js(done, total, elapsed):
        now = time.time()
        if done != total and now - last[0] < 0.5:
            return
        last[0] = now
        print(json.dumps({"event": "progress", "done": done, "total": total,
                          "elapsed": round(elapsed, 2)}), flush=True)

    return js if mode == "json" else text


def _log_for(mode):
    if mode == "json":
        return lambda m: print(json.dumps({"event": "log", "message": m}), flush=True)
    if mode == "none":
        return None
    return lambda m: print(m, file=sys.stderr, flush=True)


def cmd_render(a, kind):
    from .render import render
    job = build_job(a, kind)
    if a.save_job:
        job.save(a.save_job)
    if a.dry_run:
        print(json.dumps(job.to_dict(), indent=2))
        return 0
    m = render(job, workers=getattr(a, "workers", None), progress=_progress_printer(a.progress),
               log=_log_for(a.progress))
    if a.progress == "json":
        print(json.dumps({"event": "done", "output": job.output,
                          "manifest": os.path.splitext(job.output)[0] + ".job.json",
                          "result": m["result"]}), flush=True)
    else:
        r = m["result"]
        print(f"done in {r['render_seconds']}s  frames={r['frames']}  "
              f"digest={r['frame_digest'][:16]}")
    return 0


def cmd_run(a):
    import glob
    from .render import render
    files = [f for pat in a.jobs for f in (sorted(glob.glob(pat)) or [pat])]
    for i, path in enumerate(files, 1):
        job = Job.load(path)
        print(f"[{i}/{len(files)}] {os.path.basename(path)} -> {job.kind} {job.output}", file=sys.stderr, flush=True)
        render(job, workers=a.workers, progress=_progress_printer(a.progress), log=_log_for(a.progress))
    return 0


def cmd_verify(a):
    from .render import verify
    ok, lines = verify(a.manifest, frames=a.frames, workers=a.workers,
                       log=lambda m: print(m, file=sys.stderr))
    for line in lines:
        print(line)
    return 0 if ok else 1


def cmd_ui(a):
    from .ui.server import serve
    serve(host=a.host, port=a.port, workspace=a.workspace, open_browser=a.open)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="halokit", description="Set any images to any music.")
    p.add_argument("--version", action="version", version=f"halokit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    _job_options(sub.add_parser("render", help="render a video"), still=False)
    _job_options(sub.add_parser("still", help="render a single frame"), still=True)
    r = sub.add_parser("run", help="render job/manifest files (each as its own kind)")
    r.add_argument("jobs", nargs="+", help="job JSON files or glob patterns")
    r.add_argument("--workers", type=int)
    r.add_argument("--progress", choices=["text", "json", "none"], default="text")
    v = sub.add_parser("verify", help="re-render and compare with a manifest")
    v.add_argument("manifest")
    v.add_argument("--frames", type=int, help="only check the first N frames (rounded up to a checkpoint)")
    v.add_argument("--workers", type=int)
    u = sub.add_parser("ui", help="start the local web interface")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--workspace", default="halokit-workspace")
    u.add_argument("--open", action="store_true", help="open a browser tab")
    a = p.parse_args(argv)
    try:
        if a.cmd in ("render", "still"):
            return cmd_render(a, "video" if a.cmd == "render" else "still")
        if a.cmd == "run":
            return cmd_run(a)
        if a.cmd == "verify":
            return cmd_verify(a)
        return cmd_ui(a)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        if getattr(a, "progress", "text") == "json":
            print(json.dumps({"event": "error", "message": str(e)}), flush=True)
        else:
            print(f"error: {e}", file=sys.stderr)
        return 2
