"""Job: the complete, serialisable description of one render.

A job file (or the `job` section of a manifest) fully determines the output
pixels given the same input files and engine version.
"""
import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional

DEFAULT_BRAND = "RISE RADIO"
DEFAULT_SUBBRAND = "SYBERLABS · AUDIO/IMAGE REACTIVE SYSTEM"


@dataclass
class Job:
    audio: str = ""
    images: List[str] = field(default_factory=list)
    output: str = ""
    kind: str = "video"              # "video" | "still"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    seed: int = 0
    start: float = 0.0               # seconds into the audio
    duration: Optional[float] = None  # seconds; None = to the end
    still_at: Optional[float] = None  # stills: seconds; None = loudest moment
    halo: float = 1.0                # halo intensity (app stops: 0.65 / 1 / 1.35)
    image_order: str = "shuffle"     # "shuffle" | "sequence"
    beats_per_change: int = 16       # change image every N detected beats ...
    min_hold: float = 6.5            # ... but never sooner than this (s)
    fallback_min: float = 15.0       # quiet passages: change after a random
    fallback_max: float = 24.0       # hold in [fallback_min, fallback_max] (s)
    crossfade: float = 1.35          # s
    grain: float = 1.0
    show_brand: bool = True
    brand: str = DEFAULT_BRAND
    subbrand: str = DEFAULT_SUBBRAND
    font: Optional[str] = None
    title: Optional[str] = None      # container metadata title
    crf: int = 20
    preset: str = "medium"
    audio_bitrate: str = "192k"

    PATH_FIELDS = ("audio", "output", "font")

    def validate(self):
        errs = []
        if not self.audio:
            errs.append("audio is required")
        if not self.images:
            errs.append("at least one image is required")
        if not self.output:
            errs.append("output is required")
        if self.kind not in ("video", "still"):
            errs.append("kind must be 'video' or 'still'")
        if self.image_order not in ("shuffle", "sequence"):
            errs.append("image_order must be 'shuffle' or 'sequence'")
        if self.width < 64 or self.height < 64 or self.width % 2 or self.height % 2:
            errs.append("width/height must be even and >= 64")
        if not 1 <= self.fps <= 120:
            errs.append("fps must be 1..120")
        if self.fallback_min > self.fallback_max:
            errs.append("fallback_min must be <= fallback_max")
        if self.start < 0 or (self.duration is not None and self.duration <= 0):
            errs.append("start must be >= 0 and duration > 0")
        if not 0 <= self.crf <= 51:
            errs.append("crf must be 0..51")
        if errs:
            raise ValueError("invalid job: " + "; ".join(errs))
        return self

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self, relative_to=None):
        d = asdict(self)
        if relative_to:
            base = os.path.abspath(relative_to)
            for k in self.PATH_FIELDS:
                if d[k]:
                    d[k] = _rel(d[k], base)
            d["images"] = [_rel(p, base) for p in d["images"]]
        return d

    @classmethod
    def from_dict(cls, d, base_dir=None):
        d = dict(d.get("job", d))  # accept a bare job or a full manifest
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown job keys: {sorted(unknown)}")
        job = cls(**d)
        if base_dir:
            for k in cls.PATH_FIELDS:
                v = getattr(job, k)
                if v and not os.path.isabs(v):
                    setattr(job, k, os.path.normpath(os.path.join(base_dir, v)))
            job.images = [p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))
                          for p in job.images]
        return job

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f), base_dir=os.path.dirname(os.path.abspath(path)))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(relative_to=os.path.dirname(os.path.abspath(path))), f, indent=2)
            f.write("\n")


def _rel(p, base):
    p = os.path.abspath(p)
    try:
        return os.path.relpath(p, base).replace("\\", "/")
    except ValueError:  # different drive on Windows
        return p.replace("\\", "/")


def manifest_path_for(output):
    stem, _ = os.path.splitext(output)
    return stem + ".job.json"
