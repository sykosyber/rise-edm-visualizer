"""Inputs: ffmpeg, audio decoding, image discovery/loading, fonts, hashing."""
import glob
import hashlib
import os
import subprocess

import numpy as np
from PIL import Image

SAMPLE_RATE = 48000
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def ffmpeg_exe():
    exe = os.environ.get("HALOKIT_FFMPEG")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg_version():
    out = subprocess.run([ffmpeg_exe(), "-version"], capture_output=True, text=True).stdout
    return out.splitlines()[0] if out else "unknown"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def decode_audio(path, sr=SAMPLE_RATE):
    """Decode any ffmpeg-readable audio to mono float32 at `sr`."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"audio not found: {path}")
    cmd = [ffmpeg_exe(), "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {res.stderr.decode(errors='replace')}")
    x = np.frombuffer(res.stdout, dtype="<f4")
    if x.size == 0:
        raise RuntimeError(f"no audio samples decoded from {path}")
    return x.astype(np.float32)


def discover_images(specs):
    """Expand files / directories / glob patterns into an ordered, de-duplicated list.

    Directories expand to their image files sorted by name (non-recursive).
    Byte-identical files are dropped after their first occurrence.
    Returns (paths, duplicates) with absolute paths.
    """
    found = []
    for spec in specs:
        if os.path.isdir(spec):
            found += sorted(os.path.join(spec, f) for f in os.listdir(spec)
                            if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        elif os.path.isfile(spec):
            found.append(spec)
        else:
            matches = sorted(glob.glob(spec))
            if not matches:
                raise FileNotFoundError(f"no images match: {spec}")
            found += [m for m in matches if os.path.splitext(m)[1].lower() in IMAGE_EXTS]
    paths, dupes, seen = [], [], {}
    for p in found:
        p = os.path.abspath(p)
        h = sha256_file(p)
        if h in seen:
            dupes.append((p, seen[h]))
            continue
        seen[h] = p
        paths.append(p)
    if not paths:
        raise ValueError("no images given")
    return paths, dupes


def load_rgb(path):
    """First frame of any image as RGB; transparency is composited over black."""
    im = Image.open(path)
    im.seek(0)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def cover(im, w, h, resample=Image.LANCZOS):
    """CSS object-fit: cover - scale to fill w x h, centre-crop the overflow."""
    iw, ih = im.size
    s = max(w / iw, h / ih)
    sw, sh = max(w, round(iw * s)), max(h, round(ih * s))
    im = im.resize((sw, sh), resample)
    x0, y0 = (sw - w) // 2, (sh - h) // 2
    return im.crop((x0, y0, x0 + w, y0 + h))


_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]


def find_font(preferred=None):
    """Resolve the brand font. Returns a path, or None to fall back to PIL's bitmap font."""
    if preferred:
        if not os.path.isfile(preferred):
            raise FileNotFoundError(f"font not found: {preferred}")
        return os.path.abspath(preferred)
    for c in _FONT_CANDIDATES:
        if os.path.isfile(c):
            return c
    return None
