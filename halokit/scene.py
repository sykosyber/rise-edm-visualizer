"""Frame compositor: a faithful port of the RISE RADIO stage (index.html +
styles.css) and halo (<canvas>, visualizer.js), built on OpenCV + numpy.

Speed without losing quality:
  * The layers under the artwork (page gradient, ambient blobs, the 54px-blurred
    backdrop, the art-frame box-shadow) are all low-frequency, so they are
    composited at 1/4 resolution and upscaled once.
  * Full-resolution work is limited to where detail lives: the art frame, the
    halo's bounding box, plus the grain, vignette and final 8-bit conversion.
  * The halo is drawn with OpenCV's anti-aliased sub-pixel primitives into
    premultiplied RGBA layers combined with saturating adds - exactly canvas
    'lighter' - and its shadowBlur is rendered at 1/4 res and blurred once.
  * Everything static is pre-baked once per worker.

All compositing follows the CSS/canvas maths: premultiplied source-over,
screen, soft-light, overlay, and the Filter Effects colour matrices.
A frame is a pure function of (job, images, timeline, t).
"""
import colorsys
import math
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import media
from .analysis import FFT

cv2.setNumThreads(1)          # one thread per worker process: no oversubscription, deterministic

Q = 4                         # resolution divisor for the low-frequency under-layers
SH = 4                        # OpenCV drawing sub-pixel bits (1/16 px)
ONE = float(1 << SH)
GLOW_LEVELS, GLOW_RANGE = 96, (0.3, 3.6)


class _LRU:
    def __init__(self, cap):
        self.cap, self.d = cap, OrderedDict()

    def get(self, key, make):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        v = self.d[key] = make()
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return v


# ---- helpers ----------------------------------------------------------------------
def _hsl(h, s, l):
    return colorsys.hls_to_rgb((h % 360) / 360.0, min(max(l, 0.0), 1.0), min(max(s, 0.0), 1.0))


def _hsl_v(h, l):
    """Vectorised CSS hsl(h, 100%, l) -> (n, 3) RGB in 0..1."""
    h = np.asarray(h, np.float64) % 360
    l = np.broadcast_to(np.clip(np.asarray(l, np.float64), 0, 1), h.shape)
    a = np.minimum(l, 1 - l)
    out = []
    for n in (0, 8, 4):
        kk = (n + h / 30) % 12
        out.append(l - a * np.clip(np.minimum(kk - 3, 9 - kk), -1, 1))
    return np.stack(out, -1)


def _ease(x):
    """cubic-bezier(.25,.8,.25,1)-like: the CSS opacity transitions."""
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def _ease_out(x):
    """CSS ease-out, used by the transform transitions."""
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 2


def _blurred_box(coords, a, b, sigma):
    """Gaussian-blurred indicator of [a, b) evaluated at `coords` (exact, via erf)."""
    if sigma <= 1e-6:
        return ((coords >= a) & (coords < b)).astype(np.float64)
    coords = np.asarray(coords, np.float64)
    s = sigma * math.sqrt(2)
    return (0.5 * (_erf((coords - a) / s) - _erf((coords - b) / s))).astype(np.float64)


_erf = np.frompyfunc(math.erf, 1, 1)   # 1-D profiles only: exact and avoids importing scipy


def _blur(img, sigma_y, sigma_x=None, border=cv2.BORDER_CONSTANT):
    """Gaussian blur (per channel) with a transparent/zero outside by default."""
    sigma_x = sigma_y if sigma_x is None else sigma_x
    if sigma_x <= 0 and sigma_y <= 0:
        return img
    return cv2.GaussianBlur(np.ascontiguousarray(img, np.float32), (0, 0), max(sigma_x, 1e-3),
                            sigmaY=max(sigma_y, 1e-3), borderType=border)


def _coverage(n0, n1, a, b):
    """Fraction of each pixel [x, x+1) for x in n0..n1-1 covered by [a, b)."""
    x = np.arange(n0, n1, dtype=np.float32)
    return np.clip(np.minimum(x + 1, b) - np.maximum(x, a), 0, 1)


def _sat_matrix(s):
    """CSS saturate() colour matrix (Filter Effects spec)."""
    return np.array([[.213 + .787 * s, .715 - .715 * s, .072 - .072 * s],
                     [.213 - .213 * s, .715 + .285 * s, .072 - .072 * s],
                     [.213 - .213 * s, .715 - .715 * s, .072 + .928 * s]], np.float32)


def _over(dst, rgb, alpha):
    """Premultiplied source-over, in place: dst = dst*(1-a) + rgb. alpha is single-channel."""
    cv2.multiply(dst, cv2.cvtColor(1 - alpha, cv2.COLOR_GRAY2RGB), dst=dst)
    cv2.add(dst, rgb, dst=dst)


def _split(premul):
    """(h, w, 4) premultiplied -> contiguous (rgb, alpha)."""
    c = cv2.split(premul)
    return cv2.merge(c[:3]), c[3]


def _affine(ax, bx, ay, by):
    return np.float32([[ax, 0, bx], [0, ay, by]])


class Renderer:
    def __init__(self, job, images, font_path=None):
        self.job, self.images = job, images
        W, H = self.W, self.H = job.width, job.height
        self.u = u = min(W, H) / 800.0                   # CSS px -> output px
        self.cx, self.cy = W / 2, H / 2
        self.baseR = min(W, H) * 0.302
        self.art_side = min(W, H) * 0.78                 # .art-frame: min(78vw, 78vh)
        self.art_fade = max(job.crossfade, 1e-3)         # .art opacity 1350ms
        self.bd_fade = self.art_fade * 1600 / 1350       # .backdrop opacity 1600ms

        # quarter-res grid whose pixel centres match cv2.resize(.., (W, H))
        self.Wq, self.Hq = -(-W // Q), -(-H // Q)
        self.sx, self.sy = W / self.Wq, H / self.Hq
        self.xq = (np.arange(self.Wq) + .5) * self.sx    # full-res x of each quarter column
        self.yq = (np.arange(self.Hq) + .5) * self.sy

        self._bg_q = self._make_background()
        self._vk3 = self._make_vignette()
        self._grain = self._make_grain()
        self._flash, self._flash_sl = self._make_flash()
        self._brand = self._make_brand(font_path) if job.show_brand else None

        # halo bounding box (the <canvas> only ever draws inside it)
        ext = self.baseR * 1.47 + (70 * job.halo + 40) * u
        self.hx0, self.hy0 = max(0, int(self.cx - ext)), max(0, int(self.cy - ext * .82))
        self.hx1, self.hy1 = min(W, int(self.cx + ext) + 1), min(H, int(self.cy + ext * .82) + 1)
        hw, hh = self.hx1 - self.hx0, self.hy1 - self.hy0
        self.hw, self.hh = hw, hh
        self._acc = np.zeros((hh, hw, 4), np.uint8)      # 'lighter' accumulator
        self._lay = np.zeros((hh, hw, 4), np.uint8)      # one additive layer
        self.qw, self.qh = -(-hw // Q), -(-hh // Q)
        self._shadow = [np.zeros((self.qh, self.qw, 4), np.uint8) for _ in range(2)]

        self.F = np.empty((H, W, 3), np.float32)
        self.T = np.empty((H, W, 3), np.float32)

        self._src = _LRU(4)
        self._backdrop = _LRU(4)
        self._art = _LRU(4)
        self._glass = _LRU(8)
        self._glow = _LRU(GLOW_LEVELS)

    # ---- static layers ---------------------------------------------------------------------
    def _make_background(self):
        """.app radial gradient + the two .ambient blobs, at 1/4 res."""
        W, H, u = self.W, self.H, self.u
        X, Y = np.meshgrid(self.xq, self.yq)
        d = (np.hypot(X - self.cx, Y - self.cy) / math.hypot(self.cx, self.cy))[..., None]
        c0, c1 = np.array([6, 19, 38]) / 255, np.array([1, 4, 10]) / 255
        bg = np.where(d < .58, c0 + (c1 - c0) * (d / .58), c1 * np.clip(1 - (d - .58) / .42, 0, 1))
        for (ox, oy), col in (((.05 * W, .05 * W), (0x1e, 0x70, 0xff)),     # 46vw, blur(130px), .09
                              ((W - .05 * W, H - .05 * W), (0x8a, 0xc9, 0xff))):
            disk = (np.hypot(X - ox, Y - oy) < .23 * W).astype(np.float64)
            m = _blur(disk, 130 * u / self.sy, 130 * u / self.sx)[..., None] * .09
            bg = bg * (1 - m) + np.array(col) / 255 * m
        return bg.astype(np.float32)

    def _make_vignette(self):
        # .vignette: box-shadow inset 0 0 180px 30px rgba(0,0,0,.72)  (blur 180px -> sigma 90px)
        W, H, u = self.W, self.H, self.u
        gx = _blurred_box(np.arange(W) + .5, 30 * u, W - 30 * u, 90 * u)
        gy = _blurred_box(np.arange(H) + .5, 30 * u, H - 30 * u, 90 * u)
        keep = (1 - .72 * (1 - np.outer(gy, gx))).astype(np.float32)
        return cv2.merge([keep, keep, keep])

    def _make_grain(self):
        """.grain: 180px fractal-noise tile, opacity .115, soft-light, at the two positions of
        `animation: grain .18s steps(2)`. Stored as the soft-light gain field per position."""
        W, H, u = self.W, self.H, self.u
        n = max(8, int(round(180 * u)))
        rng = np.random.default_rng([self.job.seed, 0x6772])
        noise = np.zeros((n, n, 3), np.float32)
        for octave in range(4):                          # feTurbulence fractalNoise, numOctaves 4
            step = 2 ** octave
            m = -(-n // step)
            layer = rng.random((m, m, 3), dtype=np.float32)
            noise += np.kron(layer, np.ones((step, step, 1), np.float32))[:n, :n] / step
        pad = 3
        noise = _blur(np.pad(noise, ((pad, pad), (pad, pad), (0, 0)), mode="wrap"), .6)[pad:-pad, pad:-pad]
        s = np.clip((noise - noise.mean()) / (noise.std() * 6) + .5, 0, 1)
        gain = (.115 * .9 * self.job.grain) * (2 * s - 1)     # soft-light: d += a(2s-1)d(1-d)
        fields = []
        for ox, oy in ((-.044 * W, .022 * H), (-.022 * W, .066 * H)):
            ys = (np.arange(H) - int(round(oy))) % n
            xs = (np.arange(W) - int(round(ox))) % n
            fields.append(np.ascontiguousarray(gain[ys][:, xs]))
        return fields

    def _make_flash(self):
        # .flash: radial-gradient(circle, white .62 0%, rgba(120,185,255,.2) 13%, transparent 44%)
        R = math.hypot(self.cx, self.cy) * .44
        x0, x1 = max(0, int(self.cx - R)), min(self.W, int(self.cx + R) + 1)
        y0, y1 = max(0, int(self.cy - R)), min(self.H, int(self.cy + R) + 1)
        X, Y = np.meshgrid(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5)
        t = np.hypot(X - self.cx, Y - self.cy) / math.hypot(self.cx, self.cy)
        c0 = np.array([1, 1, 1, 1]) * .62
        c1 = np.array([120 / 255, 185 / 255, 1, 1]) * .20
        c0[3], c1[3] = .62, .20
        w = np.clip(t / .13, 0, 1)[..., None]
        w2 = np.clip((t - .13) / (.44 - .13), 0, 1)[..., None]
        g = np.where(t[..., None] < .13, c0 * (1 - w) + c1 * w, c1 * (1 - w2))
        return g[..., :3].astype(np.float32), (slice(y0, y1), slice(x0, x1))

    def _make_brand(self, font_path):
        """.brandbar text (with its text-shadow), pre-rendered as premultiplied RGBA."""
        u, job = self.u, self.job
        pad = int(24 * u)
        w, h = int(self.W * .7), int(60 * u) + 2 * pad
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d, ds = ImageDraw.Draw(im), ImageDraw.Draw(shadow)

        def font(px):
            if font_path:
                return ImageFont.truetype(font_path, max(6, int(round(px))))
            return ImageFont.load_default()

        def spaced(text, px, em, fill, top):
            f = font(px)
            x = float(pad)
            for ch in text:
                d.text((x, top), ch, font=f, fill=fill)
                ds.text((x, top + 2 * u), ch, font=f, fill=(0, 0, 0, 191))  # 0 2px 18px .75
                x += d.textlength(ch, font=f) + px * em
        if job.brand:
            spaced(job.brand, 13 * u, .33, (245, 250, 255, 242), pad)
        if job.subbrand:
            spaced(job.subbrand, 9 * u, .18, (220, 232, 255, 143), pad + 13 * u * 1.25 + 7 * u)
        a = np.asarray(im, np.float32) / 255
        sa = _blur(np.asarray(shadow, np.float32)[..., 3] / 255, 9 * u)[..., None]
        prem = np.concatenate([a[..., :3] * a[..., 3:], a[..., 3:]], -1)
        out = np.concatenate([np.zeros_like(prem[..., :3]), sa], -1)
        out *= 1 - prem[..., 3:4]
        out += prem
        return _split(out.astype(np.float32)), (int(24 * u) - pad, int(22 * u) - pad)

    # ---- per-image layers --------------------------------------------------------------------
    def _source(self, i):
        return self._src.get(i, lambda: media.load_rgb(self.images[i]))

    def backdrop(self, i):
        """.backdrop: blur(54px) saturate(1.25) brightness(.48) of the cover-fit image
        (116% of a -8% inset wrap = 134.56% of the stage), premultiplied RGBA at 1/4 res."""
        def make():
            bw, bh = int(round(self.W * 1.3456 / Q)), int(round(self.H * 1.3456 / Q))
            a = np.asarray(media.cover(self._source(i), bw, bh, Image.BILINEAR), np.float32) / 255
            a = np.concatenate([a, np.ones((bh, bw, 1), np.float32)], -1)
            a = _blur(a, 54 * self.u / Q)
            alpha = a[..., 3:]
            rgb = a[..., :3] / np.maximum(alpha, 1e-4)
            rgb = np.clip(rgb @ _sat_matrix(1.25).T, 0, 1) * .48
            return np.ascontiguousarray(np.concatenate([rgb * alpha, alpha], -1), dtype=np.float32)
        return self._backdrop.get(i, make)

    def art(self, i):
        """Cover-fit square art at 1.13x the frame size with the .art filter
        (saturate 1.04, contrast 1.03) baked in; per-frame sizes are downscales of it."""
        def make():
            side = int(math.ceil(self.art_side * 1.13))
            rgb = np.asarray(media.cover(self._source(i), side, side), np.float32) / 255
            rgb = np.clip((rgb @ _sat_matrix(1.04).T - .5) * 1.03 + .5, 0, 1)
            return (rgb * 255 + .5).astype(np.uint8)
        return self._art.get(i, make)

    def glass(self, w, h):
        """.art-glass: linear-gradient(120deg, white .10, transparent 18%, transparent 78%,
        rgba(147,203,255,.08)), screen-blended (its black radial part is a no-op under screen).
        Built once for the unscaled frame; the bass pulse only resizes it."""
        def make():
            n = int(math.ceil(self.art_side))
            ang = math.radians(120)
            dx, dy = math.sin(ang), -math.cos(ang)
            px = ((np.arange(n, dtype=np.float32) + .5 - n / 2) * dx / (n * (abs(dx) + abs(dy))))[None, :]
            py = ((np.arange(n, dtype=np.float32) + .5 - n / 2) * dy / (n * (abs(dx) + abs(dy))) + .5)[:, None]
            p = px + py
            white = np.clip(1 - p / .18, 0, 1) * .10
            blue = np.clip((p - .78) / .22, 0, 1) * .08
            return cv2.merge([white + blue * (147 / 255), white + blue * (203 / 255), white + blue])
        base = self._glass.get("base", make)
        return self._glass.get((w, h), lambda: cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR))

    def glow(self, g):
        """The two .art-frame box-shadows for --art-glow = g (their blur radius scales with g),
        premultiplied RGBA on the 1/4-res grid, quantised to GLOW_LEVELS."""
        lo, hi = GLOW_RANGE
        level = int(round((min(max(g, lo), hi) - lo) / (hi - lo) * (GLOW_LEVELS - 1)))

        def make():
            gv = lo + (hi - lo) * level / (GLOW_LEVELS - 1)
            s = self.art_side
            out = np.zeros((self.Hq, self.Wq, 4), np.float64)
            # first-listed shadow on top: rgba(88,165,255,.30) 80px over rgba(64,94,255,.18) 180px
            for blur, col, alpha in ((180, (64, 94, 255), .18), (80, (88, 165, 255), .30)):
                sig = blur * gv * self.u / 2
                m = np.outer(_blurred_box(self.yq, self.cy - s / 2, self.cy + s / 2, sig),
                             _blurred_box(self.xq, self.cx - s / 2, self.cx + s / 2, sig))
                a = (m * alpha)[..., None]
                out = np.concatenate([np.array(col) / 255 * a, a], -1) + out * (1 - a)
            return out.astype(np.float32)
        return self._glow.get(level, make)

    # ---- frame ------------------------------------------------------------------------------------
    def frame(self, tl, t, samples):
        """Render time t (s) -> uint8 RGB (H, W, 3)."""
        job = self.job
        k = tl.step_at(t)
        low, mid, high, imp = (float(tl.low[k]), float(tl.mid[k]), float(tl.high[k]),
                               float(tl.impact[k]))
        cur, prev, since = int(tl.visual[k]), int(tl.prev_visual[k]), float(tl.since[k])
        W, H, F, T = self.W, self.H, self.F, self.T
        bass = 1 + low * .026 + imp * .016

        # 1) under-layers at 1/4 res: page -> backdrops -> art-frame box-shadow
        U = self._bg_q.copy()
        e_op, e_sc = _ease(since / self.bd_fade), _ease_out(since / 6.5)
        layers = [(cur, .82 * e_op, 1.12 - .10 * e_sc)]       # opacity .82, scale 1.12 -> 1.02
        if prev != cur and e_op < 1:
            layers.insert(0, (prev, .82 * (1 - e_op), 1.02 + .10 * e_sc))
        for i, alpha, scale in layers:
            if alpha > 1e-3:
                b = self.backdrop(i)
                bh, bw = b.shape[:2]
                k_x, k_y = W * 1.3456 * scale / bw, H * 1.3456 * scale / bh
                M = _affine(k_x / self.sx, (self.cx + (.5 - bw / 2) * k_x) / self.sx - .5,
                            k_y / self.sy, (self.cy + (.5 - bh / 2) * k_y) / self.sy - .5)
                L = cv2.warpAffine(b, M, (self.Wq, self.Hq), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
                L *= alpha
                _over(U, *_split(L))
        gq = self.glow(.42 + low * 1.2 + high * .3 + imp * .55)
        M = _affine(bass, (self.cx + (.5 * self.sx - self.cx) * bass) / self.sx - .5,
                    bass, (self.cy + (.5 * self.sy - self.cy) * bass) / self.sy - .5)
        _over(U, *_split(cv2.warpAffine(gq, M, (self.Wq, self.Hq), flags=cv2.INTER_LINEAR)))
        cv2.resize(U, (W, H), dst=F, interpolation=cv2.INTER_LINEAR)

        # 2) .art-frame (scale(--art-scale), filter saturate()/brightness()) with the two .art layers
        self._art_frame(cur, prev, since, bass, mid, high, imp)

        # 3) halo <canvas>: source-over of its premultiplied 'lighter' contents (uint8)
        if job.halo > 0:
            rgb8, a8 = self._halo(tl, k, t, samples, low, mid, high, imp)
            R = F[self.hy0:self.hy1, self.hx0:self.hx1]
            cv2.multiply(R, cv2.cvtColor(cv2.bitwise_not(a8), cv2.COLOR_GRAY2RGB), dst=R,
                         scale=1 / 255, dtype=cv2.CV_32F)
            cv2.scaleAdd(rgb8.astype(np.float32), 1 / 255, R, dst=R)

        # 4) .flash (screen: d + s - d*s), 5) .vignette, 6) .grain (soft-light)
        fl = min(max(imp * .12 + high * .02, 0.0), .2)
        if fl > .002:
            R = F[self._flash_sl]
            S = self._flash * np.float32(fl)
            DS = cv2.multiply(R, S)
            cv2.add(R, S, dst=R)
            cv2.subtract(R, DS, dst=R)
        cv2.multiply(F, self._vk3, dst=F)
        if job.grain > 0:                                   # soft-light: d += g * (d - d^2)
            cv2.multiply(F, F, dst=T)
            cv2.subtract(F, T, dst=T)
            cv2.multiply(T, self._grain[int(t / .09) % 2], dst=T)
            cv2.add(F, T, dst=F)
        # .scanlines (overlay, white .08 x opacity .07) change a pixel by < 0.0028 - below one
        # 8-bit step - so they are intentionally not rendered.

        # 8) .brandbar
        if self._brand is not None:
            (rgb, al), (bx, by) = self._brand
            y0, x0 = max(0, by), max(0, bx)
            y1, x1 = min(H, by + al.shape[0]), min(W, bx + al.shape[1])
            _over(F[y0:y1, x0:x1], np.ascontiguousarray(rgb[y0 - by:y1 - by, x0 - bx:x1 - bx]),
                  np.ascontiguousarray(al[y0 - by:y1 - by, x0 - bx:x1 - bx]))
            self._meters(low, mid, high, imp, t)
        return cv2.convertScaleAbs(F, alpha=255.0)

    def _art_frame(self, cur, prev, since, bass, mid, high, imp):
        F, s = self.F, self.art_side * bass
        a0, b0 = self.cx - s / 2, self.cy - s / 2
        X0, Y0 = max(0, int(math.floor(a0))), max(0, int(math.floor(b0)))
        X1, Y1 = min(self.W, int(math.ceil(a0 + s))), min(self.H, int(math.ceil(b0 + s)))
        w, h = X1 - X0, Y1 - Y0
        region = F[Y0:Y1, X0:X1]
        cx_, cy_ = _coverage(X0, X1, a0, a0 + s), _coverage(Y0, Y1, b0, b0 + s)
        # the frame edges are sub-pixel: keep the outermost rows/cols to re-blend by coverage
        strips = [(np.s_[0:1, :], cy_[0]), (np.s_[h - 1:h, :], cy_[-1]),
                  (np.s_[:, 0:1], cx_[0]), (np.s_[:, w - 1:w], cx_[-1])]
        saved = [region[sl].copy() for sl, _ in strips]
        # CSS filter on the frame: saturate(1 + mid*.12) brightness(1 + high*.035 + impact*.035)
        cm = _sat_matrix(1 + mid * .12) * ((1 + high * .035 + imp * .035) / 255)
        e_art, z = _ease(since / self.art_fade), _ease_out(since / 5.2)   # .art 1.35 s / 5.2 s
        arts = [(cur, e_art, 1.045 - .045 * z)]
        if prev != cur and e_art < 1:
            arts.insert(0, (prev, 1 - e_art, 1 + .045 * z))
        for i, alpha, scale in arts:
            if alpha <= 1e-3:
                continue
            ref = self.art(i)
            n = ref.shape[0]
            k = self.art_side * scale * bass / n
            M = _affine(k, self.cx + (.5 - n / 2) * k - X0 - .5, k, self.cy + (.5 - n / 2) * k - Y0 - .5)
            A = cv2.warpAffine(ref, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            A = cv2.transform(A.astype(np.float32), cm)
            cv2.addWeighted(A, alpha, region, 1 - alpha, 0, dst=region)
        # .art-glass (screen): d + g - d*g
        g = self.glass(w, h)
        dg = cv2.multiply(region, g)
        cv2.add(region, g, dst=region)
        cv2.subtract(region, dg, dst=region)
        # inset 0 0 0 1px rgba(255,255,255,.12)
        e = max(1, int(round(self.u)))
        for sl in (np.s_[:e, :], np.s_[h - e:, :], np.s_[:, :e], np.s_[:, w - e:]):
            region[sl] += (1 - region[sl]) * .12
        # re-blend the partially covered outer rows/cols by their coverage
        for (sl, c), before in zip(strips, saved):
            if c < 1:
                region[sl] = before + (region[sl] - before) * c

    def _meters(self, low, mid, high, imp, t):
        """.meter: 12 bars rgba(210,235,255,.85), box-shadow 0 0 8px rgba(117,186,255,.65), opacity .78"""
        u = self.u
        pad = int(8 * u) + 2
        bw, gap, box_h = 2 * u, 3 * u, 22 * u
        x_right, top = self.W - 24 * u, 22 * u
        x_left = x_right - 12 * bw - 11 * gap
        X0, Y0 = max(0, int(x_left) - pad), max(0, int(top) - pad)
        X1, Y1 = min(self.W, int(math.ceil(x_right)) + pad), min(self.H, int(math.ceil(top + box_h)) + pad)
        bars = np.zeros((Y1 - Y0, X1 - X0), np.float32)
        x = x_left
        for i in range(12):
            q = i / 11
            e = low if q < .33 else mid if q < .72 else high
            mod = .6 + .4 * math.sin(t * 6.2 + i * .93)
            hgt = (3 + 18 * min(max(e * 1.3 * mod + imp * .22, 0), 1)) * u
            bars += np.outer(_coverage(Y0, Y1, top + box_h - hgt, top + box_h), _coverage(X0, X1, x, x + bw))
            x += bw + gap
        glow = cv2.GaussianBlur(bars, (0, 0), 4 * u)
        R = self.F[Y0:Y1, X0:X1]
        for m, col, a in ((glow, (117, 186, 255), .65 * .78), (bars, (210, 235, 255), .85 * .78)):
            aa = np.clip(m * a, 0, 1)[..., None]
            R *= 1 - aa
            R += aa * (np.array(col, np.float32) / 255)

    # ---- halo: <canvas id="halo">, globalCompositeOperation 'lighter' --------------------------
    def _halo(self, tl, k, t, samples, low, mid, high, imp):
        """Draw the halo canvas for its bounding box. Returns premultiplied (rgb, alpha) uint8."""
        u, R, HI = self.u, self.baseR, self.job.halo
        ox, oy = self.cx - self.hx0, self.cy - self.hy0     # halo centre inside the box
        acc, lay = self._acc, self._lay
        acc[:] = 0
        for sh in self._shadow:
            sh[:] = 0
        shadow_sigma = [0.0, 0.0]                            # [long (rings/wave/line), short (particles)]

        def fx(v):
            return int(round(v * ONE))

        def ink(rgb, a, width):
            """premultiplied RGBA colour + integer thickness; fractional widths fold into alpha."""
            th = max(1, int(round(width)))
            a = min(max(a * width / th, 0.0), 1.0) * 255
            return (rgb[0] * a, rgb[1] * a, rgb[2] * a, a), th

        def flush():
            cv2.add(acc, lay, dst=acc)                       # 'lighter' = saturating add
            lay[:] = 0

        def shadow(draw, rgb, alpha, width, blur):
            """canvas shadow: shadowColor alpha x drawn alpha, blurred by shadowBlur (sigma = blur/2)."""
            if blur > 0 and alpha > 0:
                c, th = ink(rgb, alpha, width / Q)
                draw(self._shadow[0], c, th, 1.0 / Q, (ox / Q, oy / Q))
                shadow_sigma[0] = max(shadow_sigma[0], blur / 2)

        # drawRings
        for i in range(7):
            p = i / 6
            r = R * (.74 + p * .66) * (1 + low * .04 + imp * .018)
            r += math.sin(t * (.55 + p * .35) + i * 1.4) * (2 + mid * 8) * u
            sq = .70 + math.sin(t * .31 + i) * .05
            rot = math.degrees(t * .06 + i * .07)
            a = (.055 + (1 - p) * .09 + high * .08 + imp * .04) * HI
            width = (.6 + (1 - p) * 1.2 + low * 1.8) * HI * u

            def ring(img, c, th, sc, o, r=r, sq=sq, rot=rot):
                cv2.ellipse(img, (fx(o[0]), fx(o[1])), (fx(r * sc), fx(r * sq * sc)), rot, 0, 360,
                            c, th, cv2.LINE_AA, SH)
            c, th = ink(_hsl(202 + p * 54 + high * 22, 1, (66 + p * 20) / 100), a, width)
            ring(lay, c, th, 1.0, (ox, oy))
            shadow(ring, _hsl(205 + p * 32, 1, .72), .55 * a, width, (9 + low * 18 + (1 - p) * 12) * HI * u)
        flush()

        # drawSpokes (no shadow)
        i = np.arange(72)
        ang = i / 72 * 2 * np.pi + t * .025
        harmonic = .5 + .5 * np.sin(i * 1.731 + t * 1.3)
        amp = (6 + high * 46 + mid * 18) * harmonic * HI * u
        r0 = R * (1.01 + .025 * np.sin(i * .8 + t))
        r1 = r0 + amp + imp * 18 * ((i % 3) / 2) * u
        ca, sa = np.cos(ang), np.sin(ang) * .78
        pts = np.round(np.stack([ox + ca * r0, oy + sa * r0, ox + ca * r1, oy + sa * r1], 1) * ONE).astype(int)
        width = (.55 + high * .8) * u
        th = max(1, int(round(width)))
        a255 = min(1.0, (.035 + high * .16 + imp * .06) * HI * width / th) * 255
        cols = np.concatenate([_hsl_v(190 + high * 70 + (i % 9), .78) * a255,
                               np.full((72, 1), a255)], 1).tolist()
        for (x0, y0, x1, y1), c in zip(pts.tolist(), cols):
            cv2.line(lay, (x0, y0), (x1, y1), c, th, cv2.LINE_AA, SH)
        flush()

        # drawWaveHalo: the time-domain waveform bent around the ring, 3 additive passes
        e = int(tl.ends[k])
        wave = samples[max(0, e - FFT):e]
        if wave.size < FFT:
            wave = np.pad(wave, (FFT - wave.size, 0))
        steps = 380
        s = np.arange(steps + 1)
        smp = wave[np.floor(s / steps * (FFT - 1)).astype(int)].astype(np.float64)
        ang = s / steps * 2 * np.pi
        for ps in range(3):
            r = (R * (1 + ps * .018) + smp * (9 + mid * 22 + ps * 3) * HI * u
                 + np.sin(ang * 6 + t * 1.8 + ps) * (1.3 + high * 4.5) * u)
            poly = np.stack([np.cos(ang) * r, np.sin(ang) * r * .78], 1)
            a = (.15 + high * .28 + imp * .12) * HI
            width = (1 + ps * 1.1 + low * 2.8) * HI * u

            def wav(img, c, th, sc, o, poly=poly):
                pts = np.round((poly * sc + o) * ONE).astype(np.int32)
                cv2.polylines(img, [pts], False, c, th, cv2.LINE_AA, SH)
            c, th = ink(_hsl(205 + ps * 16 + high * 30, 1, (74 + ps * 5) / 100), a, width)
            wav(lay, c, th, 1.0, (ox, oy))
            flush()
            shadow(wav, _hsl(210 + ps * 18, 1, .75), .75 * a, width, (10 + ps * 10 + high * 22) * HI * u)

        # drawParticles (colours vectorised; sub-pixel dots fold their area into alpha)
        P = tl.particles
        b = P["bright"].astype(np.float64)
        pulse = np.sin(t * (1.2 + b) + P["phase"]) * (.015 + high * .035)
        pr = R * (P["radius"] + pulse + imp * .018 * P["drift"])
        pa = tl.angles[k]
        xs, ys = ox + np.cos(pa) * pr, oy + np.sin(pa) * pr * .79
        rad = np.maximum(.3, P["size"] * (.65 + high * 1.9 + imp * .8) * HI) * u
        hue = 196 + 48 * b + high * 42
        a = np.minimum((.16 + b * .36 + high * .18) * HI, 1)
        rr, rq = np.maximum(rad, .5), np.maximum(rad / Q, .5)
        af = a * np.minimum(rad / rr, 1) ** 2 * 255
        aq = .65 * a * np.minimum(rad / Q / rq, 1) ** 2 * 255
        cf = np.concatenate([_hsl_v(hue, (72 + b * 22) / 100) * af[:, None], af[:, None]], 1).tolist()
        cq = np.concatenate([_hsl_v(hue, .78) * aq[:, None], aq[:, None]], 1).tolist()
        C = np.round(np.stack([xs, ys, rr, xs / Q, ys / Q, rq], 1) * ONE).astype(int).tolist()
        shq = self._shadow[1]
        for (x, y, r_, xq, yq, r_q), c, c2 in zip(C, cf, cq):
            cv2.circle(lay, (x, y), r_, c, -1, cv2.LINE_AA, SH)
            cv2.circle(shq, (xq, yq), r_q, c2, -1, cv2.LINE_AA, SH)
        shadow_sigma[1] = (5 + high * 16) * HI * u / 2
        flush()

        # drawCore's impact line
        if imp > .15:
            L = R * (.76 + imp * .13)
            width = (1 + imp * 1.7) * u
            a = min(1.0, imp * .21 * HI)

            def line(img, c, th, sc, o, L=L):
                cv2.line(img, (fx(o[0]), fx(o[1] - L * sc)), (fx(o[0]), fx(o[1] + L * sc)),
                         c, th, cv2.LINE_AA, SH)
            c, th = ink((211 / 255, 240 / 255, 1.0), a, width)
            line(lay, c, th, 1.0, (ox, oy))
            flush()
            shadow(line, (124 / 255, 192 / 255, 1.0), .9 * a, width, (18 + imp * 36) * u)

        # drawCore's radial gradient: rgba(235,249,255,a0) 0 -> rgba(111,190,255,a1) .2 -> 0 at r*1.7
        rc = R * (.22 + low * .035 + imp * .02) * 1.7
        cy0, cy1 = max(0, int(oy - rc)), min(self.hh, int(oy + rc) + 2)
        cx0, cx1 = max(0, int(ox - rc)), min(self.hw, int(ox + rc) + 2)
        dn = np.hypot((np.arange(cx0, cx1) + .5 - ox)[None, :], (np.arange(cy0, cy1) + .5 - oy)[:, None]) / rc
        dn = dn[..., None]
        c0 = np.array([235 / 255, 249 / 255, 1, 1]) * min(1.0, (.12 + high * .10 + imp * .08) * HI)
        c1 = np.array([111 / 255, 190 / 255, 1, 1]) * min(1.0, (.07 + low * .10) * HI)
        w0 = np.clip(dn / .2, 0, 1)
        core = np.where(dn < .2, c0 * (1 - w0) + c1 * w0, c1 * np.clip((1 - dn) / .8, 0, 1))
        sub = acc[cy0:cy1, cx0:cx1]
        cv2.add(sub, (core * 255 + .5).astype(np.uint8), dst=sub)

        # shadows: blur each 1/4-res layer once, sum, upscale once, add
        blurred = [cv2.GaussianBlur(sh, (0, 0), sig / Q)
                   for sh, sig in zip(self._shadow, shadow_sigma) if sig > 0]
        if blurred:
            q = blurred[0] if len(blurred) == 1 else cv2.add(blurred[0], blurred[1])
            up = cv2.resize(q, (self.qw * Q, self.qh * Q), interpolation=cv2.INTER_LINEAR)
            cv2.add(acc, up[:self.hh, :self.hw], dst=acc)
        return cv2.cvtColor(acc, cv2.COLOR_RGBA2RGB), cv2.extractChannel(acc, 3)
