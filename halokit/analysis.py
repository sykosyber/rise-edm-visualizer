"""Audio analysis + visualizer state simulation.

Mirrors visualizer.js: an AnalyserNode (fftSize 2048, Blackman window,
smoothingTimeConstant 0.70, dB range -92..-20), bandEnergy() for three bands,
detectBeat(), the envelope followers and image-change rules of animate().

The simulation always runs at SIM_HZ (the browser's 60 Hz rAF rate), so the
output frame rate never changes where beats land or how envelopes move. Every
random choice comes from generators seeded with job.seed.
"""
import math
from dataclasses import dataclass

import numpy as np

SIM_HZ = 60
FFT = 2048
BINS = FFT // 2
TAU = 0.70
MIN_DB, MAX_DB = -92.0, -20.0
BANDS = ((36, 180), (180, 2200), (2200, 12000))
N_PARTICLES = 210
_BLACKMAN = (0.42 - 0.5 * np.cos(2 * np.pi * np.arange(FFT) / FFT)
             + 0.08 * np.cos(4 * np.pi * np.arange(FFT) / FFT))


@dataclass
class Timeline:
    n: int                 # simulation steps (SIM_HZ)
    ends: np.ndarray       # sample index ending each step's analysis window
    low: np.ndarray        # smoothed envelopes, as drawn
    mid: np.ndarray
    high: np.ndarray
    impact: np.ndarray
    visual: np.ndarray     # current image index
    prev_visual: np.ndarray
    fade: np.ndarray       # 0..1 crossfade progress into `visual`
    since: np.ndarray      # seconds since `visual` appeared
    energy: np.ndarray     # composite energy (for picking the loudest moment)
    beats: int
    changes: int
    angles: np.ndarray     # (n, N_PARTICLES) particle orbit angles
    particles: dict        # static per-particle parameters

    def step_at(self, t):
        return min(self.n - 1, max(0, int(math.floor(t * SIM_HZ + 1e-9))))


def band_energies(samples, sr):
    """Per-step (low, mid, high) exactly as bandEnergy() would read them."""
    n = int(len(samples) * SIM_HZ // sr)
    ends = np.clip(np.round(np.arange(n) * sr / SIM_HZ).astype(np.int64) + FFT // 2,
                   FFT, len(samples))
    padded = np.concatenate([np.zeros(FFT, np.float32), samples])
    mags = np.empty((n, BINS), np.float64)
    idx = np.arange(FFT)
    for a in range(0, n, 1024):                      # batched FFT, bounded memory
        e = ends[a:a + 1024] + FFT                   # index into padded
        win = padded[(e - FFT)[:, None] + idx].astype(np.float64) * _BLACKMAN
        mags[a:a + 1024] = np.abs(np.fft.rfft(win, axis=1)[:, :BINS]) / FFT
    # AnalyserNode smoothing: s[k] = tau*s[k-1] + (1-tau)*x[k]
    from scipy.signal import lfilter      # main process only; render workers never import scipy
    smooth = lfilter([1 - TAU], [1, -TAU], mags, axis=0)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(smooth)
    byte = np.clip(np.floor(255.0 / (MAX_DB - MIN_DB) * (db - MIN_DB)), 0, 255) / 255.0
    nyq = sr / 2
    out = []
    for lo_hz, hi_hz in BANDS:
        lo = max(0, int(lo_hz / nyq * BINS))
        hi = min(BINS - 1, math.ceil(hi_hz / nyq * BINS))
        out.append(np.sqrt(np.mean(byte[:, lo:hi + 1] ** 2, axis=1)))
    return ends, out[0], out[1], out[2]


def simulate(samples, sr, job, n_images):
    ends, raw_low, raw_mid, raw_high = band_energies(samples, sr)
    n = len(ends)
    dt = 1.0 / SIM_HZ
    rng_vis = np.random.default_rng([job.seed, 1])
    rng_par = np.random.default_rng([job.seed, 2])

    k_low, k_mid, k_high = (1 - math.exp(-dt * 9), 1 - math.exp(-dt * 7), 1 - math.exp(-dt * 11))
    decay = math.exp(-dt * 4.6)

    low_e = mid_e = high_e = impact = 0.0
    hist = np.zeros(56)
    cur = 0
    last_beat = -1e9
    beats = changes = 0
    visual = 0 if job.image_order == "sequence" or n_images == 1 else int(rng_vis.integers(n_images))
    prev = visual
    fade, since = 1.0, 0.0   # the app calls changeVisual() on load: first image zooms in too
    last_img = 0.0
    next_fallback = float(rng_vis.uniform(job.fallback_min, job.fallback_max)) * 1000

    arr = {k: np.empty(n, np.float32) for k in ("low", "mid", "high", "impact", "fade", "since")}
    vis = np.empty(n, np.int32)
    pvis = np.empty(n, np.int32)

    def change(now):
        nonlocal visual, prev, fade, since, last_img, next_fallback, impact, changes
        if n_images < 2:
            return
        prev = visual
        if job.image_order == "sequence":
            visual = (visual + 1) % n_images
        else:
            nxt = visual
            while nxt == visual:
                nxt = int(rng_vis.integers(n_images))
            visual = nxt
        fade, since, last_img = 0.0, 0.0, now
        next_fallback = float(rng_vis.uniform(job.fallback_min, job.fallback_max)) * 1000
        impact = max(impact, 0.72)
        changes += 1

    min_hold_ms = job.min_hold * 1000
    for i in range(n):
        now = i * dt * 1000
        low = float(raw_low[i])
        # detectBeat()
        hist[cur] = low
        cur = (cur + 1) % 56
        avg = hist.mean()
        sd = math.sqrt(((hist - avg) ** 2).mean())
        if now - last_beat > 225 and low > max(0.16, avg + sd * 1.35) and low > 0.22:
            last_beat = now
            beats += 1
            impact = min(impact + 0.65, 1.6)
            if job.beats_per_change > 0 and beats % job.beats_per_change == 0 \
                    and now - last_img > min_hold_ms:
                change(now)
        low_e += (low - low_e) * k_low
        mid_e += (float(raw_mid[i]) - mid_e) * k_mid
        high_e += (float(raw_high[i]) - high_e) * k_high
        impact *= decay
        if now - last_img > next_fallback:
            change(now)
        if fade < 1.0:
            fade = min(1.0, fade + dt / max(job.crossfade, 1e-3))
        since += dt
        arr["low"][i], arr["mid"][i], arr["high"][i] = low_e, mid_e, high_e
        arr["impact"][i], arr["fade"][i], arr["since"][i] = impact, fade, since
        vis[i], pvis[i] = visual, prev

    # particles (seedParticles) and their orbits: angle += speed*dt*(0.65 + mid*1.8)
    idx = np.arange(N_PARTICLES)
    particles = {
        "angle0": (idx / N_PARTICLES * 2 * np.pi + rng_par.uniform(-.05, .05, N_PARTICLES)),
        "radius": rng_par.uniform(.88, 1.42, N_PARTICLES),
        "speed": rng_par.uniform(-.12, .12, N_PARTICLES),
        "drift": rng_par.uniform(-.2, .2, N_PARTICLES),
        "size": rng_par.uniform(.45, 1.9, N_PARTICLES),
        "phase": rng_par.uniform(0, 2 * np.pi, N_PARTICLES),
        "bright": rng_par.uniform(.25, 1, N_PARTICLES),
    }
    particles = {k: v.astype(np.float32) for k, v in particles.items()}
    travel = np.cumsum(dt * (0.65 + arr["mid"].astype(np.float64) * 1.8))
    angles = (particles["angle0"][None, :] + travel[:, None] * particles["speed"][None, :]).astype(np.float32)

    energy = (arr["low"] * .52 + arr["mid"] * .30 + arr["high"] * .18 + arr["impact"] * .25)
    return Timeline(n=n, ends=ends, low=arr["low"], mid=arr["mid"], high=arr["high"],
                    impact=arr["impact"], visual=vis, prev_visual=pvis, fade=arr["fade"],
                    since=arr["since"], energy=energy.astype(np.float32), beats=beats,
                    changes=changes, angles=angles, particles=particles)


def loudest_time(tl, start=0.0, end=None):
    """Time (s) of peak composite energy, skipping the ramp-in (first 8 s / 25%)."""
    a = tl.step_at(start)
    b = tl.n if end is None else tl.step_at(end) + 1
    skip = min(int(8 * SIM_HZ), (b - a) // 4)
    seg = tl.energy[a + skip:b]
    if seg.size == 0:
        return start
    return (a + skip + int(np.argmax(seg))) / SIM_HZ
