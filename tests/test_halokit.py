"""halokit tests: python -m unittest discover -s tests   (or: python -m pytest tests)

Everything is synthetic (a generated kick/hat loop and generated images), tiny,
and fast, so the reproducibility guarantees are checked on every run.
"""
import json
import os
import shutil
import tempfile
import unittest
import wave

import numpy as np
from PIL import Image, ImageDraw

from halokit import media
from halokit.analysis import SIM_HZ, simulate
from halokit.config import Job, manifest_path_for
from halokit.render import FrameHasher, iter_frames, prepare, render, verify

SR = media.SAMPLE_RATE


def write_loop(path, seconds=6.0, bpm=128):
    """Four-on-the-floor kick + offbeat hats, as 16-bit mono WAV."""
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    beat = 60 / bpm
    for k in np.arange(0, seconds, beat):
        i = int(k * SR)
        n = min(len(t) - i, int(.25 * SR))
        tt = np.arange(n) / SR
        x[i:i + n] += np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
        j = int((k + beat / 2) * SR)
        if j < len(t):
            m = min(len(t) - j, int(.04 * SR))
            x[j:j + m] += np.random.default_rng(int(k * 1000)).normal(0, .2, m) * np.exp(-np.arange(m) / SR * 90)
    x = np.clip(x * .7, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def write_images(folder):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, (size, col) in enumerate([((300, 300), (40, 90, 220)), ((480, 270), (220, 60, 120)),
                                     ((200, 360), (60, 200, 140))]):
        im = Image.new("RGB", size, col)
        d = ImageDraw.Draw(im)
        d.ellipse([size[0] * .2, size[1] * .2, size[0] * .8, size[1] * .8], fill=(250, 250, 255))
        d.text((10, 10), f"img {i}", fill=(0, 0, 0))
        p = os.path.join(folder, f"img{i}.png")
        im.save(p)
        paths.append(p)
    shutil.copy(paths[0], os.path.join(folder, "zz_duplicate.png"))   # byte-identical duplicate
    return paths


class HalokitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="halokit-test-")
        cls.audio = os.path.join(cls.tmp, "loop.wav")
        write_loop(cls.audio)
        cls.imgdir = os.path.join(cls.tmp, "imgs")
        cls.images = write_images(cls.imgdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def job(self, **kw):
        base = dict(audio=self.audio, images=[self.imgdir], output=os.path.join(self.tmp, "out.mp4"),
                    width=256, height=144, fps=24, seed=3, start=1.0, duration=1.0,
                    min_hold=.5, fallback_min=1.0, fallback_max=1.5, beats_per_change=2)
        base.update(kw)
        return Job(**base)

    # ---- inputs ------------------------------------------------------------------------
    def test_discover_dedupes_and_orders(self):
        paths, dupes = media.discover_images([self.imgdir])
        self.assertEqual([os.path.basename(p) for p in paths], ["img0.png", "img1.png", "img2.png"])
        self.assertEqual(len(dupes), 1)
        paths, _ = media.discover_images([os.path.join(self.imgdir, "img[12].png")])
        self.assertEqual(len(paths), 2)

    def test_cover_fit_any_aspect(self):
        im = media.cover(media.load_rgb(self.images[2]), 100, 100)
        self.assertEqual(im.size, (100, 100))

    def test_job_roundtrip_relative_paths(self):
        j = self.job()
        path = os.path.join(self.tmp, "sub", "job.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        j.save(path)
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.assertFalse(os.path.isabs(raw["audio"]))
        self.assertEqual(Job.load(path).audio, os.path.normpath(self.audio))
        with self.assertRaises(ValueError):
            Job.from_dict({"audio": "a", "bogus": 1})

    # ---- analysis -------------------------------------------------------------------------
    def test_analysis_detects_beats_and_is_deterministic(self):
        samples = media.decode_audio(self.audio)
        a = simulate(samples, SR, self.job(), 3)
        b = simulate(samples, SR, self.job(), 3)
        self.assertGreater(a.beats, 5)
        self.assertGreater(a.changes, 0)
        self.assertEqual(a.n, int(len(samples) * SIM_HZ // SR))
        for f in ("low", "impact", "visual", "angles"):
            np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
        c = simulate(samples, SR, self.job(seed=4), 3)
        self.assertFalse(np.array_equal(a.visual, c.visual) and np.array_equal(a.angles, c.angles))

    # ---- rendering -------------------------------------------------------------------------
    def digest(self, job, workers):
        h = FrameHasher()
        for f in iter_frames(prepare(job), workers=workers):
            h.update(f)
        return h.final()

    def test_frames_independent_of_worker_count(self):
        self.assertEqual(self.digest(self.job(), 1), self.digest(self.job(), 3))

    def test_render_verify_and_reproduce(self):
        out = os.path.join(self.tmp, "v", "clip.mp4")
        m1 = render(self.job(output=out), workers=2)
        self.assertTrue(os.path.isfile(out))
        self.assertFalse(os.path.exists(out.replace(".mp4", ".partial.mp4")))
        mpath = manifest_path_for(out)
        ok, lines = verify(mpath, workers=2)
        self.assertTrue(ok, lines)
        # re-render from the manifest alone -> bit-identical frames and file
        again = os.path.join(self.tmp, "v", "clip-again.mp4")
        job = Job.load(mpath)
        job.output = again
        m2 = render(job, workers=1)
        self.assertEqual(m1["result"]["frame_digest"], m2["result"]["frame_digest"])
        self.assertEqual(m1["result"]["output_sha256"], m2["result"]["output_sha256"])

    def test_verify_detects_changed_input(self):
        folder = os.path.join(self.tmp, "mut")
        shutil.copytree(self.imgdir, folder)
        out = os.path.join(self.tmp, "mut-out.mp4")
        render(self.job(images=[folder], output=out, duration=.25), workers=1)
        Image.new("RGB", (50, 50), (1, 2, 3)).save(os.path.join(folder, "img1.png"))
        ok, lines = verify(manifest_path_for(out))
        self.assertFalse(ok)
        self.assertTrue(any("changed" in line for line in lines))

    def test_still_single_image_sequence(self):
        out = os.path.join(self.tmp, "still.png")
        m = render(self.job(kind="still", output=out, images=[self.images[1]], image_order="sequence",
                            show_brand=False, still_at=2.0))
        with Image.open(out) as im:
            self.assertEqual(im.size, (256, 144))
        self.assertEqual(m["result"]["frames"], 1)
        self.assertEqual(m["result"]["image_changes"], 0)


if __name__ == "__main__":
    unittest.main()
