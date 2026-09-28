# RISE RADIO — Reactive EDM Visualizer

A browser-based, full-screen audiovisual visualizer built around the supplied RISE artwork and three supplied MP3s.

## What it does

- Plays all 3 uploaded audio files as a continuous playlist.
- Shuffles through the supplied artwork (40 unique images) with long crossfades.
- Uses the Web Audio API in real time — no pre-rendered animation.
- Separates bass / mid / treble energy and maps them to different visual behaviors.
- Adaptive beat detection drives impact flashes, ring expansion, and image changes.
- Halo system combines waveform rings, concentric luminous ellipses, radial spokes, orbital particles, bloom, CRT grain, and subtle scanlines.
- Auto-image changes happen on musical beat groups when possible, with a time-based fallback so quiet passages still evolve.
- Fullscreen, seeking, manual visual shuffle, halo intensity, keyboard controls, and mobile layout are included.

## Run locally

Browsers require a user gesture before audio playback. Serve the directory rather than double-clicking the HTML file:

```bash
cd rise-edm-visualizer
python -m http.server 8080
```

Then open `http://localhost:8080` and click **ENTER RISE RADIO**.

## Controls

- Space: play / pause
- S: shuffle visual
- F: fullscreen
- Cmd/Ctrl + Left/Right: previous / next track
- On-screen controls: seek, tracks, visual shuffle, auto visual, halo intensity, fullscreen

## Audio-specific tuning

The supplied tracks include material around ~129 BPM and one slower file around ~99 BPM. The app therefore does not hard-code one BPM; it uses a rolling bass-energy threshold and refractory interval so the halo can adapt to each track in real time.

## Notes about uploaded assets

- Playlist: Living Frames (`living-frames.mp3`), RISE // B (`rise-b.mp3`), RISE // C (`rise-c.mp3`).
- Living Frames replaced the original `rise-a.mp3`, which was byte-identical to `rise-b.mp3`.
- `visual-09` … `visual-41` are the SyberLabs RISE poster series (33 images), so the carousel shuffles 40 unique images.
- One supplied image is byte-identical to another. It remains in `assets/images`, but the live shuffle excludes the duplicate so visual transitions do not land on the same frame twice.

---

# halokit — offline renders: any images, any music

`halokit/` renders the RISE RADIO look to video or stills from the command line or a local web UI. It is an offline port of `visualizer.js` + `styles.css`: the same AnalyserNode analysis, beat detection, envelopes, image-change rules, halo layers and CSS compositing (source-over, screen, soft-light, filter matrices), computed from the actual audio.

## Setup (once)

```bash
cd rise-edm-visualizer
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock      # macOS/Linux: .venv/bin/python
```

`requirements.lock` pins the exact versions renders are verified against. ffmpeg comes bundled via `imageio-ffmpeg`; set `HALOKIT_FFMPEG` to use another binary. The examples below write `python` for the venv interpreter.

## Command line

```bash
# a video from one track and any mix of files, folders and globs
python -m halokit render song.mp3 cover.png photos/ "art/*.jpg" -o out.mp4

# common options
python -m halokit render song.mp3 art/ -o out.mp4 --size 1080x1920 --fps 60 --seed 7 \
    --halo 1.35 --order sequence --start 30 --duration 45 --title "My Song" --no-brand

# a single frame: at a time, or the loudest moment (default)
python -m halokit still song.mp3 art/ -o frame.png --at 62.5

# render saved jobs (each as its own kind) - e.g. the whole canonical RISE set
python -m halokit run jobs/*.json

# prove a render reproduces (optionally only the first N frames)
python -m halokit verify renders/rise-c.job.json --frames 300
```

`python -m halokit render --help` lists every option. `--dry-run --save-job job.json` writes the resolved job without rendering.

## Web UI

```bash
python -m halokit ui --open            # http://127.0.0.1:8765, workspace in ./halokit-workspace
```

Drop in a track and images (or import a file/folder by path), adjust the look, preview any moment (or jump to the loudest), then render a video or still. Jobs show live progress, play in the page, and link their manifest; each one has a copyable CLI command. The UI runs the same `python -m halokit` code path as the CLI, so everything it makes is reproducible from the command line.

## Reproducibility

Every render writes `<output>.job.json` next to the output:

- `job` — the fully resolved settings, including the exact de-duplicated image list (paths relative to the manifest),
- `inputs` — SHA-256 of the audio, every image and the brand font,
- `env` — halokit/engine, Python, numpy, scipy, Pillow, OpenCV and ffmpeg versions,
- `result` — frame count, a SHA-256 over all raw frames with checkpoints every 150 frames, and the output file's SHA-256.

`render --config that.job.json` rebuilds it; `verify` re-renders and compares frame digests (and flags changed or missing inputs). Guarantees, all covered by `tests/`:

- all randomness derives from `--seed`; each frame is a pure function of (job, inputs, time), so output is **bit-identical regardless of worker count**;
- analysis runs at a fixed 60 Hz (the browser's rAF rate), so changing `--fps` never moves beats or image changes;
- ffmpeg runs in bit-exact mode, so the same machine produces a byte-identical MP4.

Limits: pixel-exactness is guaranteed for the same OS, CPU family and locked dependency versions. Other platforms render the same look, but low-level SIMD/codec differences can change the last bits, and `verify` will report that. Fonts are resolved per machine (`--font` pins one), and the manifest records the font hash.

## Speed

Frames render in parallel worker processes (`--workers`, default cores−1, max 8) after a single fast analysis pass. The smooth layers under the artwork (page gradient, 54 px-blurred backdrop, box-shadow glow) are composited at quarter resolution. The halo is drawn with anti-aliased sub-pixel OpenCV primitives inside its bounding box, and everything static is pre-baked per worker. On a 4-core i7-1165G7 laptop, 1080p30 renders at ~10 fps including H.264 encoding: a 3½-minute track takes ~10 minutes at 1080p. The first engine needed ~60 minutes at 720p.

Output is written to `name.partial.mp4` and renamed only on success, so a file that exists is complete.

## Tests

```bash
python -m unittest discover -s tests
```

## Fidelity notes

The CSS scanlines (white at .08 × opacity .07, overlay) change a pixel by less than one 8-bit step, so they are intentionally skipped. Canvas `shadowBlur` is rendered as two blurred quarter-resolution shadow layers (long and short blur) instead of one blur per stroke.
