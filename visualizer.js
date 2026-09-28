(() => {
  'use strict';

  const TRACKS = [
    { title: 'Living Frames', src: 'assets/audio/living-frames.mp3' },
    { title: 'RISE // B', src: 'assets/audio/rise-b.mp3' },
    { title: 'RISE // C', src: 'assets/audio/rise-c.mp3' }
  ];

  // visual-08 is a byte-identical repeat of visual-02. Keep it in the package,
  // but remove it from the live shuffle so every transition changes the frame.
  // visual-09 .. visual-41 are the SyberLabs RISE poster series.
  const VISUALS = [
    'assets/images/visual-01.png',
    'assets/images/visual-02.png',
    'assets/images/visual-03.png',
    'assets/images/visual-04.png',
    'assets/images/visual-05.png',
    'assets/images/visual-06.png',
    'assets/images/visual-07.png',
    'assets/images/visual-09.png',
    'assets/images/visual-10.png',
    'assets/images/visual-11.png',
    'assets/images/visual-12.png',
    'assets/images/visual-13.png',
    'assets/images/visual-14.png',
    'assets/images/visual-15.png',
    'assets/images/visual-16.png',
    'assets/images/visual-17.png',
    'assets/images/visual-18.png',
    'assets/images/visual-19.png',
    'assets/images/visual-20.png',
    'assets/images/visual-21.png',
    'assets/images/visual-22.png',
    'assets/images/visual-23.png',
    'assets/images/visual-24.png',
    'assets/images/visual-25.png',
    'assets/images/visual-26.png',
    'assets/images/visual-27.png',
    'assets/images/visual-28.png',
    'assets/images/visual-29.png',
    'assets/images/visual-30.png',
    'assets/images/visual-31.png',
    'assets/images/visual-32.png',
    'assets/images/visual-33.png',
    'assets/images/visual-34.png',
    'assets/images/visual-35.png',
    'assets/images/visual-36.png',
    'assets/images/visual-37.png',
    'assets/images/visual-38.png',
    'assets/images/visual-39.png',
    'assets/images/visual-40.png',
    'assets/images/visual-41.png'
  ];

  const $ = (sel) => document.querySelector(sel);
  const app = $('#app');
  const audio = $('#audio');
  const canvas = $('#halo');
  const ctx = canvas.getContext('2d', { alpha: true });
  const artFrame = $('#artFrame');
  const artEls = [$('#artA'), $('#artB')];
  const backdrops = [$('#backdropA'), $('#backdropB')];
  const flash = $('#flash');
  const meters = [...document.querySelectorAll('#meter i')];

  const startButton = $('#startButton');
  const playPause = $('#playPause');
  const prevTrack = $('#prevTrack');
  const nextTrack = $('#nextTrack');
  const shuffleImage = $('#shuffleImage');
  const autoVisual = $('#autoVisual');
  const intensityButton = $('#intensity');
  const fullscreenButton = $('#fullscreen');
  const seek = $('#seek');
  const currentTimeEl = $('#currentTime');
  const durationEl = $('#duration');
  const trackIndexEl = $('#trackIndex');
  const trackTitleEl = $('#trackTitle');
  const energyLabel = $('#energyLabel');

  let audioCtx = null;
  let analyser = null;
  let source = null;
  let freq = null;
  let wave = null;

  let trackIndex = 0;
  let visualIndex = -1;
  let activeLayer = 0;
  let autoVisualEnabled = true;
  let haloIntensity = 1;
  let started = false;
  let raf = 0;
  let lastFrame = performance.now();
  let lastBeatAt = 0;
  let lastImageAt = 0;
  let nextFallbackImageAt = 16000;
  let beatCount = 0;
  let impact = 0;
  let lowEnvelope = 0;
  let midEnvelope = 0;
  let highEnvelope = 0;
  let energyHistory = new Float32Array(56);
  let historyCursor = 0;
  let particles = [];
  let width = 0;
  let height = 0;
  let dpr = 1;

  const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const random = (a, b) => a + Math.random() * (b - a);

  function formatTime(sec) {
    if (!Number.isFinite(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }

  function loadTrack(index, autoplay = false) {
    trackIndex = (index + TRACKS.length) % TRACKS.length;
    const track = TRACKS[trackIndex];
    audio.src = track.src;
    audio.load();
    trackIndexEl.textContent = `${String(trackIndex + 1).padStart(2, '0')} / ${String(TRACKS.length).padStart(2, '0')}`;
    trackTitleEl.textContent = track.title;
    document.title = `${track.title} — RISE RADIO`;
    if (autoplay) audio.play().catch(() => {});
  }

  function shuffledVisualIndex() {
    if (VISUALS.length < 2) return 0;
    let next;
    do next = Math.floor(Math.random() * VISUALS.length);
    while (next === visualIndex);
    return next;
  }

  function changeVisual(forceIndex = null) {
    const next = forceIndex ?? shuffledVisualIndex();
    visualIndex = next;
    activeLayer = 1 - activeLayer;

    const enteringArt = artEls[activeLayer];
    const leavingArt = artEls[1 - activeLayer];
    const enteringBackdrop = backdrops[activeLayer];
    const leavingBackdrop = backdrops[1 - activeLayer];
    const src = VISUALS[next];

    enteringArt.src = src;
    enteringBackdrop.src = src;

    requestAnimationFrame(() => {
      enteringArt.classList.add('active');
      enteringBackdrop.classList.add('active');
      leavingArt.classList.remove('active');
      leavingBackdrop.classList.remove('active');
    });

    lastImageAt = performance.now();
    nextFallbackImageAt = random(15000, 24000);
    impact = Math.max(impact, .72);
  }

  function initAudioGraph() {
    if (audioCtx) return;
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.70;
    analyser.minDecibels = -92;
    analyser.maxDecibels = -20;
    source = audioCtx.createMediaElementSource(audio);
    source.connect(analyser);
    analyser.connect(audioCtx.destination);
    freq = new Uint8Array(analyser.frequencyBinCount);
    wave = new Uint8Array(analyser.fftSize);
  }

  async function begin() {
    initAudioGraph();
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    if (!started) {
      started = true;
      app.dataset.state = 'paused';
      if (!audio.src) loadTrack(trackIndex, false);
      if (visualIndex < 0) changeVisual(Math.floor(Math.random() * VISUALS.length));
    }
    try {
      await audio.play();
    } catch (err) {
      console.warn('Playback was blocked by the browser:', err);
    }
  }

  function togglePlay() {
    if (!started) return begin();
    if (audio.paused) begin(); else audio.pause();
  }

  function bandEnergy(loHz, hiHz) {
    if (!analyser || !freq) return 0;
    const nyquist = audioCtx.sampleRate / 2;
    const lo = Math.max(0, Math.floor((loHz / nyquist) * freq.length));
    const hi = Math.min(freq.length - 1, Math.ceil((hiHz / nyquist) * freq.length));
    let sum = 0;
    let count = 0;
    for (let i = lo; i <= hi; i++) {
      const x = freq[i] / 255;
      sum += x * x;
      count++;
    }
    return count ? Math.sqrt(sum / count) : 0;
  }

  function detectBeat(low, now) {
    energyHistory[historyCursor] = low;
    historyCursor = (historyCursor + 1) % energyHistory.length;
    let avg = 0;
    for (const v of energyHistory) avg += v;
    avg /= energyHistory.length;

    let variance = 0;
    for (const v of energyHistory) variance += (v - avg) * (v - avg);
    variance /= energyHistory.length;
    const sd = Math.sqrt(variance);
    const threshold = Math.max(.16, avg + sd * 1.35);
    const refractory = now - lastBeatAt > 225;

    if (refractory && low > threshold && low > .22) {
      lastBeatAt = now;
      beatCount++;
      impact = clamp(impact + .65, 0, 1.6);
      if (autoVisualEnabled && beatCount % 16 === 0 && now - lastImageAt > 6500) changeVisual();
      return true;
    }
    return false;
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = Math.max(1, Math.round(rect.width));
    height = Math.max(1, Math.round(rect.height));
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seedParticles();
  }

  function seedParticles() {
    const count = width < 700 ? 120 : 210;
    particles = Array.from({ length: count }, (_, i) => ({
      angle: (i / count) * Math.PI * 2 + random(-.05, .05),
      radius: random(.88, 1.42),
      speed: random(-.12, .12),
      drift: random(-.2, .2),
      size: random(.45, 1.9),
      phase: random(0, Math.PI * 2),
      brightness: random(.25, 1)
    }));
  }

  function drawWaveHalo(cx, cy, baseR, low, mid, high, t) {
    if (!wave || !analyser) return;
    analyser.getByteTimeDomainData(wave);
    const steps = Math.min(380, wave.length);
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineJoin = 'round';

    for (let pass = 0; pass < 3; pass++) {
      ctx.beginPath();
      for (let s = 0; s <= steps; s++) {
        const idx = Math.floor((s / steps) * (wave.length - 1));
        const sample = (wave[idx] - 128) / 128;
        const a = (s / steps) * Math.PI * 2;
        const audioPush = sample * (9 + mid * 22 + pass * 3) * haloIntensity;
        const ripple = Math.sin(a * 6 + t * 1.8 + pass) * (1.3 + high * 4.5);
        const r = baseR * (1 + pass * .018) + audioPush + ripple;
        const x = cx + Math.cos(a) * r;
        const y = cy + Math.sin(a) * r * .78;
        if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      const alpha = (.15 + high * .28 + impact * .12) * haloIntensity;
      ctx.strokeStyle = `hsla(${205 + pass * 16 + high * 30}, 100%, ${74 + pass * 5}%, ${alpha})`;
      ctx.lineWidth = (1.0 + pass * 1.1 + low * 2.8) * haloIntensity;
      ctx.shadowBlur = (10 + pass * 10 + high * 22) * haloIntensity;
      ctx.shadowColor = `hsla(${210 + pass * 18},100%,75%,.75)`;
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawRings(cx, cy, baseR, low, mid, high, t) {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const ringCount = 7;

    for (let i = 0; i < ringCount; i++) {
      const p = i / (ringCount - 1);
      const r = baseR * (0.74 + p * .66) * (1 + low * .04 + impact * .018);
      const wobble = Math.sin(t * (.55 + p * .35) + i * 1.4) * (2 + mid * 8);
      const squash = .70 + Math.sin(t * .31 + i) * .05;
      ctx.beginPath();
      ctx.ellipse(cx, cy, r + wobble, (r + wobble) * squash, t * .06 + i * .07, 0, Math.PI * 2);
      ctx.lineWidth = (.6 + (1 - p) * 1.2 + low * 1.8) * haloIntensity;
      const alpha = (.055 + (1 - p) * .09 + high * .08 + impact * .04) * haloIntensity;
      ctx.strokeStyle = `hsla(${202 + p * 54 + high * 22}, 100%, ${66 + p * 20}%, ${alpha})`;
      ctx.shadowBlur = (9 + low * 18 + (1-p) * 12) * haloIntensity;
      ctx.shadowColor = `hsla(${205 + p * 32},100%,72%,.55)`;
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawSpokes(cx, cy, baseR, low, mid, high, t) {
    const count = 72;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < count; i++) {
      const a = (i / count) * Math.PI * 2 + t * .025;
      const harmonic = .5 + .5 * Math.sin(i * 1.731 + t * 1.3);
      const amp = (6 + high * 46 + mid * 18) * harmonic * haloIntensity;
      const r0 = baseR * (1.01 + .025 * Math.sin(i * .8 + t));
      const r1 = r0 + amp + impact * 18 * ((i % 3) / 2);
      const sx = cx + Math.cos(a) * r0;
      const sy = cy + Math.sin(a) * r0 * .78;
      const ex = cx + Math.cos(a) * r1;
      const ey = cy + Math.sin(a) * r1 * .78;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.lineWidth = .55 + high * .8;
      ctx.strokeStyle = `hsla(${190 + high * 70 + (i % 9)}, 100%, 78%, ${(0.035 + high * .16 + impact * .06) * haloIntensity})`;
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawParticles(cx, cy, baseR, low, mid, high, t, dt) {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    for (const p of particles) {
      p.angle += p.speed * dt * (0.65 + mid * 1.8);
      const pulse = Math.sin(t * (1.2 + p.brightness) + p.phase) * (.015 + high * .035);
      const r = baseR * (p.radius + pulse + impact * .018 * p.drift);
      const x = cx + Math.cos(p.angle) * r;
      const y = cy + Math.sin(p.angle) * r * .79;
      const size = p.size * (.65 + high * 1.9 + impact * .8) * haloIntensity;
      const hue = 196 + 48 * p.brightness + high * 42;
      ctx.fillStyle = `hsla(${hue},100%,${72 + p.brightness * 22}%,${(.16 + p.brightness * .36 + high * .18) * haloIntensity})`;
      ctx.shadowBlur = (5 + high * 16) * haloIntensity;
      ctx.shadowColor = `hsla(${hue},100%,78%,.65)`;
      ctx.beginPath();
      ctx.arc(x, y, Math.max(.3, size), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawCore(cx, cy, baseR, low, high, t) {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const r = baseR * (.22 + low * .035 + impact * .02);
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 1.7);
    g.addColorStop(0, `rgba(235,249,255,${(.12 + high * .10 + impact * .08) * haloIntensity})`);
    g.addColorStop(.2, `rgba(111,190,255,${(.07 + low * .10) * haloIntensity})`);
    g.addColorStop(1, 'rgba(70,110,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(cx - r*2, cy - r*2, r*4, r*4);

    if (impact > .15) {
      ctx.beginPath();
      ctx.moveTo(cx, cy - baseR * (.76 + impact * .13));
      ctx.lineTo(cx, cy + baseR * (.76 + impact * .13));
      ctx.strokeStyle = `rgba(211,240,255,${impact * .21 * haloIntensity})`;
      ctx.lineWidth = 1 + impact * 1.7;
      ctx.shadowBlur = 18 + impact * 36;
      ctx.shadowColor = 'rgba(124,192,255,.9)';
      ctx.stroke();
    }
    ctx.restore();
  }

  function animate(now) {
    raf = requestAnimationFrame(animate);
    const rawDt = Math.min(40, now - lastFrame);
    const dt = rawDt / 1000;
    lastFrame = now;
    const t = now / 1000;

    let low = 0, mid = 0, high = 0;
    if (analyser && !audio.paused) {
      analyser.getByteFrequencyData(freq);
      low = bandEnergy(36, 180);
      mid = bandEnergy(180, 2200);
      high = bandEnergy(2200, 12000);
      detectBeat(low, now);
    }

    lowEnvelope = lerp(lowEnvelope, low, 1 - Math.exp(-dt * 9));
    midEnvelope = lerp(midEnvelope, mid, 1 - Math.exp(-dt * 7));
    highEnvelope = lerp(highEnvelope, high, 1 - Math.exp(-dt * 11));
    impact *= Math.exp(-dt * 4.6);

    if (autoVisualEnabled && started && now - lastImageAt > nextFallbackImageAt) changeVisual();

    const bassScale = 1 + lowEnvelope * .026 + impact * .016;
    const glow = .42 + lowEnvelope * 1.2 + highEnvelope * .3 + impact * .55;
    artFrame.style.setProperty('--art-scale', bassScale.toFixed(4));
    artFrame.style.setProperty('--art-glow', glow.toFixed(3));
    artFrame.style.filter = `saturate(${1 + midEnvelope * .12}) brightness(${1 + highEnvelope * .035 + impact * .035})`;
    flash.style.setProperty('--flash', clamp(impact * .12 + highEnvelope * .02, 0, .2).toFixed(3));

    const compositeEnergy = clamp(lowEnvelope * .52 + midEnvelope * .30 + highEnvelope * .18);
    energyLabel.textContent = !started ? 'IDLE' : audio.paused ? 'PAUSED' : compositeEnergy > .62 ? 'PEAK' : compositeEnergy > .36 ? 'DRIVE' : 'FLOW';

    meters.forEach((bar, i) => {
      const q = i / Math.max(1, meters.length - 1);
      const e = q < .33 ? lowEnvelope : q < .72 ? midEnvelope : highEnvelope;
      const mod = .6 + .4 * Math.sin(t * 6.2 + i * .93);
      bar.style.height = `${3 + 18 * clamp(e * 1.3 * mod + impact * .22)}px`;
    });

    ctx.clearRect(0, 0, width, height);
    const cx = width / 2;
    const cy = height / 2;
    const baseR = Math.min(width, height) * (width < 700 ? .315 : .302);
    drawRings(cx, cy, baseR, lowEnvelope, midEnvelope, highEnvelope, t);
    drawSpokes(cx, cy, baseR, lowEnvelope, midEnvelope, highEnvelope, t);
    drawWaveHalo(cx, cy, baseR, lowEnvelope, midEnvelope, highEnvelope, t);
    drawParticles(cx, cy, baseR, lowEnvelope, midEnvelope, highEnvelope, t, dt);
    drawCore(cx, cy, baseR, lowEnvelope, highEnvelope, t);
  }

  function setIntensity(next) {
    haloIntensity = next;
    intensityButton.textContent = `HALO ${Math.round(haloIntensity * 100)}%`;
  }

  startButton.addEventListener('click', begin);
  playPause.addEventListener('click', togglePlay);
  prevTrack.addEventListener('click', () => loadTrack(trackIndex - 1, !audio.paused || started));
  nextTrack.addEventListener('click', () => loadTrack(trackIndex + 1, !audio.paused || started));
  shuffleImage.addEventListener('click', () => changeVisual());

  autoVisual.addEventListener('click', () => {
    autoVisualEnabled = !autoVisualEnabled;
    autoVisual.classList.toggle('active', autoVisualEnabled);
    autoVisual.setAttribute('aria-pressed', String(autoVisualEnabled));
    autoVisual.textContent = `AUTO VISUAL ${autoVisualEnabled ? 'ON' : 'OFF'}`;
  });

  intensityButton.addEventListener('click', () => {
    const stops = [.65, 1, 1.35];
    const idx = stops.findIndex(v => Math.abs(v - haloIntensity) < .03);
    setIntensity(stops[(idx + 1) % stops.length]);
  });

  fullscreenButton.addEventListener('click', async () => {
    try {
      if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
      else await document.exitFullscreen();
    } catch (_) {}
  });

  seek.addEventListener('input', () => {
    if (Number.isFinite(audio.duration) && audio.duration > 0) {
      audio.currentTime = (Number(seek.value) / 1000) * audio.duration;
    }
  });

  audio.addEventListener('play', () => {
    app.dataset.state = 'playing';
    playPause.textContent = '❚❚';
    playPause.setAttribute('aria-label', 'Pause');
  });
  audio.addEventListener('pause', () => {
    if (started) app.dataset.state = 'paused';
    playPause.textContent = '▶';
    playPause.setAttribute('aria-label', 'Play');
  });
  audio.addEventListener('loadedmetadata', () => {
    durationEl.textContent = formatTime(audio.duration);
  });
  audio.addEventListener('timeupdate', () => {
    currentTimeEl.textContent = formatTime(audio.currentTime);
    durationEl.textContent = formatTime(audio.duration);
    if (Number.isFinite(audio.duration) && audio.duration > 0) seek.value = String(Math.round((audio.currentTime / audio.duration) * 1000));
  });
  audio.addEventListener('ended', () => loadTrack(trackIndex + 1, true));

  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    else if (e.code === 'ArrowRight' && (e.metaKey || e.ctrlKey)) loadTrack(trackIndex + 1, true);
    else if (e.code === 'ArrowLeft' && (e.metaKey || e.ctrlKey)) loadTrack(trackIndex - 1, true);
    else if (e.key.toLowerCase() === 's') changeVisual();
    else if (e.key.toLowerCase() === 'f') fullscreenButton.click();
  });

  window.addEventListener('resize', resize, { passive: true });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) lastFrame = performance.now();
  });

  // Prime the visual without violating autoplay rules.
  loadTrack(0, false);
  changeVisual(Math.floor(Math.random() * VISUALS.length));
  resize();
  setIntensity(1);
  raf = requestAnimationFrame(animate);
})();
