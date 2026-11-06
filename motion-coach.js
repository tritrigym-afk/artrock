// motion-coach.js
// Camera + MediaPipe body tracking + sounds, wired to route-checker.js.
// Framework-free ES module: import it from plain HTML, React, Vue, etc.

import { PoseLandmarker, FilesetResolver } from
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
import { createRouteChecker } from './route-checker.js';

export const MEDIAPIPE = {
  wasm: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
  modelLite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
  modelFull: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
};

// MediaPipe pose landmark indices we use.
export const LM = {
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_INDEX: 19, RIGHT_INDEX: 20,
  LEFT_FOOT: 31, RIGHT_FOOT: 32,
};

// Skeleton lines for drawing (pairs of landmark indices).
export const SKELETON = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [15, 19], [16, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 31], [28, 32],
];

/** Load the pose model. Tries the GPU first, falls back to the CPU. */
export async function loadPose({ model = MEDIAPIPE.modelLite } = {}) {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE.wasm);
  const make = (delegate) => PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: model, delegate },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try { return await make('GPU'); } catch (e) {
    console.warn('GPU pose model failed, using CPU', e);
    return make('CPU');
  }
}

/** Start the rear camera in a <video>. Needs HTTPS or localhost. */
export async function startCamera(video, { facingMode = 'environment', width = 1280, height = 720 } = {}) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera not available. Open the page over HTTPS (or localhost).');
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: facingMode }, width: { ideal: width }, height: { ideal: height } },
  });
  video.srcObject = stream;
  video.playsInline = true;
  video.muted = true;
  await video.play();
  if (!video.videoWidth) await new Promise((r) => video.addEventListener('loadedmetadata', r, { once: true }));
  return stream;
}

export function stopCamera(video) {
  video.srcObject?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

/** Ding / buzz / spoken feedback. Call unlock() inside a tap or click first (phones block audio otherwise). */
export function createSounds({ speak = true } = {}) {
  let ctx = null;
  const state = { speak };

  function unlock() {
    ctx ??= new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === 'suspended') ctx.resume();
    // iOS: an empty utterance in the tap handler enables speech later.
    if ('speechSynthesis' in window) speechSynthesis.speak(new SpeechSynthesisUtterance(''));
  }

  function tone(freq, secs, type = 'sine', at = 0, vol = 0.3) {
    if (!ctx) return;
    const t0 = ctx.currentTime + at;
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = type;
    o.frequency.value = freq;
    g.gain.setValueAtTime(vol, t0);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + secs);
    o.connect(g).connect(ctx.destination);
    o.start(t0);
    o.stop(t0 + secs);
  }

  function say(text) {
    if (!state.speak || !('speechSynthesis' in window)) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'en-US';
    speechSynthesis.speak(u);
  }

  return {
    unlock,
    say,
    set speak(v) { state.speak = v; },
    get speak() { return state.speak; },
    ding() { tone(1046, 0.45); },
    wrong() { tone(180, 0.18, 'square', 0, 0.2); tone(140, 0.25, 'square', 0.2, 0.2); say('Wrong hold'); },
    top() { tone(784, 0.25); tone(1046, 0.25, 'sine', 0.15); tone(1318, 0.5, 'sine', 0.3); say('Top! Well done'); },
  };
}

/**
 * The coach: runs pose tracking on the video and checks the route.
 *
 * route / allHolds: [{ id, x, y, r }] in VIDEO pixel coordinates.
 * Take the route photo from the same fixed (tripod) camera so the coordinates match.
 *
 * onEvent({ type: 'correct' | 'top' | 'wrong', hold, index })
 * onFrame({ landmarks, points, fps })  – for drawing an overlay
 */
export function createMotionCoach({
  pose, video, route, allHolds = route, sounds,
  trackFeet = false, dwellMs = 400, minVisibility = 0.5,
  onEvent = () => {}, onFrame = () => {},
}) {
  let checker = createRouteChecker({ route, allHolds, dwellMs });
  let running = false;
  let rafId = 0;
  let lastVideoTime = -1;
  let lastTs = 0;
  let fps = 0;
  let feet = trackFeet;

  function pointsFrom(landmarks) {
    const W = video.videoWidth, H = video.videoHeight;
    const ids = [LM.LEFT_INDEX, LM.RIGHT_INDEX, LM.LEFT_WRIST, LM.RIGHT_WRIST];
    if (feet) ids.push(LM.LEFT_FOOT, LM.RIGHT_FOOT);
    return ids
      .map((i) => landmarks[i])
      .filter((p) => p && (p.visibility ?? 1) >= minVisibility)
      .map((p) => ({ x: p.x * W, y: p.y * H }));
  }

  function loop() {
    if (!running) return;
    rafId = requestAnimationFrame(loop);
    if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;

    const now = performance.now();
    const res = pose.detectForVideo(video, now);
    const landmarks = res.landmarks?.[0] ?? null;
    const points = landmarks ? pointsFrom(landmarks) : [];

    if (lastTs) fps = fps * 0.9 + (1000 / (now - lastTs)) * 0.1;
    lastTs = now;

    for (const ev of checker.update(points, now)) {
      if (sounds) {
        if (ev.type === 'correct') sounds.ding();
        else if (ev.type === 'top') sounds.top();
        else sounds.wrong();
      }
      onEvent(ev);
    }
    onFrame({ landmarks, points, fps, nextIndex: checker.nextIndex, done: checker.done });
  }

  return {
    start() { if (!running) { running = true; lastVideoTime = -1; loop(); } },
    stop() { running = false; cancelAnimationFrame(rafId); },
    reset() { checker.reset(); },
    setRoute(newRoute, newAll = newRoute) { route = newRoute; allHolds = newAll; checker = createRouteChecker({ route, allHolds, dwellMs }); },
    set trackFeet(v) { feet = v; },
    get nextIndex() { return checker.nextIndex; },
    get done() { return checker.done; },
    get running() { return running; },
  };
}
