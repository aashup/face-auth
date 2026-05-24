import type { RgbFrame, TfliteModel } from '../detector';
import { LANDMARK } from '../detector';
import { l2Normalize } from '../embeddings';
import type { DetectedFace, HeadPose, Point } from '../types';

/** Build a 468-point landmark array with the indices our code reads set to
 *  produce a target Eye-Aspect-Ratio and mouth ratio. */
export function buildLandmarks(opts: { ear?: number; mouthRatio?: number } = {}): Point[] {
  const pts: Point[] = Array.from({ length: 468 }, () => ({ x: 0, y: 0 }));

  // Eyes: horizontal span 0.1, verticals chosen so EAR == target.
  const ear = opts.ear ?? 0.3;
  const setEye = (idx: readonly number[]) => {
    const [p0, p1, p2, p3, p4, p5] = idx;
    const h = 0.1;
    // EAR = (dist(p1,p5)+dist(p2,p4)) / (2*dist(p0,p3)) = (2v+2v)/(2h) = 2v/h.
    // Want EAR == ear  →  v = ear*h/2.
    const v = (ear * h) / 2;
    pts[p0!] = { x: 0.3, y: 0.5 };
    pts[p3!] = { x: 0.3 + h, y: 0.5 };
    pts[p1!] = { x: 0.33, y: 0.5 - v };
    pts[p5!] = { x: 0.33, y: 0.5 + v };
    pts[p2!] = { x: 0.37, y: 0.5 - v };
    pts[p4!] = { x: 0.37, y: 0.5 + v };
  };
  setEye(LANDMARK.leftEye);
  setEye(LANDMARK.rightEye);

  // Mouth: width 0.2, height = width / ratio.
  const ratio = opts.mouthRatio ?? 0.4;
  const width = 0.2;
  const height = width / ratio;
  pts[LANDMARK.mouthLeft] = { x: 0.4, y: 0.7 };
  pts[LANDMARK.mouthRight] = { x: 0.4 + width, y: 0.7 };
  pts[LANDMARK.mouthTop] = { x: 0.5, y: 0.7 };
  pts[LANDMARK.mouthBottom] = { x: 0.5, y: 0.7 + height };

  return pts;
}

export function makeFace(opts: {
  box?: { x: number; y: number; width: number; height: number };
  headPose?: Partial<HeadPose>;
  ear?: number;
  mouthRatio?: number;
  score?: number;
} = {}): DetectedFace {
  return {
    box: opts.box ?? { x: 0.35, y: 0.3, width: 0.3, height: 0.4 },
    landmarks: buildLandmarks({ ear: opts.ear, mouthRatio: opts.mouthRatio }),
    headPose: { yaw: 0, pitch: 0, roll: 0, ...opts.headPose },
    score: opts.score ?? 0.95,
  };
}

/** Solid-color frame (uniform → low sharpness; relax thresholds in tests). */
export function makeFrame(size = 32, fill = 128): RgbFrame {
  return { data: new Uint8Array(size * size * 3).fill(fill), width: size, height: size };
}

/** A TFLite stub that ignores its input and returns a fixed output vector. */
export function stubModel(output: Float32Array): TfliteModel {
  return { runSync: () => [output], run: async () => [output] };
}

/**
 * Liveness branch stub.
 *
 * `decodeLiveProb` in liveness.ts treats **index 0** as the real/live class.
 * The stub therefore puts the live logit at index 0; indices 1–2 are spoof
 * classes set to 0.
 *
 *   liveLogit > 0  → large live-class logit → liveScore ≈ 1 → spoofScore ≈ 0 → passes
 *   liveLogit < 0  → small live-class logit → liveScore ≈ 0 → spoofScore ≈ 1 → rejected
 */
export function livenessStub(liveLogit: number): TfliteModel {
  return stubModel(new Float32Array([liveLogit, 0, 0]));
}

/** Make a normalized embedding deterministically from a seed. */
export function seededEmbedding(dims: number, seed: number): Float32Array {
  let s = seed >>> 0;
  const v = new Float32Array(dims);
  for (let i = 0; i < dims; i++) {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    v[i] = (s / 0x100000000) * 2 - 1;
  }
  return l2Normalize(v);
}
