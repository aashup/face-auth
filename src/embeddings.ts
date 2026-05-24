import {
  LANDMARK,
  resampleToSquare,
  type RgbFrame,
  type TfliteModel,
} from './detector';
import type { BoundingBox, DetectedFace, Point } from './types';

/**
 * MobileFaceNet embedding.
 *
 * Produces an L2-normalized feature vector for a detected face. The crucial
 * invariant (PLAN.md Phase 2) is that this preprocessing must be byte-identical
 * to the server enrollment pipeline, or device embeddings won't be comparable
 * to provisioned templates. Keep all preprocessing here, in JS, so it is the
 * same on Android + iOS and easy to mirror server-side.
 */

// Embedding model input edge. FaceNet-512 (the bundled model) is 160×160;
// MobileFaceNet would be 112. Must match the model + server preprocessing.
const INPUT = 160;

/** Pixels normalized to [-1, 1] (x/127.5 - 1) — matches the bundled embedder. */
function normalizeMinusOneToOne(rgb01: Float32Array): Float32Array {
  const out = new Float32Array(rgb01.length);
  for (let i = 0; i < rgb01.length; i++) {
    out[i] = rgb01[i]! * 2 - 1; // (x*255)/127.5 - 1  ==  x*2 - 1
  }
  return out;
}

/**
 * Similarity-align the face to a canonical 112×112 using the two eye centers,
 * so pose/scale variation doesn't leak into the embedding. We compute the eye
 * centers from the Face Mesh eye rings, derive the roll angle + inter-ocular
 * distance, and resample an upright, scale-normalized crop.
 */
export function alignFace(frame: RgbFrame, face: DetectedFace): Float32Array {
  const leftEye = centroid(face.landmarks, LANDMARK.leftEye);
  const rightEye = centroid(face.landmarks, LANDMARK.rightEye);

  if (!leftEye || !rightEye) {
    // Fallback: plain box crop (still works, just less robust to tilt).
    return normalizeMinusOneToOne(resampleToSquare(frame, face.box, INPUT));
  }

  const dx = rightEye.x - leftEye.x;
  const dy = rightEye.y - leftEye.y;
  const angle = Math.atan2(dy, dx); // radians, frame-normalized space
  const eyeDist = Math.hypot(dx * frame.width, dy * frame.height); // pixels

  // Canonical: eyes span ~38% of the 112px crop, centered ~38% down.
  const desiredEyeDist = INPUT * 0.38;
  const scale = eyeDist > 1 ? desiredEyeDist / eyeDist : 1;

  const eyeMid: Point = {
    x: (leftEye.x + rightEye.x) / 2,
    y: (leftEye.y + rightEye.y) / 2,
  };
  const cx = eyeMid.x * frame.width;
  const cy = eyeMid.y * frame.height;

  const out = new Float32Array(INPUT * INPUT * 3);
  const cos = Math.cos(-angle);
  const sin = Math.sin(-angle);
  const targetEyeY = INPUT * 0.38;

  for (let y = 0; y < INPUT; y++) {
    for (let x = 0; x < INPUT; x++) {
      // map output px -> source px: undo center/scale/rotation
      const ox = (x - INPUT / 2) / scale;
      const oy = (y - targetEyeY) / scale;
      const sxp = cx + ox * cos - oy * sin;
      const syp = cy + ox * sin + oy * cos;
      const fx = Math.min(frame.width - 1, Math.max(0, Math.round(sxp)));
      const fy = Math.min(frame.height - 1, Math.max(0, Math.round(syp)));
      const src = (fy * frame.width + fx) * 3;
      const dst = (y * INPUT + x) * 3;
      out[dst] = frame.data[src]! / 255;
      out[dst + 1] = frame.data[src + 1]! / 255;
      out[dst + 2] = frame.data[src + 2]! / 255;
    }
  }
  return normalizeMinusOneToOne(out);
}

/** Run the embedding model (off-thread) and return an L2-normalized vector. */
export async function computeEmbedding(
  frame: RgbFrame,
  face: DetectedFace,
  model: TfliteModel,
): Promise<Float32Array> {
  const input = alignFace(frame, face);
  const raw = (await model.run([input]))[0] as Float32Array;
  return l2Normalize(raw);
}

/** Average several per-frame embeddings (then renormalize) to cut variance. */
export function averageEmbeddings(embeddings: Float32Array[]): Float32Array {
  if (embeddings.length === 0) throw new Error('no embeddings to average');
  const dims = embeddings[0]!.length;
  const acc = new Float32Array(dims);
  for (const e of embeddings) {
    for (let i = 0; i < dims; i++) acc[i]! += e[i]!;
  }
  for (let i = 0; i < dims; i++) acc[i]! /= embeddings.length;
  return l2Normalize(acc);
}

/**
 * Dot product of two L2-normalized vectors.
 *
 * For unit-length embeddings this is mathematically identical to cosine
 * similarity but skips the magnitude divisions — the cheapest possible
 * similarity metric for an O(N) matching loop.
 */
export function dotProduct(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error('embedding dimension mismatch');
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
  return dot;
}

/** @deprecated Vectors are L2-normalised — use {@link dotProduct} directly. */
export const cosineSimilarity = dotProduct;

/** Encode an embedding as base64 little-endian float32 (server wire format). */
export function embeddingToBase64(v: Float32Array): string {
  const bytes = new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
  if (typeof globalThis.btoa === 'function') {
    let bin = '';
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]!);
    return globalThis.btoa(bin);
  }
  return (globalThis as any).Buffer.from(bytes).toString('base64');
}

export function l2Normalize(v: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i]! * v[i]!;
  norm = Math.sqrt(norm) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i]! / norm;
  return out;
}

function centroid(landmarks: Point[], indices: readonly number[]): Point | null {
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const i of indices) {
    const p = landmarks[i];
    if (!p) continue;
    sx += p.x;
    sy += p.y;
    n += 1;
  }
  return n === 0 ? null : { x: sx / n, y: sy / n };
}

/** Re-export for callers that crop directly from a box (server parity tooling). */
export function cropForEmbedding(frame: RgbFrame, box: BoundingBox): Float32Array {
  return normalizeMinusOneToOne(resampleToSquare(frame, box, INPUT));
}
