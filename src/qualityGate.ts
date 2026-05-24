import type { Thresholds } from './config';
import type { RgbFrame } from './detector';
import type {
  DetectedFace,
  QualityIssue,
  QualityResult,
} from './types';

/**
 * Gate a detected frame before it is allowed into liveness / embedding.
 * Rejects frames that would produce unreliable embeddings or are trivially
 * spoofable, and returns human-readable guidance for the worst issue so the
 * host UI can coach the user ("Move closer", "Look straight at the camera").
 */

const GUIDANCE: Record<QualityIssue, string> = {
  no_face: 'Position your face in the frame',
  multiple_faces: 'Only one person at a time',
  face_too_small: 'Move closer',
  off_center: 'Center your face',
  bad_pose: 'Look straight at the camera',
  too_dark: 'Find better lighting',
  too_bright: 'Reduce glare / bright light',
  blurry: 'Hold still',
};

/** Priority order for which issue to surface first. */
const PRIORITY: QualityIssue[] = [
  'no_face',
  'multiple_faces',
  'face_too_small',
  'off_center',
  'bad_pose',
  'too_dark',
  'too_bright',
  'blurry',
];

export interface QualityInput {
  /** Faces detected this frame (0, 1, or many). */
  faces: DetectedFace[];
  /**
   * The frame pixels, used for brightness + sharpness.
   * Optional: when absent, pass `brightness` directly (worklet mode).
   */
  frame?: RgbFrame | null;
  /**
   * Pre-computed mean luma (0–255) from the worklet.
   * Used only when `frame` is absent.
   */
  brightness?: number;
}

export function evaluateQuality(
  input: QualityInput,
  t: Thresholds,
): QualityResult {
  const issues: QualityIssue[] = [];
  const { faces, frame } = input;
  if (faces.length === 0) {
    return fail(['no_face']);
  }
  if (faces.length > 1) {
    issues.push('multiple_faces');
  }
  const face = largestFace(faces);
  const area = face.box.width * face.box.height;
  if (area < t.minFaceArea) issues.push('face_too_small');

  if (!isCentered(face, t)) issues.push('off_center');

  const { yaw, pitch, roll } = face.headPose;
  if (Math.abs(yaw) > t.maxYaw || Math.abs(pitch) > t.maxPitch || Math.abs(roll) > t.maxRoll) {
    issues.push('bad_pose');
  }

  // Brightness: use precomputed value from worklet when frame is absent.
  const luma = frame ? meanLuma(frame, face) : (input.brightness ?? 128);
  if (luma < t.minBrightness) issues.push('too_dark');
  else if (luma > t.maxBrightness) issues.push('too_bright');

  // Sharpness: only checkable with raw frame data; skip in worklet mode.
  if (frame && sharpness(frame, face) < t.minSharpness) issues.push('blurry');

  return issues.length === 0
    ? { ok: true, issues: [], guidance: '' }
    : fail(issues);
}

function fail(issues: QualityIssue[]): QualityResult {
  const top = PRIORITY.find((i) => issues.includes(i)) ?? issues[0]!;
  return { ok: false, issues, guidance: GUIDANCE[top] };
}

export function largestFace(faces: DetectedFace[]): DetectedFace {
  return faces.reduce((a, b) =>
    a.box.width * a.box.height >= b.box.width * b.box.height ? a : b,
  );
}

/** Face centre must fall within the configurable margin from each edge. */
function isCentered(face: DetectedFace, t: Thresholds): boolean {
  const cx = face.box.x + face.box.width  / 2;
  const cy = face.box.y + face.box.height / 2;
  const m  = t.centerMargin;
  return cx > m && cx < (1 - m) && cy > m && cy < (1 - m);
}

/** Mean luma (0..255) over the face box region. Samples on a grid to stay cheap. */
function meanLuma(frame: RgbFrame, face: DetectedFace): number {
  const { sum, count } = sampleRegion(frame, face, (r, g, b, acc) => {
    acc.sum += 0.299 * r + 0.587 * g + 0.114 * b;
    acc.count += 1;
  });
  return count === 0 ? 0 : sum / count;
}

/** Variance-of-Laplacian style sharpness estimate over the face region. */
function sharpness(frame: RgbFrame, face: DetectedFace): number {
  const x0 = Math.floor(face.box.x * frame.width);
  const y0 = Math.floor(face.box.y * frame.height);
  const w = Math.floor(face.box.width * frame.width);
  const h = Math.floor(face.box.height * frame.height);
  const step = Math.max(1, Math.floor(Math.min(w, h) / 32));
  let mean = 0;
  let m2 = 0;
  let n = 0;
  const luma = (px: number, py: number): number => {
    const cx = Math.min(frame.width - 1, Math.max(0, px));
    const cy = Math.min(frame.height - 1, Math.max(0, py));
    const i = (cy * frame.width + cx) * 3;
    return 0.299 * frame.data[i]! + 0.587 * frame.data[i + 1]! + 0.114 * frame.data[i + 2]!;
  };
  for (let y = y0 + step; y < y0 + h - step; y += step) {
    for (let x = x0 + step; x < x0 + w - step; x += step) {
      const lap =
        4 * luma(x, y) - luma(x - step, y) - luma(x + step, y) - luma(x, y - step) - luma(x, y + step);
      n += 1;
      const delta = lap - mean;
      mean += delta / n;
      m2 += delta * (lap - mean);
    }
  }
  return n > 1 ? m2 / (n - 1) : 0;
}

function sampleRegion(
  frame: RgbFrame,
  face: DetectedFace,
  fn: (r: number, g: number, b: number, acc: { sum: number; count: number }) => void,
): { sum: number; count: number } {
  const acc = { sum: 0, count: 0 };
  const x0 = Math.floor(face.box.x * frame.width);
  const y0 = Math.floor(face.box.y * frame.height);
  const w = Math.floor(face.box.width * frame.width);
  const h = Math.floor(face.box.height * frame.height);
  const step = Math.max(1, Math.floor(Math.min(w, h) / 24));
  for (let y = y0; y < y0 + h; y += step) {
    for (let x = x0; x < x0 + w; x += step) {
      const cx = Math.min(frame.width - 1, Math.max(0, x));
      const cy = Math.min(frame.height - 1, Math.max(0, y));
      const i = (cy * frame.width + cx) * 3;
      fn(frame.data[i]!, frame.data[i + 1]!, frame.data[i + 2]!, acc);
    }
  }
  return acc;
}
