import { useCallback, useRef } from 'react';
import { useFrameProcessor, type Frame } from 'react-native-vision-camera';
import { useRunOnJS, useSharedValue } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';

import type { TfliteModel } from './detector';

// ─── constants ────────────────────────────────────────────────────────────────

/** Single resize size — all model crops are drawn from this buffer in the worklet. */
export const FRAME_SIZE = 480;

// Model input edges (must match bundled .tflite files)
const BF = 128;  // BlazeFace
const FM = 192;  // FaceMesh
const LV = 80;   // MiniFASNet liveness
const EM = 160;  // FaceNet-512

// ─── public types ─────────────────────────────────────────────────────────────

/** All TFLite models run synchronously inside the frame-processor worklet. */
export interface FrameSourceModels {
  blazeface: TfliteModel;
  faceMesh: TfliteModel;
  liveness0: TfliteModel;  // spoof_2_7.tflite
  liveness1: TfliteModel;  // spoof_4_0.tflite
  embedding: TfliteModel;
}

/**
 * Centering status delivered from the worklet to the JS thread.
 * Fired only when the status or arrow direction changes — not every frame.
 */
export type CenteringStatus =
  | { detected: false }
  | { detected: true; centered: boolean; cx: number; cy: number };

/**
 * Result marshalled from the native worklet thread to the JS thread.
 * Every field is a plain JS primitive or number[] — the only types that
 * worklets-core v1.x can safely transfer through useRunOnJS.
 * TypedArrays / ArrayBuffers are NOT supported and will throw at init time.
 */
export type WorkletFrameResult =
  | {
      detected: false;
      /** Raw camera-sensor pixel dimensions (not display-rotated).
       *  Logged once to diagnose aspect-ratio / orientation issues. */
      frameW: number;
      frameH: number;
    }
  | {
      detected: true;
      /** Normalised [x, y, w, h, score] 0..1 */
      bbox: [number, number, number, number, number];
      /** 468 × [x, y] flat, normalised 0..1. Length = 936. */
      landmarks: number[];
      /** Mean luma (0–255) of the face region in the 256×256 frame. */
      brightness: number;
      /** Averaged passive-liveness score (0 = spoof, 1 = live). */
      livenessScore: number;
      /**
       * DEBUG — raw softmax output of each liveness branch BEFORE any index
       * selection. Array layout: [branch0_val0, branch0_val1, ..., SEPARATOR(-1),
       * branch1_val0, branch1_val1, ...].
       * Log this to diagnose wrong class-index or double-normalisation.
       * Remove once liveness is confirmed working.
       */
      livenessRaw: number[];
      /** L2-normalised 512-d FaceNet embedding, or null if not needed this frame. */
      embedding: number[] | null;
      /** Raw camera-sensor pixel dimensions (not display-rotated).
       *  Logged once to diagnose aspect-ratio / orientation issues. */
      frameW: number;
      frameH: number;
    };

/**
 * Snapshot of the raw 256×256 uint8 RGB buffer (downsampled to 64×64 for
 * transfer efficiency) captured on demand from the worklet thread.
 * Useful for verifying exactly what image the TFLite models receive.
 */
/**
 * Three BMP images encoded as base64 strings — produced entirely in the worklet
 * thread so only strings (not TypedArrays) cross to JS.
 *
 * Files saved to faceauth_debug/:
 *   src_frame_<ts>_N.bmp  — 160×160 full frame (raw resize-plugin output, uint8 RGB)
 *   bf_input_<ts>_N.bmp   — 128×128 full-frame BlazeFace input (float32 → uint8)
 *   face_crop_<ts>_N.bmp  — 192×192 expanded face crop as FaceMesh sees it (empty if no face)
 *
 * Saved on every triggerDebugCapture() call, even when no face is detected,
 * so you can diagnose detection failures visually.
 */
export interface ModelDebugFrame {
  /** base64 24-bit BMP — full 480×480 frame downsampled to 160×160 (uint8 RGB as resize plugin delivers it) */
  srcFrameBmp: string;
  /** base64 24-bit BMP — 128×128 full frame as BlazeFace sees it (float32 [0,1]) */
  blazefaceBmp: string;
  /** base64 24-bit BMP — 192×192 face crop as FaceMesh sees it; empty string when no face detected */
  faceCropBmp: string;
  /** Detected face bbox [x, y, w, h] in 0-1 normalised coords, or null when no face detected */
  bboxNorm: [number, number, number, number] | null;
}

export interface FrameSourceOptions {
  models: FrameSourceModels;
  onFrameResult: (result: WorkletFrameResult) => void | Promise<void>;
  minIntervalMs?: number;
  /**
   * SharedValue set to true by FaceAuthView when the pipeline enters the
   * 'analyzing' stage and needs passive-liveness scores.
   * Keeps gating and challenge frames free of the two ~40 ms MiniFASNet calls.
   */
  needsLiveness: { value: boolean };
  /**
   * SharedValue set to true by FaceAuthView when the pipeline enters the
   * 'analyzing' stage and actually needs an embedding vector.
   * Keeps the common case (gating / challenge) free of the 80 ms FaceNet call.
   */
  needsEmbedding: { value: boolean };
  /**
   * Called once per debug capture request with a 64×64 thumbnail of the
   * exact buffer the models see. Trigger via the returned `triggerDebugCapture`.
   */
  onModelDebug?: (frame: ModelDebugFrame) => void;
  /**
   * Called from the JS thread when the face-centering status or arrow direction
   * changes. Fires at most a few times per second — never on every frame.
   * Drives the FaceMaskOverlay colour and direction arrows.
   */
  onCentering?: (status: CenteringStatus) => void;
  /**
   * Margin from each frame edge for the centering check (0-1 normalised).
   * Mirrors Thresholds.centerMargin. Default 0.10.
   */
  centerMargin?: number;
}

// ─── BlazeFace SSD anchor table (computed once at module load) ────────────────
// MediaPipe face_detection_short_range.tflite: num_layers=4, strides=[8,16,16,16],
// fixed_anchor_size=true, 2 anchors per cell.
// Total: 16×16×2 + (8×8×2)×3 = 512 + 384 = 896 anchors.
//
// IMPORTANT: must be plain number[] — NOT Float32Array / TypedArray.
// worklets-core serialises closure-captured values to the native thread at mount
// time. TypedArrays cannot be serialised and cause an immediate fatal crash
// ("Cannot serialize object"). A regular JS Array is copied as a JS value and
// works correctly inside the worklet.
function _buildBFAnchors(): [number[], number[]] {
  const CX: number[] = new Array(896);
  const CY: number[] = new Array(896);
  let idx = 0;
  // Layer 0 — stride 8 — feature map 16×16 — 2 anchors per cell
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      const cx = (x + 0.5) / 16;
      const cy = (y + 0.5) / 16;
      CX[idx] = cx; CY[idx] = cy; idx++;
      CX[idx] = cx; CY[idx] = cy; idx++;
    }
  }
  // Layers 1-3 — stride 16 — feature map 8×8 — 2 anchors per cell (×3 layers)
  for (let layer = 0; layer < 3; layer++) {
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const cx = (x + 0.5) / 8;
        const cy = (y + 0.5) / 8;
        CX[idx] = cx; CY[idx] = cy; idx++;
        CX[idx] = cx; CY[idx] = cy; idx++;
      }
    }
  }
  return [CX, CY];
}
const BF_ANCHORS = _buildBFAnchors();
const BF_ANCHOR_CX: number[] = BF_ANCHORS[0];
const BF_ANCHOR_CY: number[] = BF_ANCHORS[1];

// ─── worklet-safe helper functions ───────────────────────────────────────────
// Marked with 'worklet' so worklets-core's Babel transform serialises them
// into the native-thread Hermes instance. Pure synchronous math only.

function wSigmoid(x: number): number {
  'worklet';
  return 1 / (1 + Math.exp(-x));
}

/**
 * Nearest-neighbour resample from a Uint8Array (FRAME_SIZE×FRAME_SIZE×3 uint8)
 * into a Float32Array normalised to [0, 1].
 * Box coords are normalised 0..1.
 */
function wResample01(
  src: Uint8Array,
  srcW: number,
  srcH: number,
  boxX: number,
  boxY: number,
  boxW: number,
  boxH: number,
  size: number,
): Float32Array {
  'worklet';
  const out = new Float32Array(size * size * 3);
  const sx0 = boxX * srcW;
  const sy0 = boxY * srcH;
  const sw = boxW * srcW;
  const sh = boxH * srcH;
  for (let y = 0; y < size; y++) {
    const fy = Math.min(srcH - 1, Math.floor(sy0 + (y / size) * sh));
    for (let x = 0; x < size; x++) {
      const fx = Math.min(srcW - 1, Math.floor(sx0 + (x / size) * sw));
      const si = (fy * srcW + fx) * 3;
      const di = (y * size + x) * 3;
      out[di]     = (src[si]     ?? 0) / 255;
      out[di + 1] = (src[si + 1] ?? 0) / 255;
      out[di + 2] = (src[si + 2] ?? 0) / 255;
    }
  }
  return out;
}

/**
 * Resample and preprocess a crop for MiniFASNet (Silent-Face-Anti-Spoofing).
 *
 * MiniFASNet was trained in PyTorch with:
 *   transforms.ToTensor()           → /255  → [0,1]
 *   transforms.Normalize(
 *     mean=[0.406, 0.456, 0.485],   (BGR order — OpenCV default)
 *     std =[0.225, 0.224, 0.229])
 *
 * Our resize plugin delivers RGB, so we must:
 *   1. Swap channels: RGB → BGR
 *   2. Apply per-channel ImageNet normalisation: (x/255 - mean) / std
 *
 * Without this the model receives a completely out-of-distribution input
 * and collapses to predicting "spoof" for everything.
 */
function wResampleLiveness(
  src: Uint8Array,
  srcW: number,
  srcH: number,
  boxX: number,
  boxY: number,
  boxW: number,
  boxH: number,
  size: number,
): Float32Array {
  'worklet';
  // MiniFASNet expects NHWC (interleaved, channels-last) layout — the standard
  // TFLite format.  Each pixel contributes 3 consecutive floats in BGR order
  // with ImageNet normalisation (the model was trained on OpenCV BGR frames):
  //   out[flatIdx*3 + 0] = (B/255 - 0.406) / 0.225
  //   out[flatIdx*3 + 1] = (G/255 - 0.456) / 0.224
  //   out[flatIdx*3 + 2] = (R/255 - 0.485) / 0.229
  //
  // Why NHWC and NOT NCHW:
  //   Device testing showed:
  //     NHWC + index 1: live≈0.024 → implies index 0≈0.976 (real class)
  //     NCHW + index 0: live≈0.076 (still low — NCHW is the wrong layout)
  //   NHWC is the native TFLite tensor format; the NCHW path gave consistently
  //   low scores for BOTH classes, confirming the model never received valid input.
  const out = new Float32Array(size * size * 3);
  const sx0 = boxX * srcW;
  const sy0 = boxY * srcH;
  const sw  = boxW  * srcW;
  const sh  = boxH  * srcH;
  for (let y = 0; y < size; y++) {
    const fy = Math.min(srcH - 1, Math.floor(sy0 + (y / size) * sh));
    for (let x = 0; x < size; x++) {
      const fx      = Math.min(srcW - 1, Math.floor(sx0 + (x / size) * sw));
      const si      = (fy * srcW + fx) * 3;
      const di      = (y * size + x) * 3;   // NHWC: interleaved per pixel
      // RGB source → BGR channel order + ImageNet normalisation
      const r = (src[si]     ?? 0) / 255;
      const g = (src[si + 1] ?? 0) / 255;
      const b = (src[si + 2] ?? 0) / 255;
      out[di]     = (b - 0.406) / 0.225;   // B
      out[di + 1] = (g - 0.456) / 0.224;   // G
      out[di + 2] = (r - 0.485) / 0.229;   // R
    }
  }
  return out;
}

/** Same as wResample01 but normalises to [-1, 1] — required by FaceNet. */
function wResampleM1P1(
  src: Uint8Array,
  srcW: number,
  srcH: number,
  boxX: number,
  boxY: number,
  boxW: number,
  boxH: number,
  size: number,
): Float32Array {
  'worklet';
  const out = new Float32Array(size * size * 3);
  const sx0 = boxX * srcW;
  const sy0 = boxY * srcH;
  const sw = boxW * srcW;
  const sh = boxH * srcH;
  for (let y = 0; y < size; y++) {
    const fy = Math.min(srcH - 1, Math.floor(sy0 + (y / size) * sh));
    for (let x = 0; x < size; x++) {
      const fx = Math.min(srcW - 1, Math.floor(sx0 + (x / size) * sw));
      const si = (fy * srcW + fx) * 3;
      const di = (y * size + x) * 3;
      out[di]     = (src[si]     ?? 0) / 127.5 - 1;
      out[di + 1] = (src[si + 1] ?? 0) / 127.5 - 1;
      out[di + 2] = (src[si + 2] ?? 0) / 127.5 - 1;
    }
  }
  return out;
}

/** Expand box by `factor` on each side — mirrors detector.ts expandBox. */
function wExpandBox(
  x: number, y: number, w: number, h: number, factor: number,
): [number, number, number, number] {
  'worklet';
  const dx = w * factor;
  const dy = h * factor;
  const nx = Math.max(0, x - dx);
  const ny = Math.max(0, y - dy);
  return [nx, ny, Math.min(1 - nx, w + 2 * dx), Math.min(1 - ny, h + 2 * dy)];
}

/** Scale box about its center — mirrors liveness.ts scaleBox. */
function wScaleBox(
  x: number, y: number, w: number, h: number, scale: number,
): [number, number, number, number] {
  'worklet';
  const cx = x + w / 2;
  const cy = y + h / 2;
  const nw = Math.min(1, w * scale);
  const nh = Math.min(1, h * scale);
  return [
    Math.max(0, Math.min(1 - nw, cx - nw / 2)),
    Math.max(0, Math.min(1 - nh, cy - nh / 2)),
    nw,
    nh,
  ];
}

/**
 * Decode MediaPipe BlazeFace short-range model output (two-tensor SSD format).
 *
 * reg  = bfRaw[0]: Float32Array length 896×16 — anchor-relative regression
 *        offsets per anchor i: [cx_off, cy_off, w_off, h_off, kp0x, kp0y, …].
 *        Scale factor = 128 (= model input size). Offsets CAN be negative.
 * cls  = bfRaw[1]: Float32Array length 896 — raw classification logits.
 * anchorCx / anchorCy: pre-built module-level anchor centre tables (0-1 range).
 *
 * Returns [x, y, w, h, score] in normalised 0-1 coords (top-left origin), or null.
 */
function wDecodeBlazeFace(
  reg: Float32Array | Int32Array | Uint8Array,
  cls: Float32Array | Int32Array | Uint8Array,
  anchorCx: number[],
  anchorCy: number[],
  minScore: number,
): [number, number, number, number, number] | null {
  'worklet';
  const r = reg as Float32Array;
  const s = cls as Float32Array;
  let bestScore = -1;
  let bx = 0, by = 0, bw = 0, bh = 0;
  for (let i = 0; i < s.length; i++) {
    const score = wSigmoid(s[i] ?? 0);
    if (score < minScore || score <= bestScore) continue;
    const base = i * 16;
    // Decode anchor-relative offsets → absolute normalised coords.
    const cx = (anchorCx[i] ?? 0) + (r[base]     ?? 0) / 128;
    const cy = (anchorCy[i] ?? 0) + (r[base + 1] ?? 0) / 128;
    const w  = Math.exp((r[base + 2] ?? 0) / 128);
    const h  = Math.exp((r[base + 3] ?? 0) / 128);
    bestScore = score;
    bx = cx - w / 2;
    by = cy - h / 2;
    bw = w;
    bh = h;
  }
  if (bestScore < 0) return null;
  return [bx, by, bw, bh, bestScore];
}

/**
 * Decode FaceMesh output (x, y, z triples in model-pixel space) to a flat
 * [x0,y0, x1,y1, …] array in normalised frame coordinates.
 */
function wDecodeFaceMesh(
  raw: Float32Array | Int32Array | Uint8Array,
  cropX: number, cropY: number, cropW: number, cropH: number,
): number[] {
  'worklet';
  const r = raw as Float32Array;
  const pts: number[] = [];
  for (let i = 0; i + 3 <= r.length; i += 3) {
    pts.push(cropX + ((r[i]     ?? 0) / FM) * cropW);
    pts.push(cropY + ((r[i + 1] ?? 0) / FM) * cropH);
  }
  return pts;
}

/**
 * MiniFASNet output class ordering for bundled spoof_2_7.tflite / spoof_4_0.tflite:
 *   [fake_2d, fake_3d, real]  — index 2 is the live/real class.
 * Confirmed by device logits: index 0 ≈ -3.17, index 1 ≈ -0.74, index 2 ≈ +3.91 for a real face.
 *
 * Detects whether softmax is already baked into the TFLite graph by checking
 * if the values sum to ~1.0.  If so, returns index 2 directly — applying
 * softmax again would heavily squash the probabilities and destroy accuracy.
 * Falls back to manual softmax when raw logits are present.
 */
function wDecodeLiveness(raw: Float32Array | Int32Array | Uint8Array): number {
  'worklet';
  const r = raw as Float32Array;
  if (r.length < 3) return 0;

  // If the model already applied softmax, values sum to ~1.0.
  const sumCheck = (r[0] ?? 0) + (r[1] ?? 0) + (r[2] ?? 0);
  if (Math.abs(sumCheck - 1.0) < 0.05) {
    return r[2] ?? 0;  // index 2 = real/live
  }

  // Raw logits path: apply softmax manually (numerically stable via max subtraction).
  let max = -1e9;
  for (let i = 0; i < r.length; i++) max = Math.max(max, r[i] ?? -1e9);
  let sum = 0;
  const exps: number[] = [];
  for (let i = 0; i < r.length; i++) {
    const e = Math.exp((r[i] ?? 0) - max);
    exps.push(e);
    sum += e;
  }
  return sum > 0 ? (exps[2] ?? 0) / sum : 0;  // index 2 = real/live
}

/**
 * DEBUG — convert a typed array to a plain number[] so it can be passed
 * safely through useRunOnJS. Also rounds to 4 decimal places for readability.
 */
function wTypedToArray(raw: Float32Array | Int32Array | Uint8Array): number[] {
  'worklet';
  const r = raw as Float32Array;
  const out: number[] = new Array(r.length);
  for (let i = 0; i < r.length; i++) {
    out[i] = Math.round((r[i] ?? 0) * 10000) / 10000;
  }
  return out;
}

/**
 * L2-normalise a Float32Array and return a plain number[] (safe for useRunOnJS).
 */
function wL2NormalizeToArray(raw: Float32Array | Int32Array | Uint8Array): number[] {
  'worklet';
  const v = raw as Float32Array;
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += (v[i] ?? 0) ** 2;
  norm = Math.sqrt(norm) || 1;
  const out: number[] = new Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = (v[i] ?? 0) / norm;
  return out;
}

/**
 * Mean luma (0–255) over the face-bbox region. Sampled on a grid (~64 pts).
 */
function wMeanLuma(
  src: Uint8Array,
  srcW: number, srcH: number,
  boxX: number, boxY: number, boxW: number, boxH: number,
): number {
  'worklet';
  const x0 = Math.floor(boxX * srcW);
  const y0 = Math.floor(boxY * srcH);
  const w  = Math.max(1, Math.floor(boxW * srcW));
  const h  = Math.max(1, Math.floor(boxH * srcH));
  const step = Math.max(1, Math.floor(Math.min(w, h) / 8));
  let sum = 0, count = 0;
  for (let y = y0; y < y0 + h; y += step) {
    for (let x = x0; x < x0 + w; x += step) {
      const cx = Math.min(srcW - 1, x);
      const cy = Math.min(srcH - 1, y);
      const i  = (cy * srcW + cx) * 3;
      sum += 0.299 * (src[i] ?? 0) + 0.587 * (src[i + 1] ?? 0) + 0.114 * (src[i + 2] ?? 0);
      count += 1;
    }
  }
  return count === 0 ? 128 : sum / count;
}

// ─── worklet BMP encoder ──────────────────────────────────────────────────────

/** Base64 alphabet — module-level constant, captured by the worklet closure. */
const B64C = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

/**
 * Encode a Uint8Array as a base64 string.
 * Written without array destructuring or other Babel-transformed syntax so it
 * can safely run inside a react-native-worklets-core worklet.
 */
function wBytesToBase64(bytes: Uint8Array): string {
  'worklet';
  let out = '';
  const len = bytes.length;
  for (let i = 0; i < len; i += 3) {
    const b0 = bytes[i]     ?? 0;
    const b1 = i + 1 < len ? (bytes[i + 1] ?? 0) : 0;
    const b2 = i + 2 < len ? (bytes[i + 2] ?? 0) : 0;
    out += B64C[b0 >> 2]                        ?? '';
    out += B64C[((b0 & 3) << 4) | (b1 >> 4)]   ?? '';
    out += i + 1 < len ? (B64C[((b1 & 15) << 2) | (b2 >> 6)] ?? '') : '=';
    out += i + 2 < len ? (B64C[b2 & 63]         ?? '') : '=';
  }
  return out;
}

/**
 * Encode a Float32Array (values 0-1, HWC RGB layout) as a 24-bit BMP base64.
 * The resulting string can be passed through useRunOnJS and saved by RNFS.
 *
 * Row stride for `size × size` RGB: size×3 bytes.
 * 128×128 → ~65 KB base64.  192×192 → ~147 KB base64.
 */
function wFloat32ToBmpBase64(f32: Float32Array, size: number): string {
  'worklet';
  const rowBytes   = size * 3;
  const pad        = (4 - (rowBytes % 4)) % 4;   // align rows to 4 bytes
  const pixData    = (rowBytes + pad) * size;
  const fileSize   = 54 + pixData;
  const buf        = new Uint8Array(fileSize);
  let o = 0;

  // little-endian writers
  const w16 = (v: number) => { buf[o++] = v & 0xff; buf[o++] = (v >> 8) & 0xff; };
  const w32 = (v: number) => {
    buf[o++] = v & 0xff;
    buf[o++] = (v >> 8)  & 0xff;
    buf[o++] = (v >> 16) & 0xff;
    buf[o++] = (v >> 24) & 0xff;
  };

  // ── BITMAPFILEHEADER (14 bytes) ──
  buf[o++] = 0x42; buf[o++] = 0x4d; // 'BM'
  w32(fileSize);   // total file size
  w32(0);          // reserved
  w32(54);         // pixel data offset (14 + 40)

  // ── BITMAPINFOHEADER (40 bytes) ──
  w32(40);            // header size
  w32(size);          // width
  w32(-size >>> 0);   // negative height → top-to-bottom rows
  w16(1);             // color planes
  w16(24);            // bits per pixel
  w32(0);             // compression = BI_RGB
  w32(pixData);       // pixel data size
  w32(2835); w32(2835); // ~72 DPI
  w32(0); w32(0);     // clr table / important

  // ── Pixel data (BMP = BGR byte order) ──
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 3;
      const r = Math.min(255, Math.max(0, Math.round((f32[i]     ?? 0) * 255)));
      const g = Math.min(255, Math.max(0, Math.round((f32[i + 1] ?? 0) * 255)));
      const b = Math.min(255, Math.max(0, Math.round((f32[i + 2] ?? 0) * 255)));
      buf[o++] = b; buf[o++] = g; buf[o++] = r;
    }
    for (let p = 0; p < pad; p++) buf[o++] = 0;
  }

  return wBytesToBase64(buf);
}

/**
 * Nearest-neighbour downsample a uint8 RGB buffer (srcW×srcH×3) to `size×size`
 * and encode as a 24-bit BMP base64 string.
 *
 * This gives you the exact pixels the resize plugin produced — no float
 * conversion — so the saved image is byte-identical to what the worklet sees
 * before model preprocessing.
 *
 * 160×160 output → ~100 KB base64 string (manageable for useRunOnJS).
 */
function wUint8ToBmpBase64(src: Uint8Array, srcW: number, srcH: number, size: number): string {
  'worklet';
  const rowBytes = size * 3;
  const pad      = (4 - (rowBytes % 4)) % 4;
  const pixData  = (rowBytes + pad) * size;
  const fileSize = 54 + pixData;
  const buf      = new Uint8Array(fileSize);
  let o = 0;

  const w16 = (v: number) => { buf[o++] = v & 0xff; buf[o++] = (v >> 8) & 0xff; };
  const w32 = (v: number) => {
    buf[o++] = v & 0xff;
    buf[o++] = (v >> 8)  & 0xff;
    buf[o++] = (v >> 16) & 0xff;
    buf[o++] = (v >> 24) & 0xff;
  };

  // BITMAPFILEHEADER (14 bytes)
  buf[o++] = 0x42; buf[o++] = 0x4d;
  w32(fileSize); w32(0); w32(54);

  // BITMAPINFOHEADER (40 bytes)
  w32(40); w32(size); w32(-size >>> 0);
  w16(1); w16(24); w32(0); w32(pixData);
  w32(2835); w32(2835); w32(0); w32(0);

  // Pixel rows — nearest-neighbour resample, BMP = BGR byte order
  for (let y = 0; y < size; y++) {
    const fy = Math.min(srcH - 1, Math.floor((y / size) * srcH));
    for (let x = 0; x < size; x++) {
      const fx = Math.min(srcW - 1, Math.floor((x / size) * srcW));
      const si = (fy * srcW + fx) * 3;
      buf[o++] = src[si + 2] ?? 0;  // B
      buf[o++] = src[si + 1] ?? 0;  // G
      buf[o++] = src[si]     ?? 0;  // R
    }
    for (let p = 0; p < pad; p++) buf[o++] = 0;
  }

  return wBytesToBase64(buf);
}

// ─── hook ─────────────────────────────────────────────────────────────────────

export function useFrameSource({
  models,
  onFrameResult,
  minIntervalMs = 200,
  needsLiveness,
  needsEmbedding,
  onModelDebug,
  onCentering,
  centerMargin = 0.10,
}: FrameSourceOptions) {
  const { resize } = useResizePlugin();

  const isBusy      = useSharedValue(false);
  const lastRunTime = useSharedValue(0);

  /** Set to true by triggerDebugCapture(); worklet captures once then resets. */
  const captureNext = useSharedValue(false);

  // ── Centering signal SharedValues (worklet-thread state, no JS crossing) ────
  /** null-encoded as -1 (SharedValue doesn't support null). 0=false, 1=true. */
  const lastCenteredSV  = useSharedValue(-1);   // -1 = no face
  /** Last cx/cy sent to JS — used to suppress redundant calls. */
  const lastArrowCxSV   = useSharedValue(-1);
  const lastArrowCySV   = useSharedValue(-1);
  /** Centering margin — updated when prop changes (rare). */
  const marginSV        = useSharedValue(centerMargin);
  marginSV.value = centerMargin;                // keep in sync if config changes

  const cbRef = useRef(onFrameResult);
  cbRef.current = onFrameResult;

  const dbgRef = useRef(onModelDebug);
  dbgRef.current = onModelDebug;

  const centeringRef = useRef(onCentering);
  centeringRef.current = onCentering;

  // JS-thread receiver: accepts a plain WorkletFrameResult (no binary blobs).
  const deliver = useRunOnJS(
    async (result: WorkletFrameResult) => {
      try {
        await cbRef.current(result);
      } finally {
        isBusy.value = false;
      }
    },
    [isBusy],
  );

  // JS-thread receiver for debug model frame (one-shot, fired by captureNext).
  const deliverModelDebug = useRunOnJS(
    (frame: ModelDebugFrame) => {
      dbgRef.current?.(frame);
    },
    [],
  );

  // JS-thread receiver for centering status.
  // Fires only when status or arrow direction actually changes — never per-frame.
  const deliverCentering = useRunOnJS(
    (status: CenteringStatus) => {
      centeringRef.current?.(status);
    },
    [],
  );

  // Destructure so each JSI HostObject is individually captured by the worklet.
  const { blazeface, faceMesh, liveness0, liveness1, embedding: embModel } = models;

  const frameProcessor = useFrameProcessor(
    (frame: Frame) => {
      'worklet';

      if (isBusy.value) return;
      const now = Date.now();
      if (now - lastRunTime.value < minIntervalMs) return;
      isBusy.value    = true;
      lastRunTime.value = now;

      // ── 1. Single resize to 256×256 uint8 (one native call) ───────────────
      const src = resize(frame, {
        scale: { width: FRAME_SIZE, height: FRAME_SIZE },
        pixelFormat: 'rgb',
        dataType: 'uint8',
      }) as unknown as Uint8Array;

      // ── 2. BlazeFace: full frame 128×128, [0,1] ────────────────────────────
      // NOTE on aspect ratio: the resize plugin squishes the camera frame to a
      // square. Correcting for this requires knowing the *pixel buffer*
      // orientation, which differs from frame.width/frame.height (those report
      // the raw sensor dimensions, not the display-rotated pixel layout).
      // We use the full 480×480 buffer here and rely on the debug images
      // (triggerDebugCapture) to determine the actual pixel orientation before
      // applying any crop correction.
      const bfInput = wResample01(src, FRAME_SIZE, FRAME_SIZE, 0, 0, 1, 1, BF);
      const bfRaw   = blazeface.runSync([bfInput]);
      // bfRaw[0] = 896×16 anchor-relative regression offsets
      // bfRaw[1] = 896×1  classification logits
      const bbox    = wDecodeBlazeFace(
        bfRaw[0] as Float32Array,
        bfRaw[1] as Float32Array,
        BF_ANCHOR_CX,
        BF_ANCHOR_CY,
        0.4,   // lowered from 0.5 — better recall on slightly tilted faces
      );

      // ── 3. No-face branch ─────────────────────────────────────────────────
      if (!bbox) {
        // Signal centering loss only on transition (was detected → now gone).
        if (lastCenteredSV.value !== -1) {
          lastCenteredSV.value  = -1;
          lastArrowCxSV.value   = -1;
          lastArrowCySV.value   = -1;
          deliverCentering({ detected: false });
        }
        // Debug capture even without a face — helps diagnose detection failures.
        if (captureNext.value) {
          captureNext.value = false;
          const srcFrameBmp  = wUint8ToBmpBase64(src, FRAME_SIZE, FRAME_SIZE, 160);
          const blazefaceBmp = wFloat32ToBmpBase64(bfInput, BF);
          deliverModelDebug({ srcFrameBmp, blazefaceBmp, faceCropBmp: '', bboxNorm: null });
        }
        deliver({ detected: false, frameW: frame.width, frameH: frame.height });
        return;
      }
      // Avoid array destructuring (_slicedToArray is not worklet-compatible).
      const bx     = bbox[0];
      const by     = bbox[1];
      const bw     = bbox[2];
      const bh     = bbox[3];
      const bScore = bbox[4];

      // ── 3b. Centering signal (worklet-thread only, JS called on change) ────
      // Keeps the JS / main thread free from per-frame work.
      const fcx = bx + bw / 2;
      const fcy = by + bh / 2;
      const m   = marginSV.value;
      const centered = fcx > m && fcx < (1 - m) && fcy > m && fcy < (1 - m);
      const centeredInt = centered ? 1 : 0;  // SharedValue stores number, not bool

      // Arrow direction: only re-signal if |Δcx| or |Δcy| > 0.05
      const arrowDirChanged =
        !centered &&
        (Math.abs(fcx - lastArrowCxSV.value) > 0.05 ||
         Math.abs(fcy - lastArrowCySV.value) > 0.05);

      if (centeredInt !== lastCenteredSV.value || arrowDirChanged) {
        lastCenteredSV.value  = centeredInt;
        lastArrowCxSV.value   = fcx;
        lastArrowCySV.value   = fcy;
        deliverCentering({ detected: true, centered, cx: fcx, cy: fcy });
      }

      // ── 4. Brightness (cheap grid sample over face region) ─────────────────
      const brightness = wMeanLuma(src, FRAME_SIZE, FRAME_SIZE, bx, by, bw, bh);

      // ── 5. FaceMesh: expanded bbox, 192×192, [0,1] ────────────────────────
      const exp     = wExpandBox(bx, by, bw, bh, 0.25);
      const ex = exp[0]; const ey = exp[1]; const ew = exp[2]; const eh = exp[3];
      const fmInput    = wResample01(src, FRAME_SIZE, FRAME_SIZE, ex, ey, ew, eh, FM);
      const fmRaw      = faceMesh.runSync([fmInput]);
      const landmarks  = wDecodeFaceMesh(fmRaw[0] as Float32Array, ex, ey, ew, eh);

      // ── 6. Liveness: two MiniFASNet branches — only in 'analyzing' stage ────
      // Gated by needsLiveness.value so the two ~40 ms MiniFASNet calls never
      // run during gating or challenge frames.
      // Square crop (max(w,h)) avoids aspect-ratio distortion.
      // wResampleLiveness applies BGR channel swap + ImageNet normalisation.
      let livenessScore = 0;
      const livenessRaw: number[] = [];   // debug raw logits, empty when skipped

      if (needsLiveness.value) {
        const faceSize = Math.max(bw, bh);
        const faceCx   = bx + bw / 2;
        const faceCy   = by + bh / 2;
        const sqX      = Math.max(0, faceCx - faceSize / 2);
        const sqY      = Math.max(0, faceCy - faceSize / 2);
        const sqW      = Math.min(1 - sqX, faceSize);
        const sqH      = Math.min(1 - sqY, faceSize);
        // Scale directly from the raw square — no pre-expansion.
        // The scale factor (2.7× / 4.0×) already controls context inclusion;
        // expanding the box first would double-count padding and push the crop
        // far beyond the face, destroying model accuracy.
        const sc0      = wScaleBox(sqX, sqY, sqW, sqH, 2.7);
        const lv0Input = wResampleLiveness(src, FRAME_SIZE, FRAME_SIZE, sc0[0], sc0[1], sc0[2], sc0[3], LV);
        const lv0Raw   = liveness0.runSync([lv0Input]);
        const live0    = wDecodeLiveness(lv0Raw[0] as Float32Array);
        // DEBUG: raw softmax values of branch 0 (all indices, before selection)
        const lv0Debug = wTypedToArray(lv0Raw[0] as Float32Array);

        const sc1      = wScaleBox(sqX, sqY, sqW, sqH, 4.0);
        const lv1Input = wResampleLiveness(src, FRAME_SIZE, FRAME_SIZE, sc1[0], sc1[1], sc1[2], sc1[3], LV);
        const lv1Raw   = liveness1.runSync([lv1Input]);
        const live1    = wDecodeLiveness(lv1Raw[0] as Float32Array);
        // DEBUG: raw softmax values of branch 1
        const lv1Debug = wTypedToArray(lv1Raw[0] as Float32Array);

        livenessScore = (live0 + live1) / 2;
        // Pack both branches separated by -1 sentinel for JS-thread logging.
        // No spread operator — Babel compiles spread to _toConsumableArray which
        // is a regular function and crashes the worklet.
        for (let i = 0; i < lv0Debug.length; i++) livenessRaw.push(lv0Debug[i]!);
        livenessRaw.push(-1);
        for (let i = 0; i < lv1Debug.length; i++) livenessRaw.push(lv1Debug[i]!);
      }

      // ── 7. Embedding: only when the analyzing stage requests it ───────────
      let embedding: number[] | null = null;
      if (needsEmbedding.value) {
        const emInput = wResampleM1P1(src, FRAME_SIZE, FRAME_SIZE, ex, ey, ew, eh, EM);
        const emRaw   = embModel.runSync([emInput]);
        embedding     = wL2NormalizeToArray(emRaw[0] as Float32Array);
      }

      // ── 8. Debug capture — FACE-DETECTED branch ───────────────────────────
      // Three images saved per trigger:
      //   src_frame_*.bmp  — 160×160 full frame (resize-plugin uint8 output)
      //   bf_input_*.bmp   — 128×128 BlazeFace float input
      //   face_crop_*.bmp  — 192×192 FaceMesh crop float input
      if (captureNext.value) {
        captureNext.value = false;
        const srcFrameBmp  = wUint8ToBmpBase64(src, FRAME_SIZE, FRAME_SIZE, 160);
        const blazefaceBmp = wFloat32ToBmpBase64(bfInput, BF);
        const faceCropBmp  = wFloat32ToBmpBase64(fmInput, FM);
        deliverModelDebug({ srcFrameBmp, blazefaceBmp, faceCropBmp, bboxNorm: [bx, by, bw, bh] });
      }

      deliver({
        detected: true,
        bbox: [bx, by, bw, bh, bScore],
        landmarks,
        brightness,
        livenessScore,
        livenessRaw,
        embedding,
        frameW: frame.width,
        frameH: frame.height,
      });
    },
    [resize, deliver, isBusy, lastRunTime, needsLiveness, needsEmbedding, captureNext,
     deliverModelDebug, blazeface, faceMesh, liveness0, liveness1, embModel,
     BF_ANCHOR_CX, BF_ANCHOR_CY,
     deliverCentering, lastCenteredSV, lastArrowCxSV, lastArrowCySV, marginSV],
  );

  /**
   * Call this (e.g. from a 5-second timer) to request a snapshot of the exact
   * buffers the TFLite models receive on the next worklet frame.
   * The result arrives via `onModelDebug` as three BMP base64 strings:
   *   src_frame  — 160×160 full resize-plugin output
   *   bf_input   — 128×128 BlazeFace input
   *   face_crop  — 192×192 FaceMesh crop (only when face detected)
   *
   * Wrapped in useCallback so the reference is stable across renders.
   * This is critical: FaceAuthView's debug useEffect lists this in its deps;
   * an unstable reference would restart the timer on every render (causing the
   * "Periodic capture stopped — 1 frames" noise seen in logcat).
   */
  const triggerDebugCapture = useCallback(() => {
    captureNext.value = true;
  }, [captureNext]);

  return { frameProcessor, triggerDebugCapture };
}
