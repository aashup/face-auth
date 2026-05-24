/**
 * Debug utilities for frame inspection and score analysis.
 *
 * Saved frames go to:
 *   /storage/emulated/0/Android/data/<package>/files/faceauth_debug/
 *
 * On the device, open:
 *   Files app → Internal Storage → Android → data → com.faceauthexample → files → faceauth_debug
 *
 * No extra permissions needed on Android 10+ (app-specific external storage).
 *
 * Only active when __DEV__ is true — stripped from release builds.
 */

import type React from 'react';
import type { Camera } from 'react-native-vision-camera';
import type { ModelDebugFrame } from '../frameSource';
import { log, warn } from '../logger';

// react-native-fs is an optional dep (only used in __DEV__).
// Dynamic require keeps it out of release bundles and avoids a hard peerDep.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let RNFS: any = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  RNFS = require('react-native-fs');
} catch {
  // Library not installed — file saving disabled, score logging still works.
}

// ─── types ────────────────────────────────────────────────────────────────────

/** All scores available at JS-thread time after each processed frame. */
export interface DebugScores {
  /** Date.now() of the frame */
  ts: number;
  /** Current pipeline stage */
  stage: string;
  /** BlazeFace detection confidence 0→1 */
  detectionScore: number;
  /** Mean face-region luma 0→255 */
  brightness: number;
  /** Averaged MiniFASNet live probability 0→1  (want > 0.5 to pass) */
  livenessScore: number;
  /** Head yaw in degrees  (negative = left, positive = right) */
  yaw: number;
  /** Head pitch in degrees */
  pitch: number;
  /** Normalised face bbox [x, y, w, h] */
  bbox: [number, number, number, number];
  /** Embedding vector length (0 = not computed this frame) */
  embeddingDims: number;
}

// ─── score logger ─────────────────────────────────────────────────────────────

/**
 * Pretty-print all pipeline scores to the Metro console.
 * Filter in logcat:  adb logcat | grep "\[DBG\]"
 *
 * Example:
 *   [DBG] 14:32:05.123 | stage=gating    | detect=0.972 | bright=148 | live=0.831 ✓live | yaw= -2.1° pitch=  3.4°
 */
export function logDetailedScores(s: DebugScores): void {
  if (!__DEV__) return;

  const time    = new Date(s.ts).toISOString().slice(11, 23);
  const liveTag = s.livenessScore > 0.5 ? '✓live' : '✗SPOOF';
  const embed   = s.embeddingDims > 0 ? ` | embed=${s.embeddingDims}d` : '';
  const yawStr  = (s.yaw  >= 0 ? ' ' : '') + s.yaw.toFixed(1);
  const pitStr  = (s.pitch >= 0 ? ' ' : '') + s.pitch.toFixed(1);

  log(
    `[DBG] ${time}` +
    ` | stage=${s.stage.padEnd(9)}` +
    ` | detect=${s.detectionScore.toFixed(3)}` +
    ` | bright=${String(Math.round(s.brightness)).padStart(3)}/255` +
    ` | live=${s.livenessScore.toFixed(3)} ${liveTag}` +
    ` | yaw=${yawStr}° pitch=${pitStr}°` +
    ` | bbox=[${s.bbox.map(v => v.toFixed(2)).join(',')}]` +
    embed,
  );
}

// ─── periodic photo capture ───────────────────────────────────────────────────

let _captureTimer: ReturnType<typeof setInterval> | null = null;
let _captureCount = 0;
let _debugDir: string | null = null;

/** Returns (and lazily creates) the on-device debug folder. */
async function _ensureDebugDir(): Promise<string | null> {
  if (!RNFS) return null;
  if (_debugDir) return _debugDir;
  try {
    // ExternalDirectoryPath = /storage/emulated/0/Android/data/<pkg>/files
    // Visible in Files app: no permission required on Android 10+.
    const base = RNFS.ExternalDirectoryPath as string;
    const dir  = `${base}/faceauth_debug`;
    await RNFS.mkdir(dir);
    _debugDir = dir;
    log(`[DBG] Debug folder ready: ${dir}`);
    log(`[DBG] On device: Files → Internal Storage → Android → data → com.faceauthexample → files → faceauth_debug`);
    return dir;
  } catch (e: any) {
    log('[DBG] Could not create debug dir:', e?.message);
    return null;
  }
}

/**
 * Take a full-resolution JPEG and copy it to the visible debug folder.
 * The file will appear in the device Files app immediately.
 */
async function _takeAndSavePhoto(ref: React.RefObject<Camera>): Promise<void> {
  try {
    const cam = ref.current;
    if (!cam) {
      warn('[DBG] capturePhoto: camera ref is null');
      return;
    }

    const photo = await cam.takePhoto({ flash: 'off' });
    _captureCount += 1;

    if (!RNFS) {
      // No RNFS — just log the cache path so adb pull still works.
      log(`[DBG] 📸 Frame #${_captureCount} (cache) → ${photo.path}`);
      log(`[DBG]    react-native-fs not found; install it to auto-save to Files app.`);
      return;
    }

    const dir = await _ensureDebugDir();
    if (!dir) return;

    // Timestamp-based filename so frames sort chronologically.
    const ts   = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const dest = `${dir}/frame_${ts}_${_captureCount}.jpg`;

    await RNFS.copyFile(photo.path, dest);
  } catch {
    // Silently swallow — non-fatal debug helper
  }
}

/**
 * Start saving a camera frame every `intervalMs` milliseconds (default 5 s).
 * Frames are saved as JPEG to the faceauth_debug folder visible in the Files app.
 *
 * Call `stopPeriodicCapture()` when the auth view unmounts.
 */
export function startPeriodicCapture(
  ref: React.RefObject<Camera>,
  intervalMs = 5_000,
): void {
  if (!__DEV__) return;
  stopPeriodicCapture();
  _captureCount = 0;

  // Capture one immediately — don't wait for the first interval tick.
  void _takeAndSavePhoto(ref);
  _captureTimer = setInterval(() => void _takeAndSavePhoto(ref), intervalMs);
}

/**
 * Stop the periodic frame capture (no-op if not running).
 */
export function stopPeriodicCapture(): void {
  if (_captureTimer !== null) {
    clearInterval(_captureTimer);
    _captureTimer = null;
    if (_captureCount > 0) {
      log(`[DBG] Periodic capture stopped — ${_captureCount} frames in faceauth_debug/`);
    }
  }
}

// ─── model input frame saver ─────────────────────────────────────────────────

let _modelFrameCount = 0;

/**
 * Save the exact images the TFLite models receive as BMP files.
 *
 * Three files are written per call:
 *   src_frame_<ts>_N.bmp  — 160×160  full frame as the resize plugin outputs it (uint8 RGB)
 *   bf_input_<ts>_N.bmp   — 128×128  full-frame as BlazeFace sees it (float → uint8)
 *   face_crop_<ts>_N.bmp  — 192×192  expanded face region as FaceMesh sees it (omitted if no face)
 *
 * src_frame is the most useful for debugging: it shows the actual pixels the
 * worklet processes before any model-specific cropping or normalisation.
 *
 * Find them at:
 *   Files → Internal Storage → Android → data → com.faceauthexample → files → faceauth_debug
 */
export async function saveModelDebugFrame(frame: ModelDebugFrame): Promise<void> {
  if (!__DEV__) return;

  _modelFrameCount += 1;
  const n  = _modelFrameCount;
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

  if (frame.bboxNorm) {
    const bx = frame.bboxNorm[0].toFixed(2);
    const by = frame.bboxNorm[1].toFixed(2);
    const bw = frame.bboxNorm[2].toFixed(2);
    const bh = frame.bboxNorm[3].toFixed(2);
    log(`[DBG] 🔬 Model inputs #${n} | face bbox: x=${bx} y=${by} w=${bw} h=${bh}`);
  } else {
    log(`[DBG] 🔬 Model inputs #${n} | no face detected — saving src + bf frames`);
  }

  const dir = await _ensureDebugDir();
  if (!dir || !RNFS) {
    log('[DBG]    (RNFS not available — install react-native-fs to save files)');
    return;
  }

  const srcFile  = `${dir}/src_frame_${ts}_${n}.bmp`;
  const bfFile   = `${dir}/bf_input_${ts}_${n}.bmp`;
  const cropFile = `${dir}/face_crop_${ts}_${n}.bmp`;

  // 1. Full resized frame — always present
  try {
    await RNFS.writeFile(srcFile, frame.srcFrameBmp, 'base64');
    log(`[DBG]    Full frame      (160×160) → src_frame_${ts}_${n}.bmp`);
  } catch (e: any) {
    warn('[DBG] save src_frame failed:', e?.message);
  }

  // 2. BlazeFace input — always present
  try {
    await RNFS.writeFile(bfFile, frame.blazefaceBmp, 'base64');
    log(`[DBG]    BlazeFace input (128×128) → bf_input_${ts}_${n}.bmp`);
  } catch (e: any) {
    warn('[DBG] save bf_input failed:', e?.message);
  }

  // 3. FaceMesh crop — only when a face was detected
  if (frame.faceCropBmp) {
    try {
      await RNFS.writeFile(cropFile, frame.faceCropBmp, 'base64');
      log(`[DBG]    Face crop       (192×192) → face_crop_${ts}_${n}.bmp`);
    } catch (e: any) {
      warn('[DBG] save face_crop failed:', e?.message);
    }
  }

  log('[DBG]    Open: Files → Android → data → com.faceauthexample → files → faceauth_debug');
}
