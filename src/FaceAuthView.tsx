import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useCameraFormat,
} from 'react-native-vision-camera';
import { useSharedValue } from 'react-native-worklets-core';

import { estimateHeadPose } from './detector';
import { useFrameSource, type FrameSourceModels, type WorkletFrameResult, type CenteringStatus } from './frameSource';
import { getConfig } from './runtime';
import { resolveThresholds } from './config';
import { useFaceAuth } from './useFaceAuth';
import type { PipelineModels } from './pipeline';
import type { AuthMode, AuthResult, Challenge, DetectedFace, Guidance, Point } from './types';
import { FaceMaskOverlay } from './FaceMaskOverlay';
import { saveModelDebugFrame } from './debug';
import { log } from './logger';

/** Big, glanceable instruction for the current challenge / state. */
const CHALLENGE_PROMPT: Record<Challenge, { icon: string; title: string }> = {
  blink: { icon: '😉', title: 'Blink your eyes' },
  smile: { icon: '😊', title: 'Smile' },
  turn_head: { icon: '↔️', title: 'Turn your head' },
};

/** All TFLite models the view needs — matches FrameSourceModels exactly. */
export type FaceAuthModels = FrameSourceModels;

export interface FaceAuthViewProps {
  mode: AuthMode;
  /** Required for `verify` (1:1). */
  personnelId?: string;
  /** TFLite models loaded by the host (e.g. via fast-tflite `useTensorflowModel`). */
  models: FaceAuthModels;
  /** Auth completed (success or failure). Attendance is recorded automatically. */
  onResult?: (result: AuthResult) => void;
  /** Live coaching messages (centering, lighting, challenge prompts). */
  onGuidance?: (g: Guidance) => void;
  /** Back/cancel pressed — host should dismiss the view. */
  onClose?: () => void;
  /** Auto-start the session on mount. Default true. */
  autoStart?: boolean;
  /** Optional style override for the camera container. */
  style?: object;
}

// ─── JS-thread model decode helpers ──────────────────────────────────────────
// Mirror the worklet functions (wDecodeLiveness, wL2NormalizeToArray) but run
// on the JS thread after react-native-fast-tflite's async .run() resolves.

type Tensor = Float32Array | Int32Array | Uint8Array;

/**
 * Decode MiniFASNet output → live-face probability (0 = spoof, 1 = live).
 * Class ordering for bundled spoof_2_7 / spoof_4_0: [fake_2d, fake_3d, real].
 * Index 2 = real/live class.
 * Detects pre-applied softmax (sum ≈ 1) and skips redundant re-normalisation.
 */
function decodeLiveness(raw: Tensor): number {
  const r = raw as Float32Array;
  if (r.length < 3) return 0;
  const sumCheck = (r[0] ?? 0) + (r[1] ?? 0) + (r[2] ?? 0);
  if (Math.abs(sumCheck - 1.0) < 0.05) return r[2] ?? 0; // pre-softmaxed
  // Raw logits → numerically-stable softmax
  let max = -1e9;
  for (let i = 0; i < r.length; i++) max = Math.max(max, r[i] ?? -1e9);
  let sum = 0;
  const exps: number[] = [];
  for (let i = 0; i < r.length; i++) {
    const e = Math.exp((r[i] ?? 0) - max);
    exps.push(e);
    sum += e;
  }
  return sum > 0 ? (exps[2] ?? 0) / sum : 0;
}

/** L2-normalise a model output tensor into a Float32Array. */
function l2NormalizeToFloat32(raw: Tensor): Float32Array {
  const v = raw as Float32Array;
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += (v[i] ?? 0) ** 2;
  norm = Math.sqrt(norm) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = (v[i] ?? 0) / norm;
  return out;
}

/**
 * Camera surface that runs a full authentication session. Each frame is
 * downsampled to RGB in a worklet (frameSource), handed to the JS thread where
 * BlazeFace + FaceMesh detect the face, and fed into the `useFaceAuth`
 * pipeline. Shows guidance and reports the result. No native code.
 *
 * Parallelism: the worklet is intentionally lightweight (BlazeFace + FaceMesh
 * only). MiniFASNet ×2 and FaceNet run on the JS thread via Promise.all,
 * exploiting react-native-fast-tflite's native background thread pool so all
 * three models execute concurrently while the worklet picks up the next frame.
 */
export function FaceAuthView({
  mode,
  personnelId,
  models,
  onResult,
  onGuidance,
  onClose,
  autoStart = true,
  style,
}: FaceAuthViewProps) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');
  const format = useCameraFormat(device, [
    { videoResolution: { width: 720, height: 1280 } },
    { fps: 30 },
  ]);

  // Debug: camera ref enables takePhoto() for periodic frame saves.
  const cameraRef = useRef<Camera>(null);
  const DEBUG_INTERVAL_MS = 5_000;

  const [hint, setHint] = useState<string>('Position your face in the frame');
  const [challenge, setChallenge] = useState<Challenge | undefined>();
  const [verifying, setVerifying] = useState(false);

  // Centering overlay state — driven by worklet thread via onCentering callback.
  const [faceCentered, setFaceCentered] = useState<boolean | null>(null);
  const [facePos, setFacePos] = useState<{ cx: number; cy: number } | null>(null);

  if (mode === 'verify' && !personnelId) {
    throw new Error('[offline-face-auth] verify mode requires a personnelId prop.');
  }

  const pipelineModels: PipelineModels = useMemo(
    () => ({
      embedding: models.embedding,
      liveness: { branches: [models.liveness0, models.liveness1] },
    }),
    [models.embedding, models.liveness0, models.liveness1],
  );

  // Only re-render when the guidance text actually changes (avoids a setState
  // on every processed frame, which thrashes React on the JS thread).
  const lastHintRef = useRef(hint);
  const lastChallengeRef = useRef<Challenge | undefined>(undefined);
  const handleGuidance = useCallback(
    (g: Guidance) => {
      if (g.message !== lastHintRef.current) {
        lastHintRef.current = g.message;
        setHint(g.message);
        setVerifying(g.message === 'Verifying…');
      }
      if (g.challenge !== lastChallengeRef.current) {
        lastChallengeRef.current = g.challenge;
        setChallenge(g.challenge);
      }
      onGuidance?.(g);
    },
    [onGuidance],
  );

  // SharedValues flipped to true when the pipeline enters the 'analyzing' stage.
  // Keeps gating + challenge frames free of the heavy model calls:
  //   needsLiveness — gates the two ~40 ms MiniFASNet branches
  //   needsEmbedding — gates the ~80 ms FaceNet embedding
  const needsLiveness  = useSharedValue(false);
  const needsEmbedding = useSharedValue(false);

  // Wrap handleGuidance: flip both flags when 'Verifying…' appears.
  // 'Verifying…' is the sole message emitted by the analyzing stage.
  // Must be defined BEFORE useFaceAuth so it can be passed as onGuidance.
  const handleGuidanceWithEmbedFlag = useCallback(
    (g: Guidance) => {
      const analyzing = g.message === 'Verifying…';
      needsLiveness.value  = analyzing;
      needsEmbedding.value = analyzing;
      handleGuidance(g);
    },
    [needsLiveness, needsEmbedding, handleGuidance],
  );

  const { start, cancel, feed, active, stageRef } = useFaceAuth({
    mode,
    personnelId,
    models: pipelineModels,
    onResult,
    onGuidance: handleGuidanceWithEmbedFlag,
  });

  // Stable ref so onFrameResult can access the latest models without being
  // recreated on every render (models prop is stable after first load).
  const modelsRef = useRef(models);
  modelsRef.current = models;

  // DEBUG — log liveness scores once per session (first analyzing frame).
  const livenessRawLoggedRef = useRef(false);
  // DEBUG — log camera sensor dimensions once.
  const frameDimsLoggedRef = useRef(false);

  /**
   * Receives the lightweight worklet result (bbox, landmarks, crop arrays) and
   * runs the heavy TFLite models on the JS thread in parallel via Promise.all.
   *
   * Improvement 3 — parallel model inference:
   *   liveness0.run()  ┐
   *   liveness1.run()  ├─ all three fire simultaneously on fast-tflite's
   *   embedding.run()  ┘  native background thread pool
   *
   * Effective latency = max(~15 ms, ~15 ms, ~35 ms) ≈ 35 ms
   * vs. the old sequential worklet path: 40 + 40 + 80 = 160 ms.
   *
   * Improvement 4 (overlap): while JS awaits the Promise.all, the worklet has
   * already unlocked isBusy and can process the next camera frame.
   */
  const onFrameResult = useCallback(
    async (result: WorkletFrameResult) => {
      if (!active) return;

      // Log sensor dimensions once.
      if (__DEV__ && !frameDimsLoggedRef.current) {
        frameDimsLoggedRef.current = true;
        log(
          `[FrameDims] sensor: ${result.frameW}×${result.frameH}` +
          ` aspect=${(result.frameW / result.frameH).toFixed(3)}` +
          ` detected=${result.detected}`,
        );
      }

      if (!result.detected) {
        await feed(null, []);
        return;
      }

      // Reconstruct Point[] landmarks from flat [x0,y0,x1,y1,...] array.
      const lms: Point[] = [];
      for (let i = 0; i + 1 < result.landmarks.length; i += 2) {
        lms.push({ x: result.landmarks[i]!, y: result.landmarks[i + 1]! });
      }

      const headPose = estimateHeadPose(lms);
      const face: DetectedFace = {
        box: {
          x: result.bbox[0],
          y: result.bbox[1],
          width: result.bbox[2],
          height: result.bbox[3],
        },
        score: result.bbox[4],
        landmarks: lms,
        headPose,
      };

      // ── Parallel model inference on the JS thread ─────────────────────────
      // react-native-fast-tflite dispatches each .run() to a dedicated native
      // background thread; firing all three simultaneously gives true CPU
      // parallelism across the XNNPACK thread pools of each model instance.
      const m = modelsRef.current;

      const [livenessScore, embedding] = await Promise.all([
        // Liveness branches — both in parallel with each other AND with FaceNet.
        (result.lv0Crop && result.lv1Crop)
          ? Promise.all([
              m.liveness0.run([new Float32Array(result.lv0Crop)]),
              m.liveness1.run([new Float32Array(result.lv1Crop)]),
            ]).then(([r0, r1]) => {
              const live0 = decodeLiveness(r0[0] as Float32Array);
              const live1 = decodeLiveness(r1[0] as Float32Array);
              if (__DEV__ && !livenessRawLoggedRef.current) {
                livenessRawLoggedRef.current = true;
                log(
                  `[LivenessDebug] branch0 live=${live0.toFixed(4)}` +
                  ` branch1 live=${live1.toFixed(4)}` +
                  ` avg=${((live0 + live1) / 2).toFixed(4)}`,
                );
              }
              return (live0 + live1) / 2;
            })
          : Promise.resolve(undefined as number | undefined),

        // FaceNet embedding — runs concurrently with both liveness branches.
        result.emCrop
          ? m.embedding
              .run([new Float32Array(result.emCrop)])
              .then(r => l2NormalizeToFloat32(r[0] as Float32Array))
          : Promise.resolve(null as Float32Array | null),
      ]);

      await feed(null, [face], {
        brightness: result.brightness,
        livenessScore,   // undefined when lv0/lv1 crops were null (non-analyzing frames)
        embedding,       // null when emCrop was null
      });
    },
    [active, feed, modelsRef, livenessRawLoggedRef, frameDimsLoggedRef],
  );

  // Resolve config once per render.
  // getConfig().performance is already fully resolved by setConfig() in runtime.ts.
  const cfg        = getConfig();
  const thresholds = resolveThresholds(cfg.thresholds);
  const perf       = cfg.performance; // already Required<ModelPerformanceConfig>

  // Centering callback — fired from the worklet thread (via useRunOnJS) only
  // when the centering status or arrow direction actually changes.
  const handleCentering = useCallback((s: CenteringStatus) => {
    if (!s.detected) {
      setFaceCentered(null);
      setFacePos(null);
    } else {
      setFaceCentered(s.centered);
      setFacePos({ cx: s.cx, cy: s.cy });
    }
  }, []);

  const { frameProcessor, triggerDebugCapture } = useFrameSource({
    models,
    onFrameResult,
    minIntervalMs: 0,           // semaphore + back-pressure guard throttle naturally
    needsLiveness,
    needsEmbedding,
    onModelDebug: __DEV__ ? saveModelDebugFrame : undefined,
    onCentering: handleCentering,
    centerMargin: thresholds.centerMargin,
    frameSize: perf.frameSize,              // configurable resize-buffer size
    maxConcurrentFrames: perf.maxConcurrentFrames, // semaphore depth (default 2)
  });

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  // Start the session exactly once when the camera becomes available.
  const startedRef = useRef(false);
  useEffect(() => {
    if (autoStart && hasPermission && device && !startedRef.current) {
      startedRef.current = true;
      start();
    }
  }, [autoStart, hasPermission, device, start]);

  // ── Debug: every DEBUG_INTERVAL_MS — save model-input BMPs from the worklet ──
  // Saves three BMP files to faceauth_debug/ via react-native-fs on each tick:
  //   src_frame_*.bmp   — 160×160 full resize-plugin output (what the worklet sees)
  //   bf_input_*.bmp    — 128×128 BlazeFace input (after wResample01)
  //   face_crop_*.bmp   — 192×192 FaceMesh crop (only when face detected)
  //
  // NOTE: the full-resolution camera JPEGs (3 MB each) are NOT saved here — those
  // come from startPeriodicCapture which calls takePhoto().  We only want the
  // small model-input images so we skip that call entirely.
  //
  // To view: Files → Internal Storage → Android → data → com.faceauthexample → files → faceauth_debug
  // Or via adb:  adb pull /sdcard/Android/data/com.faceauthexample/files/faceauth_debug/ .
  useEffect(() => {
    if (!__DEV__) return;
    if (!active) return;
    // Fire once immediately, then every DEBUG_INTERVAL_MS.
    triggerDebugCapture();
    const t = setInterval(triggerDebugCapture, DEBUG_INTERVAL_MS);
    return () => clearInterval(t);
  }, [active, triggerDebugCapture]);

  const handleBack = useCallback(() => {
    cancel();
    onClose?.();
  }, [cancel, onClose]);

  const BackButton = onClose ? (
    <TouchableOpacity style={styles.backBtn} onPress={handleBack} hitSlop={12}>
      <Text style={styles.backText}>‹ Back</Text>
    </TouchableOpacity>
  ) : null;

  if (!hasPermission) {
    return (
      <View style={[styles.center, style]}>
        {BackButton}
        <Text style={styles.hint}>Camera permission required</Text>
      </View>
    );
  }
  if (device == null) {
    return (
      <View style={[styles.center, style]}>
        {BackButton}
        <Text style={styles.hint}>No front camera available</Text>
      </View>
    );
  }

  return (
    <View style={[styles.container, style]}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        format={format}
        fps={30}
        isActive={true}
        photo={true}
        // pixelFormat="rgb"
        resizeMode="cover"
        frameProcessor={frameProcessor}
      />

      {BackButton}
      <FaceMaskOverlay centered={faceCentered} facePos={facePos} />
      {/* Prominent centered prompt: the challenge instruction or "Verifying…" */}
      {(challenge || verifying) && (
        <View style={styles.promptCard} pointerEvents="none">
          {challenge && !verifying ? (
            <>
              <Text style={styles.promptIcon}>{CHALLENGE_PROMPT[challenge].icon}</Text>
              <Text style={styles.promptTitle}>{CHALLENGE_PROMPT[challenge].title}</Text>
              <Text style={styles.promptSub}>Hold still and follow the prompt</Text>
            </>
          ) : (
            <>
              <Text style={styles.promptIcon}>⏳</Text>
              <Text style={styles.promptTitle}>Verifying…</Text>
            </>
          )}
        </View>
      )}

      <View style={styles.hintBar} pointerEvents="none">
        <Text style={styles.hint}>{hint}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' },
  backBtn: {
    position: 'absolute',
    top: 16,
    left: 12,
    zIndex: 10,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 20,
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  backText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  promptCard: {
    position: 'absolute',
    top: '18%',
    left: 24,
    right: 24,
    alignItems: 'center',
    paddingVertical: 22,
    paddingHorizontal: 18,
    borderRadius: 20,
    backgroundColor: 'rgba(14,165,233,0.92)',
  },
  promptIcon: { fontSize: 52, marginBottom: 6 },
  promptTitle: { color: '#fff', fontSize: 28, fontWeight: '800', textAlign: 'center' },
  promptSub: { color: 'rgba(255,255,255,0.85)', fontSize: 14, marginTop: 6, textAlign: 'center' },
  hintBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingVertical: 16,
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  hint: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
