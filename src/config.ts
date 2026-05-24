import type { ModelManifest, ModelReader } from './secure/modelIntegrity';
import type { Challenge } from './types';

/**
 * Strategy used to collapse the M×K probe-vs-template score matrix into a
 * single confidence value during multi-frame matching.
 *
 * - `'mean'`  — arithmetic average of all scores (simple, sensitive to outliers).
 * - `'max'`   — single best-matching frame pair (optimistic; good for diverse
 *               enrollment angles but can be fooled by a lucky frame).
 * - `'top_k'` — average of the `topK` highest scores; balances robustness
 *               against single-frame noise with sensitivity to the best frames.
 *               Recommended default.
 */
export type FusionStrategy = 'mean' | 'max' | 'top_k';

/** Tunable decision thresholds. Defaults are conservative starting points and
 *  should be calibrated against a validation set (see PLAN.md Phase 4). */
export interface Thresholds {
  /** Minimum detector confidence to accept a face. */
  detectScore: number;
  /** Minimum fraction of frame area the face box must cover. */
  minFaceArea: number;
  /** Max |yaw|/|pitch|/|roll| in degrees for the quality gate. */
  maxYaw: number;
  maxPitch: number;
  maxRoll: number;
  /** Mean luma bounds (0..255). */
  minBrightness: number;
  maxBrightness: number;
  /** Variance-of-Laplacian sharpness floor (higher = sharper). */
  minSharpness: number;
  /**
   * Passive liveness: reject if spoof probability >= this.
   * Real faces typically score spoof ≈ 0.05–0.15 (i.e. live ≈ 0.85–0.95).
   * Printed photos / screen replays typically score spoof ≈ 0.85–0.99.
   * Default 0.5 = reject when model is ≥50% confident it is a spoof.
   * Never set below 0.3 in production — will reject legitimate live faces.
   */
  spoofReject: number;
  /**
   * @deprecated Use `fineThreshold` instead. Kept for backwards-compat with
   * single-frame matching code paths (e.g. provisioned server templates that
   * have not yet migrated to multi-frame storage).
   * Cosine similarity required for a legacy single-probe match.
   */
  matchCosine: number;
  /**
   * Number of good frames to sample liveness across before making a decision.
   * Scores are averaged — all N frames must be collected before pass/fail.
   * Default 4. Set to 1 to restore single-frame behaviour.
   */
  livenessFrames: number;
  /**
   * Normalised margin from each frame edge within which the face centre must
   * fall to pass the centering check. Default 0.10 → face centre accepted
   * anywhere in the middle 80 % of the frame (0.10 → 0.90 on both axes).
   * Increase toward 0.20 for a tighter center zone; decrease toward 0.05 to
   * accept faces that are slightly off-center.
   */
  centerMargin: number;

  // ── Multi-frame matching ─────────────────────────────────────────────────

  /**
   * Cosine similarity floor for the fast coarse 1:N pass.
   * Only probe[0] vs each person's first enrolled template is tested here.
   * Identities scoring below this are dropped before the cross-matrix step.
   * Default 0.55.  Lower → more candidates reach step 2 (more CPU, fewer
   * false rejects); higher → faster but risks dropping genuine matches.
   */
  coarseThreshold: number;
  /**
   * Final acceptance threshold applied after score fusion across the full
   * M×K cross-product matrix.  Default 0.72.
   *
   * ⚠️  For FaceNet-512 same-person cosine scores typically fall in 0.55–0.90.
   * Setting this above 0.85 without per-deployment calibration will produce
   * significant false-reject rates.  Tighten via `FaceAuth.init` overrides.
   */
  fineThreshold: number;
  /**
   * Number of live probe frames fed into the cross-matrix.  Must be ≤ the
   * number of frames collected by the pipeline (`embedFrames`).  Default 3.
   */
  probeFrameCount: number;
  /**
   * Maximum number of enrolled frames per person used in the cross-matrix.
   * If a person was enrolled with more shots, only the first N are used.
   * Default 3.
   */
  templateFrameCount: number;
  /**
   * How to collapse the M×K score matrix into a single confidence value.
   * Default `'top_k'` — averages the `topK` highest-scoring frame pairs,
   * which is robust against a single blurry or off-angle frame.
   */
  fusionStrategy: FusionStrategy;
  /**
   * K for `top_k` fusion — how many top scores to include in the average.
   * Ignored when `fusionStrategy` is not `'top_k'`.  Default 3.
   */
  topK: number;
}

export interface FaceAuthConfig {
  /**
   * AWS endpoint that receives attendance batches and serves provisioned
   * templates. **Optional** — omit (or leave undefined) for a fully offline
   * deployment where `FaceAuth.enrollLocal()` is used for enrollment.
   * When absent, `provision()` and `syncNow()` are no-ops and the background
   * net-watcher is not started.
   */
  awsSyncUrl?: string;
  /**
   * Per-device bearer token issued during provisioning.
   * Required only when `awsSyncUrl` is provided.
   */
  deviceToken?: string;
  /**
   * Enable SDK debug logging to the native console (Metro / logcat / Xcode).
   * Set to `true` during development; leave `false` (default) in production
   * to keep the console clean.
   */
  debug?: boolean;
  /** Override any subset of thresholds. */
  thresholds?: Partial<Thresholds>;
  /** Which active challenges are allowed; one is picked at random per auth. */
  challenges?: Challenge[];
  /** Background sync cadence in ms (also fires on network reconnect). */
  syncIntervalMs?: number;
  /** Delete local attendance rows once the server acks them. */
  purgeAfterSync?: boolean;
  /** Max active-challenge duration before timeout (ms). */
  challengeTimeoutMs?: number;
  /** Secret driving the cancelable template transform (rotate to revoke).
   *  Defaults to the device hardware key if omitted. */
  cancelableKey?: string;
  /** Output dimensionality of the cancelable transform. Default 512. */
  cancelableDims?: number;
  /** Stable device identifier stamped onto attendance events. */
  deviceId?: string;
  /** Returns a fresh platform attestation token (Play Integrity / App Attest). */
  getAttestationToken?: () => Promise<string | null>;
  /** Bundled model manifest (e.g. require('../models/manifest.json')) verified
   *  at boot when `readModelBytes` is also supplied. */
  modelManifest?: ModelManifest;
  /** Host-supplied reader for a bundled model's bytes (platform file access).
   *  When provided alongside `modelManifest`, `init()` verifies every model's
   *  SHA-256 and throws if any has been tampered with. */
  readModelBytes?: ModelReader;
  /** Force re-provisioning when the last successful provision is older than
   *  this many ms. `FaceAuth.needsReprovision()` reflects it. Default: 7 days. */
  templateTtlMs?: number;
}

export const DEFAULT_THRESHOLDS: Thresholds = {
  detectScore:        0.8,
  minFaceArea:        0.4,
  maxYaw:             30,
  maxPitch:           50,
  maxRoll:            50,
  minBrightness:      50,
  maxBrightness:      230,
  minSharpness:       80,
  spoofReject:        0.5,
  matchCosine:        0.65,  // legacy single-frame path
  livenessFrames:     4,
  centerMargin:       0.10,
  // Multi-frame matching
  coarseThreshold:    0.55,
  fineThreshold:      0.72,
  probeFrameCount:    3,
  templateFrameCount: 3,
  fusionStrategy:     'top_k',
  topK:               3,
};

export const DEFAULT_CONFIG: {
  challenges: Challenge[];
  syncIntervalMs: number;
  purgeAfterSync: boolean;
  challengeTimeoutMs: number;
  templateTtlMs: number;
} = {
  challenges: ['blink', 'smile', 'turn_head'],
  syncIntervalMs: 5 * 60 * 1000,
  purgeAfterSync: true,
  challengeTimeoutMs: 10 * 60 * 1000,
  templateTtlMs: 7 * 24 * 60 * 60 * 1000,
};

export function resolveThresholds(overrides?: Partial<Thresholds>): Thresholds {
  return { ...DEFAULT_THRESHOLDS, ...(overrides ?? {}) };
}
