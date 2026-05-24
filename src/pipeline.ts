import { ActiveChallenge } from './activeChallenge';
import type { Thresholds } from './config';
import { averageEmbeddings, computeEmbedding, embeddingToBase64 } from './embeddings';
import type { RgbFrame, TfliteModel } from './detector';
import { checkLiveness, type LivenessModels } from './liveness';
import { log } from './logger';
import {
  groupTemplates,
  identifyMultiFrame,
  verifyMultiFrame,
  type MultiFrameMatchConfig,
  type Template,
} from './matcher';
import { evaluateQuality, largestFace } from './qualityGate';
import type {
  AuthMode,
  AuthResult,
  Challenge,
  DetectedFace,
  Guidance,
} from './types';

/**
 * Session orchestrator implementing a **bifurcated execution loop**:
 *
 *  - High-frequency tracking (gating + challenge): only the lightweight
 *    BlazeFace/Face Mesh models run, so the loop stays fast enough (tens of fps)
 *    to catch a 100–150 ms blink and head-yaw trajectory for the active
 *    challenge.
 *  - Low-frequency analysis (analyzing): the heavy MiniFASNet (passive liveness)
 *    and embedding models run once, on high-quality frames, only after the
 *    active challenge has passed.
 *
 * Matching uses the **3-step cascading multi-frame engine**:
 *  1. Coarse filter (probe[0] vs first enrolled template — fast 1:N).
 *  2. Cross-product score matrix (all probe frames × all enrolled frames).
 *  3. Score fusion (`mean | max | top_k`) + fine threshold gate.
 *
 * Pull-based and side-effect free, so it is trivial to unit-test and identical
 * across platforms.
 */

export type Stage = 'gating' | 'challenge' | 'analyzing' | 'done';

/** Debug logger — prints to Metro/logcat with a [FaceAuth] tag. */
const dbg = (...args: unknown[]) => log('[FaceAuth]', ...args);
const f2 = (n: number) => n.toFixed(3);

export interface PipelineModels {
  embedding: TfliteModel;
  liveness: LivenessModels;
}

/** Pre-computed values from the worklet thread — skip JS-side model calls when set. */
export interface PrecomputedFrameData {
  /** Mean luma (0–255) of the face region; replaces frame-based brightness check. */
  brightness?: number;
  /** Live-face probability from the worklet (0 = spoof, 1 = live). Must be inverted to spoofScore before use. */
  livenessScore?: number;
  /** L2-normalised FaceNet embedding from the worklet; null = not ready yet. */
  embedding?: Float32Array | null;
}

export interface PipelineOptions {
  mode: AuthMode;
  personnelId?: string;
  templates: Template[];
  thresholds: Thresholds;
  challengePool: Challenge[];
  challengeTimeoutMs: number;
  /** Consecutive good frames required before leaving the gate. */
  gateFrames?: number;
  /**
   * Number of probe frames to collect before triggering the match step.
   * Should be ≥ `thresholds.probeFrameCount` so the cross-matrix has enough
   * frames to work with.  Default 3 (matches `DEFAULT_THRESHOLDS.probeFrameCount`).
   */
  embedFrames?: number;
  /**
   * Applied to each probe embedding before matching (cancelable transform).
   * Must match the transform used on the stored templates. Default: identity.
   */
  probeTransform?: (embedding: Float32Array) => Float32Array;
  rng?: () => number;
}

export interface Tick {
  stage: Stage;
  guidance: Guidance;
  done: boolean;
  result?: AuthResult;
}

export class FaceAuthSession {
  private stage: Stage = 'gating';
  private goodGateFrames = 0;
  private challenge: ActiveChallenge | null = null;
  /**
   * Per-frame **spoof** probabilities (0 = definitely live, 1 = definitely spoof).
   * Renamed from `livenessScores` to prevent the semantic confusion where high
   * "liveness score" was actually a high spoof probability.
   */
  private spoofSamples: number[] = [];
  private embeddings: Float32Array[] = [];
  private lastLog = 0;
  private readonly gateFrames: number;
  private readonly embedFrames: number;

  /** Number of frames to sample before making a liveness decision. */
  private get livenessFrames(): number {
    return this.opts.thresholds.livenessFrames ?? 4;
  }

  /** Running average spoof probability (0 = live, 1 = spoof). Used for the gate check. */
  private get avgSpoofScore(): number {
    const s = this.spoofSamples;
    return s.length ? s.reduce((a, b) => a + b, 0) / s.length : 0;
  }

  /** Live probability for result reporting (0 = spoof, 1 = live — intuitive direction). */
  private get avgLiveScore(): number {
    return 1 - this.avgSpoofScore;
  }

  /** Build `MultiFrameMatchConfig` from the resolved `Thresholds`. */
  private get matchCfg(): MultiFrameMatchConfig {
    const t = this.opts.thresholds;
    return {
      coarseThreshold:    t.coarseThreshold,
      fineThreshold:      t.fineThreshold,
      probeFrameCount:    t.probeFrameCount,
      templateFrameCount: t.templateFrameCount,
      fusionStrategy:     t.fusionStrategy,
      topK:               t.topK,
    };
  }

  constructor(
    private readonly models: PipelineModels,
    private readonly opts: PipelineOptions,
  ) {
    if (opts.mode === 'verify' && !opts.personnelId) {
      throw new Error('verify mode requires a personnelId');
    }
    this.gateFrames  = opts.gateFrames ?? 3;
    // Default 3: collect enough frames to fill the cross-matrix (probeFrameCount=3).
    this.embedFrames = opts.embedFrames ?? 3;
  }

  get currentStage(): Stage {
    return this.stage;
  }

  /**
   * Advance the session by one camera frame.
   * `frame` is null in worklet mode (all inference is pre-computed).
   * `pre` carries worklet-computed values; falls back to JS-side inference when absent.
   */
  async process(
    frame: RgbFrame | null,
    faces: DetectedFace[],
    now: number,
    pre?: PrecomputedFrameData,
  ): Promise<Tick> {
    // Throttled tracking log (don't flood at high fps).
    if (now - this.lastLog > 700) {
      this.lastLog = now;
      const top = faces[0];
      dbg(
        `stage=${this.stage} faces=${faces.length}` +
          (top ? ` score=${f2(top.score)} yaw=${top.headPose.yaw.toFixed(0)}` : ''),
      );
    }
    switch (this.stage) {
      case 'gating':
        return this.doGate(frame, faces, pre);
      case 'challenge':
        return this.doChallenge(faces, now);
      case 'analyzing':
        return this.doAnalyzing(frame, faces, pre);
      case 'done':
        return { stage: 'done', guidance: { message: '' }, done: true };
    }
  }

  cancel(): Tick {
    this.stage = 'done';
    return this.finish({
      ok: false,
      matchScore: 0,
      livenessScore: this.avgLiveScore,
      challengePassed: this.challenge?.status === 'passed',
      failureReason: 'cancelled',
    });
  }

  // ── High-frequency tracking loop (lightweight: BlazeFace box only) ────────

  private doGate(frame: RgbFrame | null, faces: DetectedFace[], pre?: PrecomputedFrameData): Tick {
    const q = evaluateQuality(
      { faces, frame: frame ?? undefined, brightness: pre?.brightness },
      this.opts.thresholds,
    );
    if (!q.ok) {
      this.goodGateFrames = 0;
      return { stage: 'gating', guidance: { message: q.guidance }, done: false };
    }
    this.goodGateFrames += 1;
    if (this.goodGateFrames >= this.gateFrames) {
      // Enroll mode: skip the active challenge — the user is registering their
      // own face, so there is no fraud risk and no point asking them to blink.
      // Go straight to analyzing so only the embedding is captured.
      if (this.opts.mode === 'enroll') {
        this.stage = 'analyzing';
        dbg('gate passed → analyzing (enroll — challenge skipped)');
        return { stage: 'analyzing', guidance: { message: 'Hold still…' }, done: false };
      }

      // Identify / verify: go through the active challenge for liveness proof.
      this.challenge = new ActiveChallenge(
        {
          pool: this.opts.challengePool,
          timeoutMs: this.opts.challengeTimeoutMs,
          thresholds: this.opts.thresholds,
        },
        Date.now(),
        this.opts.rng,
      );
      this.stage = 'challenge';
      dbg(`gate passed → challenge: ${this.challenge.challenge}`);
      return {
        stage: 'challenge',
        guidance: { message: this.challenge.prompt, challenge: this.challenge.challenge },
        done: false,
      };
    }
    return { stage: 'gating', guidance: { message: 'Hold steady' }, done: false };
  }

  private doChallenge(faces: DetectedFace[], now: number): Tick {
    const c = this.challenge!;
    // Require exactly one tracked face; otherwise treat as lost tracking.
    const face = faces.length === 1 ? faces[0]! : null;
    const state = c.feed(face, now);
    if (state === 'failed') {
      this.stage = 'done';
      dbg(`challenge FAILED (${c.challenge}) — timeout or lost tracking`);
      return this.finish({
        ok: false,
        matchScore: 0,
        livenessScore: this.avgLiveScore,
        challengePassed: false,
        failureReason: 'challenge_failed',
      });
    }
    if (state === 'passed') {
      this.stage = 'analyzing';
      dbg(`challenge PASSED (${c.challenge}) → analyzing`);
      return { stage: 'analyzing', guidance: { message: 'Verifying…' }, done: false };
    }
    return {
      stage: 'challenge',
      guidance: { message: c.prompt, challenge: c.challenge },
      done: false,
    };
  }

  // ── Low-frequency analysis (heavy: MiniFASNet + embedding) ───────────────

  /**
   * Improvement 5 — simultaneous liveness + embedding collection.
   *
   * Old behaviour: collect N liveness frames → check pass/fail → collect M
   * embedding frames → match.  Total: N + M frames (serial).
   *
   * New behaviour: collect liveness AND embedding on the same frames.
   * Match when BOTH counters reach their target.
   * Total: max(N, M) frames — typically 4 vs. 7, saving ~3 × 30–40 ms ≈ 90–120 ms.
   *
   * Liveness failure is checked as soon as the last sample lands; any embeddings
   * collected on that frame are discarded with the session.
   */
  private async doAnalyzing(
    frame: RgbFrame | null,
    faces: DetectedFace[],
    pre?: PrecomputedFrameData,
  ): Promise<Tick> {
    if (faces.length === 0) {
      return { stage: 'analyzing', guidance: { message: 'Hold steady' }, done: false };
    }
    const q = evaluateQuality(
      { faces, frame: frame ?? undefined, brightness: pre?.brightness },
      this.opts.thresholds,
    );
    if (!q.ok) {
      return { stage: 'analyzing', guidance: { message: q.guidance }, done: false };
    }
    const face      = largestFace(faces);
    const threshold = this.opts.thresholds.spoofReject;

    // ── Step A: accumulate one liveness sample (if still needed) ─────────────
    // Key change: we do NOT return early after collecting the sample; we fall
    // through to Step B so both liveness and embedding are gathered per frame.
    let livenessJustFailed = false;

    if (this.spoofSamples.length < this.livenessFrames) {
      let spoofScore: number;
      let liveScore: number;

      if (pre?.livenessScore !== undefined) {
        // Primary path: JS-thread Promise.all already ran the models.
        liveScore  = pre.livenessScore;
        spoofScore = 1 - liveScore;
      } else if (frame) {
        // Fallback: JS-side inference (unit tests / headless mode).
        const lv = await checkLiveness(frame, face, this.models.liveness, threshold);
        liveScore  = lv.liveScore;
        spoofScore = lv.spoofScore;
      } else {
        // No data at all for this frame — skip (will retry next frame).
        return { stage: 'analyzing', guidance: { message: 'Verifying…' }, done: false };
      }

      this.spoofSamples.push(spoofScore);
      const n        = this.spoofSamples.length;
      const runSpoof = this.avgSpoofScore;

      dbg(
        `[Liveness] sample ${n}/${this.livenessFrames}` +
        ` | liveScore=${f2(liveScore)} spoofScore=${f2(spoofScore)}` +
        ` | runAvgSpoof=${f2(runSpoof)} threshold=${f2(threshold)}`,
      );

      if (n >= this.livenessFrames) {
        // All liveness samples collected — make the final decision.
        const finalSpoofAvg = runSpoof;
        const finalLiveAvg  = 1 - finalSpoofAvg;
        const rejected      = finalSpoofAvg >= threshold;

        dbg(
          `[Liveness] FINAL` +
          ` | samples(spoof)=[${this.spoofSamples.map(f2).join(', ')}]` +
          ` | avgSpoof=${f2(finalSpoofAvg)} avgLive=${f2(finalLiveAvg)}` +
          ` | threshold=${f2(threshold)}` +
          ` | ${rejected ? 'REJECTED (spoof detected)' : 'PASSED (live face)'}`,
        );

        if (rejected) {
          livenessJustFailed = true;
          // Fall through to Step B so we collect the embedding on this frame
          // too — then return failure below (after the embedding attempt).
          // This avoids a wasted async call when liveness is clearly spoofed.
        }
      }
      // ── No early return here — fall through to embedding collection ──────────
    }

    // ── Step B: accumulate one embedding (if still needed) ───────────────────
    if (this.embeddings.length < this.embedFrames) {
      if (pre?.embedding) {
        this.embeddings.push(pre.embedding);
      } else if (frame) {
        this.embeddings.push(await computeEmbedding(frame, face, this.models.embedding));
      }
      // If neither source is available for this frame, silently skip
      // (will be retried on the next frame).
    }

    // ── Step C: decide ────────────────────────────────────────────────────────
    if (livenessJustFailed) {
      this.stage = 'done';
      return this.finish({
        ok: false,
        matchScore: 0,
        livenessScore: this.avgLiveScore,
        challengePassed: true,
        failureReason: 'liveness_failed',
      });
    }

    const livenessComplete  = this.spoofSamples.length >= this.livenessFrames;
    const embeddingComplete = this.embeddings.length   >= this.embedFrames;

    if (livenessComplete && embeddingComplete) {
      return this.match();
    }

    return { stage: 'analyzing', guidance: { message: 'Verifying…' }, done: false };
  }

  private match(): Tick {
    this.stage = 'done';
    const probes = this.embeddings; // raw per-frame embeddings, each already L2-normalised

    // ── Enroll: return averaged embedding for storage — no match needed ───────
    if (this.opts.mode === 'enroll') {
      const averaged = averageEmbeddings(probes);
      dbg(`ENROLL captured embedding (dims=${averaged.length})`);
      return this.finish({
        ok: true,
        matchScore: 1,
        livenessScore: this.avgLiveScore,
        challengePassed: true,
        embeddingB64: embeddingToBase64(averaged),
      });
    }

    if (this.opts.templates.length === 0) {
      dbg('match: no templates provisioned');
      return this.finish({
        ok: false,
        matchScore: 0,
        livenessScore: this.avgLiveScore,
        challengePassed: true,
        failureReason: 'no_template',
      });
    }

    // Apply cancelable transform to each probe frame individually.
    // The transform is linear so per-frame ≡ transform(averaged).
    const transformedProbes = this.opts.probeTransform
      ? probes.map((p) => this.opts.probeTransform!(p))
      : probes;

    const cfg = this.matchCfg;

    // ── Verify (1:1) ─────────────────────────────────────────────────────────
    if (this.opts.mode === 'verify') {
      const registry = groupTemplates(this.opts.templates);
      const group = registry.find((g) => g.personnelId === this.opts.personnelId!) ?? null;
      const v = verifyMultiFrame(transformedProbes, group, cfg);
      const noTemplate = group === null || group.embeddings.length === 0;
      const safeScore  = v.score ?? 0;
      dbg(
        `verify ${this.opts.personnelId}: score=${noTemplate ? 'N/A (no template)' : f2(v.score!)}` +
        ` coarse=${f2(cfg.coarseThreshold)} fine=${f2(cfg.fineThreshold)} ok=${v.ok}`,
      );
      return this.finish({
        ok: v.ok,
        personnelId: v.ok ? this.opts.personnelId : undefined,
        matchScore: safeScore,
        livenessScore: this.avgLiveScore,
        challengePassed: true,
        failureReason: v.ok ? undefined : (noTemplate ? 'no_template' : 'no_match'),
      });
    }

    // ── Identify (1:N) ───────────────────────────────────────────────────────
    const registry = groupTemplates(this.opts.templates);
    const { match, candidatesEvaluated } = identifyMultiFrame(transformedProbes, registry, cfg);
    dbg(
      `identify: candidates=${candidatesEvaluated}/${registry.length}` +
      ` best=${match?.personnelId ?? '—'} score=${f2(match?.score ?? 0)}` +
      ` coarse=${f2(cfg.coarseThreshold)} fine=${f2(cfg.fineThreshold)}`,
    );
    return this.finish({
      ok: match !== null,
      personnelId: match?.personnelId,
      matchScore: match?.score ?? 0,
      livenessScore: this.avgLiveScore,
      challengePassed: true,
      failureReason: match ? undefined : 'no_match',
    });
  }

  private finish(result: AuthResult): Tick {
    return { stage: 'done', guidance: { message: '' }, done: true, result };
  }
}
