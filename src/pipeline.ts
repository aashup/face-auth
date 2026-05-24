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
  /** Passive-liveness score from the worklet (0 = spoof, 1 = live). */
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
  /** Per-frame liveness scores collected during the analyzing stage. */
  private livenessScores: number[] = [];
  private embeddings: Float32Array[] = [];
  private lastLog = 0;
  private readonly gateFrames: number;
  private readonly embedFrames: number;

  /** Number of frames to sample before making a liveness decision. */
  private get livenessFrames(): number {
    return this.opts.thresholds.livenessFrames ?? 4;
  }

  /** Running average of all collected liveness scores (0 = spoof, 1 = live). */
  private get livenessScore(): number {
    const s = this.livenessScores;
    return s.length ? s.reduce((a, b) => a + b, 0) / s.length : 0;
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
      livenessScore: this.livenessScore,
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
        { pool: this.opts.challengePool, timeoutMs: this.opts.challengeTimeoutMs },
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
        livenessScore: this.livenessScore,
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

  private async doAnalyzing(
    frame: RgbFrame | null,
    faces: DetectedFace[],
    pre?: PrecomputedFrameData,
  ): Promise<Tick> {
    if (faces.length === 0) {
      return { stage: 'analyzing', guidance: { message: 'Hold steady' }, done: false };
    }
    // Only spend the heavy models on a high-quality frame.
    const q = evaluateQuality(
      { faces, frame: frame ?? undefined, brightness: pre?.brightness },
      this.opts.thresholds,
    );
    if (!q.ok) {
      return { stage: 'analyzing', guidance: { message: q.guidance }, done: false };
    }
    const face = largestFace(faces);

    // ── Passive liveness: collect livenessFrames samples, then decide ────────
    if (this.livenessScores.length < this.livenessFrames) {
      let sampleScore: number;

      if (pre?.livenessScore !== undefined) {
        // Worklet path: worklet already provides a value in [0,1].
        // The worklet computes spoofScore = 1 - liveScore and sends that value.
        sampleScore = pre.livenessScore;
      } else if (frame) {
        // JS-side inference fallback (unit tests / headless mode).
        // Store spoofScore so that finalSpoof = average spoof probability and
        // the check `finalSpoof >= spoofReject` is semantically correct.
        const lv = await checkLiveness(frame, face, this.models.liveness, this.opts.thresholds.spoofReject);
        sampleScore = lv.spoofScore;
      } else {
        return { stage: 'analyzing', guidance: { message: 'Verifying…' }, done: false };
      }
      this.livenessScores.push(sampleScore);
      const n      = this.livenessScores.length;
      const runAvg = this.livenessScore;

      dbg(
        `liveness sample ${n}/${this.livenessFrames}: ` +
        `spoof=${f2(sampleScore)} live=${f2(1 - sampleScore)} ` +
        `runningAvg=${f2(runAvg)} threshold=${f2(this.opts.thresholds.spoofReject)}`,
      );

      if (n < this.livenessFrames) {
        return { stage: 'analyzing', guidance: { message: 'Verifying…' }, done: false };
      }

      const finalAvg   = runAvg;
      const finalSpoof = finalAvg;
      dbg(
        `liveness FINAL: samples=[${this.livenessScores.map(f2).join(', ')}] ` +
        `avg=${f2(finalAvg)} spoofAvg=${f2(finalSpoof)} ` +
        `${finalSpoof >= this.opts.thresholds.spoofReject ? 'REJECTED' : 'PASSED'}`,
      );

      if (finalSpoof >= this.opts.thresholds.spoofReject) {
        this.stage = 'done';
        return this.finish({
          ok: false,
          matchScore: 0,
          livenessScore: finalAvg,
          challengePassed: true,
          failureReason: 'liveness_failed',
        });
      }
      // Passed — fall through to embedding collection below.
    }

    // Embedding: use precomputed value or fall back to JS-side inference.
    if (pre?.embedding) {
      this.embeddings.push(pre.embedding);
    } else if (frame) {
      this.embeddings.push(await computeEmbedding(frame, face, this.models.embedding));
    } else {
      return { stage: 'analyzing', guidance: { message: 'Verifying…' }, done: false };
    }
    if (this.embeddings.length < this.embedFrames) {
      return { stage: 'analyzing', guidance: { message: 'Verifying…' }, done: false };
    }
    return this.match();
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
        livenessScore: this.livenessScore,
        challengePassed: true,
        embeddingB64: embeddingToBase64(averaged),
      });
    }

    if (this.opts.templates.length === 0) {
      dbg('match: no templates provisioned');
      return this.finish({
        ok: false,
        matchScore: 0,
        livenessScore: this.livenessScore,
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
        livenessScore: this.livenessScore,
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
      livenessScore: this.livenessScore,
      challengePassed: true,
      failureReason: match ? undefined : 'no_match',
    });
  }

  private finish(result: AuthResult): Tick {
    return { stage: 'done', guidance: { message: '' }, done: true, result };
  }
}
