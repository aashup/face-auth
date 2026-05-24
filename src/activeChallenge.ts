import { LANDMARK } from './detector';
import type { Thresholds } from './config';
import type { DetectedFace } from './types';
import type { Challenge } from './types';

export interface ChallengeConfig {
  pool: Challenge[];
  timeoutMs: number;
  /** Gesture + tracking-loss thresholds sourced from the resolved config. */
  thresholds: Pick<Thresholds,
    | 'blinkCloseEar'
    | 'blinkOpenEar'
    | 'smileRatio'
    | 'turnYawThreshold'
    | 'turnReturnThreshold'
    | 'challengeLostFrameTolerance'
  >;
}

type ChallengeState = 'pending' | 'passed' | 'failed';

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function eyeAspectRatio(lm: { x: number; y: number }[], ring: readonly number[]): number {
  const [p0, p1, p2, p3, p4, p5] = ring;
  const a = dist(lm[p1!]!, lm[p5!]!);
  const b = dist(lm[p2!]!, lm[p4!]!);
  const c = dist(lm[p0!]!, lm[p3!]!);
  return (a + b) / (2 * c + 1e-6);
}

/**
 * Mouth width-to-height ratio. Unlike EAR (height/width, small=closed),
 * this is width/height — large when the mouth is open wide in a smile.
 * Threshold direction: ratio >= smileRatio → smile detected.
 */
function mouthWidthRatio(lm: { x: number; y: number }[]): number {
  const w = dist(lm[LANDMARK.mouthLeft]!, lm[LANDMARK.mouthRight]!);
  const h = dist(lm[LANDMARK.mouthTop]!, lm[LANDMARK.mouthBottom]!);
  return w / (h + 1e-6);
}

/**
 * Stateful active liveness challenge runner.
 *
 * Pull-based: the session loop calls `.feed(face, now)` each frame.
 * Returns `'pending'` until the gesture is complete or the timeout fires.
 */
export class ActiveChallenge {
  readonly challenge: Challenge;
  readonly prompt: string;
  private _status: ChallengeState = 'pending';

  // blink state machine
  private blinkSeenClosed = false;

  // turn state machine
  private turnSeenYaw = false;

  // lost-tracking grace window
  private lostFrameCount = 0;

  private readonly startMs: number;
  private readonly timeoutMs: number;
  private readonly config: ChallengeConfig;

  constructor(
    config: ChallengeConfig,
    startMs: number,
    rng: () => number = Math.random,
  ) {
    this.config = config;
    this.startMs = startMs;
    this.timeoutMs = config.timeoutMs;
    const idx = Math.floor(rng() * config.pool.length);
    this.challenge = config.pool[idx % config.pool.length]!;
    this.prompt = ActiveChallenge.promptFor(this.challenge);
  }

  get status(): ChallengeState {
    return this._status;
  }

  feed(face: DetectedFace | null, now: number): ChallengeState {
    if (this._status !== 'pending') return this._status;

    // Timeout
    if (now - this.startMs > this.timeoutMs) {
      this._status = 'failed';
      return 'failed';
    }

    // Lost tracking — allow a short grace window before failing.
    if (face === null) {
      this.lostFrameCount += 1;
      if (this.lostFrameCount > this.config.thresholds.challengeLostFrameTolerance) {
        this._status = 'failed';
        return 'failed';
      }
      return 'pending';
    }
    this.lostFrameCount = 0;

    switch (this.challenge) {
      case 'blink':
        return this.feedBlink(face);
      case 'smile':
        return this.feedSmile(face);
      case 'turn_head':
        return this.feedTurn(face);
    }
  }

  private feedBlink(face: DetectedFace): ChallengeState {
    const lm = face.landmarks;
    const ear =
      (eyeAspectRatio(lm, LANDMARK.leftEye) + eyeAspectRatio(lm, LANDMARK.rightEye)) / 2;
    const { blinkCloseEar, blinkOpenEar } = this.config.thresholds;

    if (!this.blinkSeenClosed && ear < blinkCloseEar) {
      this.blinkSeenClosed = true;
    } else if (this.blinkSeenClosed && ear >= blinkOpenEar) {
      this._status = 'passed';
    }
    return this._status;
  }

  private feedSmile(face: DetectedFace): ChallengeState {
    const ratio = mouthWidthRatio(face.landmarks);
    if (ratio >= this.config.thresholds.smileRatio) {
      this._status = 'passed';
    }
    return this._status;
  }

  private feedTurn(face: DetectedFace): ChallengeState {
    const yaw = Math.abs(face.headPose.yaw);
    const { turnYawThreshold, turnReturnThreshold } = this.config.thresholds;
    if (!this.turnSeenYaw && yaw > turnYawThreshold) {
      this.turnSeenYaw = true;
    } else if (this.turnSeenYaw && yaw < turnReturnThreshold) {
      this._status = 'passed';
    }
    return this._status;
  }

  private static promptFor(c: Challenge): string {
    switch (c) {
      case 'blink':     return 'Blink your eyes';
      case 'smile':     return 'Smile';
      case 'turn_head': return 'Turn your head';
    }
  }
}
