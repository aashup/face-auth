import { LANDMARK } from './detector';
import type { DetectedFace } from './types';
import type { Challenge } from './types';

export interface ChallengeConfig {
  pool: Challenge[];
  timeoutMs: number;
}

type ChallengeState = 'pending' | 'passed' | 'failed';

// Thresholds
const BLINK_CLOSE_EAR = 0.18;
const BLINK_OPEN_EAR = 0.22;
const SMILE_RATIO = 0.72;
const TURN_YAW_THRESHOLD = 15;
const TURN_RETURN_THRESHOLD = 5;

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

function mouthAspectRatio(lm: { x: number; y: number }[]): number {
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

  private readonly startMs: number;
  private readonly timeoutMs: number;

  constructor(
    config: ChallengeConfig,
    startMs: number,
    rng: () => number = Math.random,
  ) {
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

    // Lost tracking
    if (face === null) {
      this._status = 'failed';
      return 'failed';
    }

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

    if (!this.blinkSeenClosed && ear < BLINK_CLOSE_EAR) {
      this.blinkSeenClosed = true;
    } else if (this.blinkSeenClosed && ear >= BLINK_OPEN_EAR) {
      this._status = 'passed';
    }
    return this._status;
  }

  private feedSmile(face: DetectedFace): ChallengeState {
    const ratio = mouthAspectRatio(face.landmarks);
    if (ratio >= SMILE_RATIO) {
      this._status = 'passed';
    }
    return this._status;
  }

  private feedTurn(face: DetectedFace): ChallengeState {
    const yaw = Math.abs(face.headPose.yaw);
    if (!this.turnSeenYaw && yaw > TURN_YAW_THRESHOLD) {
      this.turnSeenYaw = true;
    } else if (this.turnSeenYaw && yaw < TURN_RETURN_THRESHOLD) {
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
