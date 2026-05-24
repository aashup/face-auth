/**
 * Public type surface for react-native-offline-face-auth.
 * Only types re-exported from `src/index.ts` are considered stable API.
 */

/** A 2D point in normalized frame coordinates (0..1). */
export interface Point {
  x: number;
  y: number;
}

/** Axis-aligned face bounding box in normalized frame coordinates (0..1). */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Head orientation in degrees, derived from landmarks. */
export interface HeadPose {
  yaw: number; // left/right
  pitch: number; // up/down
  roll: number; // tilt
}

/**
 * Output of the on-device detector for a single frame.
 * `landmarks` is the Face Mesh point set (subset or full 468) in normalized coords.
 */
export interface DetectedFace {
  box: BoundingBox;
  landmarks: Point[];
  headPose: HeadPose;
  /** Detector confidence 0..1. */
  score: number;
}

/** Why a frame failed the quality gate (for user guidance). */
export type QualityIssue =
  | 'no_face'
  | 'multiple_faces'
  | 'face_too_small'
  | 'off_center'
  | 'bad_pose'
  | 'too_dark'
  | 'too_bright'
  | 'blurry';

export interface QualityResult {
  ok: boolean;
  issues: QualityIssue[];
  /** Human-readable guidance for the top issue, e.g. "Move closer". */
  guidance: string;
}

/** Active liveness challenge kinds. */
export type Challenge = 'blink' | 'smile' | 'turn_head';

export type AuthMode = 'identify' | 'verify' | 'enroll';

/** Result of one authentication attempt. */
export interface AuthResult {
  ok: boolean;
  /** Matched personnel id (identify mode) or the verified id (verify mode). */
  personnelId?: string;
  matchScore: number;
  livenessScore: number;
  challengePassed: boolean;
  /** Enroll mode only: the captured L2-normalized embedding (base64 float32,
   *  pre-cancelable) to POST to the server for this personnel. */
  embeddingB64?: string;
  /** Present when ok === false. */
  failureReason?:
    | 'no_template'
    | 'liveness_failed'
    | 'challenge_failed'
    | 'no_match'
    | 'timeout'
    | 'cancelled';
}

/** Live guidance pushed to the host UI during a session. */
export interface Guidance {
  /** Short message, e.g. "Center your face", "Blink now". */
  message: string;
  /** Current active challenge, if one is in progress. */
  challenge?: Challenge;
}

export type SyncState = 'idle' | 'syncing' | 'error';

export interface SyncStatus {
  state: SyncState;
  pendingCount: number;
  lastSyncAt?: number;
  lastError?: string;
}
