/**
 * react-native-offline-face-auth — public API.
 *
 * Only the symbols exported here are stable. Internal modules (detector,
 * qualityGate, secure/*, sync/*) may change without notice.
 */

export { FaceAuth } from './faceAuth';
export type { DuplicateCheckResult } from './faceAuth';

export { FaceAuthView } from './FaceAuthView';
export type { FaceAuthViewProps, FaceAuthModels } from './FaceAuthView';

export { FaceAuthModal } from './FaceAuthModal';
export type { FaceAuthModalProps } from './FaceAuthModal';

export { useFaceAuth } from './useFaceAuth';
export type { UseFaceAuth, UseFaceAuthArgs } from './useFaceAuth';

export type { PipelineModels, PrecomputedFrameData, Stage, Tick } from './pipeline';
export type { RgbFrame, TfliteModel, DetectorModels } from './detector';
export type { FrameSourceModels, WorkletFrameResult } from './frameSource';
export type { LivenessModels } from './liveness';
export type { Template, MultiFrameTemplate, MultiFrameMatchConfig, FinalMatch } from './matcher';

export type { FaceAuthConfig, Thresholds, FusionStrategy } from './config';
export { DEFAULT_THRESHOLDS } from './config';

export type {
  AuthMode,
  AuthResult,
  Challenge,
  DetectedFace,
  Guidance,
  QualityIssue,
  QualityResult,
  SyncStatus,
  SyncState,
  BoundingBox,
  HeadPose,
  Point,
} from './types';
