import {
  DEFAULT_CONFIG,
  DEFAULT_PERFORMANCE,
  resolveThresholds,
  resolvePerformance,
  type FaceAuthConfig,
  type ModelPerformanceConfig,
  type Thresholds,
} from './config';
import { setDebug } from './logger';
import type { Challenge } from './types';

/** Fully-resolved runtime configuration, available after `init()`. */
export interface ResolvedConfig {
  /** Undefined when running in fully-offline / local-enrollment mode. */
  awsSyncUrl: string | undefined;
  /** Undefined when running in fully-offline / local-enrollment mode. */
  deviceToken: string | undefined;
  thresholds: Thresholds;
  challenges: Challenge[];
  syncIntervalMs: number;
  purgeAfterSync: boolean;
  challengeTimeoutMs: number;
  templateTtlMs: number;
  debug: boolean;
  /** Resolved model performance config (thread counts + frameSize). */
  performance: Required<ModelPerformanceConfig>;
}

let current: ResolvedConfig | null = null;

export function setConfig(config: FaceAuthConfig): ResolvedConfig {
  const debug = config.debug ?? false;
  setDebug(debug);
  current = {
    awsSyncUrl: config.awsSyncUrl,
    deviceToken: config.deviceToken ?? undefined,
    thresholds: resolveThresholds(config.thresholds),
    challenges: config.challenges ?? DEFAULT_CONFIG.challenges,
    syncIntervalMs: config.syncIntervalMs ?? DEFAULT_CONFIG.syncIntervalMs,
    purgeAfterSync: config.purgeAfterSync ?? DEFAULT_CONFIG.purgeAfterSync,
    challengeTimeoutMs: config.challengeTimeoutMs ?? DEFAULT_CONFIG.challengeTimeoutMs,
    templateTtlMs: config.templateTtlMs ?? DEFAULT_CONFIG.templateTtlMs,
    debug,
    performance: resolvePerformance(config.performance),
  };
  return current;
}

/** Merge server-provisioned threshold overrides into the live config. */
export function updateThresholds(partial: Partial<Thresholds>): void {
  if (!current) {
    throw new Error('[offline-face-auth] not initialized — cannot update thresholds.');
  }
  current.thresholds = { ...current.thresholds, ...partial };
}

/** Throws if `FaceAuth.init()` has not been called yet. */
export function getConfig(): ResolvedConfig {
  if (!current) {
    throw new Error(
      '[offline-face-auth] not initialized — call FaceAuth.init(config) before use.',
    );
  }
  return current;
}

export function isInitialized(): boolean {
  return current !== null;
}

export function clearConfig(): void {
  current = null;
}
