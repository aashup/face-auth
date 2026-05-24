import { useCallback, useMemo, useRef, useState } from 'react';

import { FaceAuth } from './faceAuth';
import { getConfig } from './runtime';
import { loadAll } from './secure/templateStore';
import {
  FaceAuthSession,
  type PipelineModels,
  type PrecomputedFrameData,
  type Stage,
  type Tick,
} from './pipeline';
import type { RgbFrame } from './detector';
import type { AuthMode, AuthResult, DetectedFace, Guidance } from './types';

/**
 * Headless driver for a face-auth session. The host owns the camera + the
 * native frame source; each frame it calls `feed(frame, faces)`. The hook
 * advances the pipeline, surfaces guidance, and on completion records the
 * attendance event and invokes `onResult`. Use this for custom UIs;
 * `FaceAuthView` wraps it with a ready-made camera surface.
 */

export interface UseFaceAuthArgs {
  mode: AuthMode;
  personnelId?: string;
  models: PipelineModels;
  onResult?: (result: AuthResult) => void;
  onGuidance?: (g: Guidance) => void;
}

export interface UseFaceAuth {
  start: () => void;
  cancel: () => void;
  /**
   * Advance the pipeline with one frame's worth of data.
   * `frame` is null in worklet mode; `pre` carries worklet-computed values.
   */
  feed: (frame: RgbFrame | null, faces: DetectedFace[], pre?: PrecomputedFrameData) => Promise<void>;
  active: boolean;
  guidance: Guidance;
  /** Current pipeline stage (ref, so reading it doesn't trigger re-renders). */
  stageRef: React.MutableRefObject<Stage>;
}

export function useFaceAuth(args: UseFaceAuthArgs): UseFaceAuth {
  const sessionRef = useRef<FaceAuthSession | null>(null);
  const stageRef = useRef<Stage>('gating');
  const [active, setActive] = useState(false);
  const [guidance, setGuidance] = useState<Guidance>({ message: '' });

  const buildSession = useCallback(() => {
    const cfg = getConfig();
    return new FaceAuthSession(args.models, {
      mode: args.mode,
      personnelId: args.personnelId,
      templates: loadAll(),
      thresholds: cfg.thresholds,
      challengePool: cfg.challenges,
      challengeTimeoutMs: cfg.challengeTimeoutMs,
      probeTransform: FaceAuth.probeTransform,
    });
  }, [args.models, args.mode, args.personnelId]);

  const finish = useCallback(
    (tick: Tick) => {
      sessionRef.current = null;
      setActive(false);
      if (tick.result) {
        // Enroll captures a template; it isn't an attendance event.
        if (args.mode !== 'enroll') FaceAuth.recordAttendance(tick.result);
        args.onResult?.(tick.result);
      }
    },
    [args],
  );

  const start = useCallback(() => {
    sessionRef.current = buildSession();
    stageRef.current = 'gating';
    setActive(true);
    setGuidance({ message: 'Position your face in the frame' });
  }, [buildSession]);

  const cancel = useCallback(() => {
    const s = sessionRef.current;
    if (s) finish(s.cancel());
  }, [finish]);

  const feed = useCallback(
    async (frame: RgbFrame | null, faces: DetectedFace[], pre?: PrecomputedFrameData) => {
      const s = sessionRef.current;
      if (!s) return;
      const tick = await s.process(frame, faces, Date.now(), pre);
      // Session may have been cancelled/finished while awaiting inference.
      if (sessionRef.current !== s) return;
      stageRef.current = tick.stage;
      setGuidance(tick.guidance);
      args.onGuidance?.(tick.guidance);
      if (tick.done) finish(tick);
    },
    [args, finish],
  );

  return useMemo(
    () => ({ start, cancel, feed, active, guidance, stageRef }),
    [start, cancel, feed, active, guidance],
  );
}
