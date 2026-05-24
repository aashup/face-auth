import { DEFAULT_THRESHOLDS } from '../config';
import { FaceAuthSession, type PipelineModels, type PipelineOptions } from '../pipeline';
import type { Template } from '../matcher';
import {
  livenessStub,
  makeFace,
  makeFrame,
  seededEmbedding,
  stubModel,
} from './_helpers';

const DIMS = 128;
const RELAXED = {
  ...DEFAULT_THRESHOLDS,
  minSharpness:                0,
  minBrightness:               0,
  minFaceArea:                 0,
  livenessFrames:              1,   // one analyzing frame is enough in unit tests
  coarseThreshold:             0,   // disable coarse filter so all candidates reach step 2
  fineThreshold:               0.5, // same as matchCosine legacy default
  challengeLostFrameTolerance: 0,   // fail on first missing frame for deterministic tests
};

function makeOptions(over: Partial<PipelineOptions> = {}): PipelineOptions {
  return {
    mode: 'identify',
    templates: [{ personnelId: 'A', embedding: seededEmbedding(DIMS, 1) }] as Template[],
    thresholds: RELAXED,
    challengePool: ['smile'],
    challengeTimeoutMs: 10000,
    gateFrames: 1,
    embedFrames: 1,
    rng: () => 0,
    ...over,
  };
}

function models(liveLogit = 5, embedSeed = 1): PipelineModels {
  return {
    embedding: stubModel(seededEmbedding(DIMS, embedSeed)),
    liveness: { branches: [livenessStub(liveLogit)] },
  };
}

const frame = makeFrame(32, 128);

describe('FaceAuthSession', () => {
  it('authenticates on the happy path', async () => {
    const s = new FaceAuthSession(models(), makeOptions());
    await s.process(frame, [makeFace()], 1); // gate → challenge (smile)
    await s.process(frame, [makeFace({ mouthRatio: 0.4 })], 2); // neutral
    await s.process(frame, [makeFace({ mouthRatio: 1.0 })], 3); // smile → analyzing
    const tick = await s.process(frame, [makeFace()], 4); // liveness + embed → match
    expect(tick.done).toBe(true);
    expect(tick.result?.ok).toBe(true);
    expect(tick.result?.personnelId).toBe('A');
    expect(tick.result?.challengePassed).toBe(true);
  });

  it('fails on spoof (liveness) — after the challenge passes', async () => {
    const s = new FaceAuthSession(models(-5), makeOptions());
    await s.process(frame, [makeFace()], 1); // → challenge
    await s.process(frame, [makeFace({ mouthRatio: 0.4 })], 2);
    await s.process(frame, [makeFace({ mouthRatio: 1.0 })], 3); // → analyzing
    const tick = await s.process(frame, [makeFace()], 4); // liveness fails
    expect(tick.done).toBe(true);
    expect(tick.result?.failureReason).toBe('liveness_failed');
  });

  it('fails when the challenge is not satisfied (lost tracking)', async () => {
    const s = new FaceAuthSession(models(), makeOptions());
    await s.process(frame, [makeFace()], 1); // → challenge
    const tick = await s.process(frame, [], 2); // no face → lost tracking
    expect(tick.result?.failureReason).toBe('challenge_failed');
  });

  it('reports no_template when roster is empty', async () => {
    const s = new FaceAuthSession(models(), makeOptions({ templates: [] }));
    await s.process(frame, [makeFace()], 1);
    await s.process(frame, [makeFace({ mouthRatio: 0.4 })], 2);
    await s.process(frame, [makeFace({ mouthRatio: 1.0 })], 3);
    const tick = await s.process(frame, [makeFace()], 4);
    expect(tick.result?.failureReason).toBe('no_template');
  });

  it('reports no_match when the probe is far from all templates', async () => {
    // embedding model emits a vector unrelated to template A
    const s = new FaceAuthSession(models(5, 9999), makeOptions());
    await s.process(frame, [makeFace()], 1);
    await s.process(frame, [makeFace({ mouthRatio: 0.4 })], 2);
    await s.process(frame, [makeFace({ mouthRatio: 1.0 })], 3);
    const tick = await s.process(frame, [makeFace()], 4);
    expect(tick.result?.ok).toBe(false);
    expect(tick.result?.failureReason).toBe('no_match');
  });

  it('cancel produces a cancelled result', () => {
    const s = new FaceAuthSession(models(), makeOptions());
    const tick = s.cancel();
    expect(tick.result?.failureReason).toBe('cancelled');
  });
});
