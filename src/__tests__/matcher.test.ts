import {
  identify,
  verify,
  groupTemplates,
  identifyMultiFrame,
  verifyMultiFrame,
  type Template,
  type MultiFrameMatchConfig,
} from '../matcher';
import { seededEmbedding } from './_helpers';

// ── Shared fixtures ────────────────────────────────────────────────────────────

const DIMS = 128;
const a    = seededEmbedding(DIMS, 1);
const b    = seededEmbedding(DIMS, 2);
const far  = seededEmbedding(DIMS, 9999);

const templates: Template[] = [
  { personnelId: 'A', embedding: a },
  { personnelId: 'B', embedding: b },
];

// Helper: pre-filter as the pipeline does before calling verify.
const templatesFor = (id: string) => templates.filter((t) => t.personnelId === id);

// Default multi-frame config — generous thresholds so unit tests are not
// fragile against specific embedding geometry.
const cfg: MultiFrameMatchConfig = {
  coarseThreshold:    0.50,
  fineThreshold:      0.60,
  probeFrameCount:    3,
  templateFrameCount: 3,
  fusionStrategy:     'top_k',
  topK:               3,
};

// ── Single-frame matching (legacy path) ───────────────────────────────────────

describe('identify (single-frame)', () => {
  it('picks the closest above threshold', () => {
    const { match, best } = identify(a, templates, 0.65);
    expect(match?.personnelId).toBe('A');
    expect(best?.personnelId).toBe('A');
    expect(best!.score).toBeCloseTo(1, 5);
  });

  it('returns null match when nothing clears threshold', () => {
    const { match } = identify(far, templates, 0.99);
    expect(match).toBeNull();
  });
});

describe('verify (single-frame)', () => {
  it('accepts the true identity', () => {
    expect(verify(a, templatesFor('A'), 0.65).ok).toBe(true);
  });

  it('rejects a wrong identity', () => {
    expect(verify(a, templatesFor('B'), 0.65).ok).toBe(false);
  });

  it('returns score: null (not 0) when the claimed id has no template', () => {
    const result = verify(a, templatesFor('ZZZ'), 0.65);
    expect(result.ok).toBe(false);
    expect(result.score).toBeNull();
  });

  it('returns a real score when template exists but does not meet threshold', () => {
    const result = verify(a, templatesFor('B'), 0.99); // high threshold — will fail
    expect(result.ok).toBe(false);
    expect(result.score).not.toBeNull();
    expect(typeof result.score).toBe('number');
  });
});

// ── groupTemplates ─────────────────────────────────────────────────────────────

describe('groupTemplates', () => {
  it('groups flat rows into per-person entries', () => {
    const flat: Template[] = [
      { personnelId: 'A', embedding: a },
      { personnelId: 'A', embedding: seededEmbedding(DIMS, 11) },
      { personnelId: 'B', embedding: b },
    ];
    const groups = groupTemplates(flat);
    expect(groups).toHaveLength(2);
    const ga = groups.find((g) => g.personnelId === 'A')!;
    expect(ga.embeddings).toHaveLength(2);
    const gb = groups.find((g) => g.personnelId === 'B')!;
    expect(gb.embeddings).toHaveLength(1);
  });

  it('returns empty array for empty input', () => {
    expect(groupTemplates([])).toEqual([]);
  });
});

// ── identifyMultiFrame ────────────────────────────────────────────────────────

describe('identifyMultiFrame', () => {
  // Registry with person A (3 shots near `a`) and person B (1 shot = `b`).
  const registry = groupTemplates([
    { personnelId: 'A', embedding: a },
    { personnelId: 'A', embedding: seededEmbedding(DIMS, 11) },
    { personnelId: 'A', embedding: seededEmbedding(DIMS, 12) },
    { personnelId: 'B', embedding: b },
  ]);

  it('returns the correct match for a known probe', () => {
    const { match, candidatesEvaluated } = identifyMultiFrame([a, a, a], registry, cfg);
    expect(match).not.toBeNull();
    expect(match!.personnelId).toBe('A');
    expect(typeof match!.score).toBe('number');
    expect(candidatesEvaluated).toBeGreaterThanOrEqual(1);
  });

  it('short-circuits to 0 candidates when coarse filter rejects everything', () => {
    // `far` is orthogonal to both a and b — coarse pass should reject both.
    const strictCfg: MultiFrameMatchConfig = { ...cfg, coarseThreshold: 0.99 };
    const { match, candidatesEvaluated } = identifyMultiFrame([far], registry, strictCfg);
    expect(match).toBeNull();
    expect(candidatesEvaluated).toBe(0);
  });

  it('returns null match when fine threshold is not cleared', () => {
    const strictCfg: MultiFrameMatchConfig = { ...cfg, coarseThreshold: 0.0, fineThreshold: 0.9999 };
    const { match, candidatesEvaluated } = identifyMultiFrame([a], registry, strictCfg);
    expect(match).toBeNull();
    // Candidates still evaluated (coarse let them through)
    expect(candidatesEvaluated).toBeGreaterThan(0);
  });

  it('returns { match: null, candidatesEvaluated: 0 } for empty registry', () => {
    const result = identifyMultiFrame([a], [], cfg);
    expect(result).toEqual({ match: null, candidatesEvaluated: 0 });
  });

  it('returns { match: null, candidatesEvaluated: 0 } for empty probes', () => {
    const result = identifyMultiFrame([], registry, cfg);
    expect(result).toEqual({ match: null, candidatesEvaluated: 0 });
  });

  it('fusion strategy: max yields score ≥ mean', () => {
    const maxCfg:  MultiFrameMatchConfig = { ...cfg, fusionStrategy: 'max',  fineThreshold: 0 };
    const meanCfg: MultiFrameMatchConfig = { ...cfg, fusionStrategy: 'mean', fineThreshold: 0 };
    const { match: mMax  } = identifyMultiFrame([a, a, a], registry, maxCfg);
    const { match: mMean } = identifyMultiFrame([a, a, a], registry, meanCfg);
    // max ≥ mean always holds for a non-trivial score list
    expect(mMax!.score).toBeGreaterThanOrEqual(mMean!.score);
  });
});

// ── verifyMultiFrame ──────────────────────────────────────────────────────────

describe('verifyMultiFrame', () => {
  const groupA = { personnelId: 'A', embeddings: [a, seededEmbedding(DIMS, 11), seededEmbedding(DIMS, 12)] };
  const groupB = { personnelId: 'B', embeddings: [b] };

  it('accepts the correct identity', () => {
    const result = verifyMultiFrame([a, a, a], groupA, cfg);
    expect(result.ok).toBe(true);
    expect(typeof result.score).toBe('number');
    expect(result.score).not.toBeNull();
  });

  it('rejects a wrong identity', () => {
    // Probe matches A but group is B — scores will be low
    const strictCfg: MultiFrameMatchConfig = { ...cfg, fineThreshold: 0.80 };
    const result = verifyMultiFrame([a, a, a], groupB, strictCfg);
    expect(result.ok).toBe(false);
  });

  it('returns score: null when personnelGroup is null (not enrolled)', () => {
    const result = verifyMultiFrame([a], null, cfg);
    expect(result.ok).toBe(false);
    expect(result.score).toBeNull();
  });

  it('returns score: null when personnelGroup has empty embeddings', () => {
    const result = verifyMultiFrame([a], { personnelId: 'X', embeddings: [] }, cfg);
    expect(result.ok).toBe(false);
    expect(result.score).toBeNull();
  });

  it('returns a real (non-null) score even when below fine threshold', () => {
    // Use `far` probe (orthogonal to groupA) so fused score won't clear the threshold.
    // Set coarseThreshold: 0 so the coarse filter doesn't block it — we want to reach
    // the fine threshold gate and confirm the fallback score path is exercised.
    const testCfg: MultiFrameMatchConfig = { ...cfg, coarseThreshold: 0, fineThreshold: 0.9999 };
    const result = verifyMultiFrame([far, far, far], groupA, testCfg);
    expect(result.ok).toBe(false);
    expect(result.score).not.toBeNull(); // score is present for diagnostics even on rejection
    expect(typeof result.score).toBe('number');
  });
});
