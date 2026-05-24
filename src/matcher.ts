import { dotProduct } from './embeddings';
import type { FusionStrategy } from './config';

/**
 * 1:1 (verify) and 1:N (identify) matching over locally-held templates.
 *
 * A template is one personnel's enrolled embedding (already L2-normalized, and
 * already passed through the cancelable transform — see secure/cancelable.ts).
 * The probe embedding fed in here must have gone through the *same* transform.
 *
 * Performance notes
 * ─────────────────
 * • `dotProduct` is used instead of a full cosine-similarity computation.
 *   Because every embedding (probe and template) is L2-normalized before
 *   storage, the dot product IS the cosine similarity — skipping the two
 *   magnitude divisions saves ~2 × N multiplications per comparison.
 * • `verify` accepts a **pre-filtered** slice (only the templates that belong
 *   to the claimed identity), so 1:1 lookups are O(K) where K is the number
 *   of enrolled shots for that person, not O(all templates in the store).
 *   The caller is responsible for the O(1) map lookup before calling verify.
 * • Both inner loops use a numeric index instead of `for…of` to avoid the
 *   iterator-protocol overhead in tight V8 numeric loops.
 */

// ── Shared types ─────────────────────────────────────────────────────────────

export interface Template {
  personnelId: string;
  embedding: Float32Array;
}

export interface Match {
  personnelId: string;
  score: number;
}

// ── Multi-frame matching types ────────────────────────────────────────────────

/**
 * Templates grouped by person — one entry per enrolled identity, carrying all
 * their enrolled frame embeddings.  Built from a flat `Template[]` via
 * `groupTemplates()`.
 */
export interface MultiFrameTemplate {
  personnelId: string;
  /** All enrolled embeddings for this person (each L2-normalized). */
  embeddings: Float32Array[];
}

/** Tuning knobs for the 3-step cascading match engine. */
export interface MultiFrameMatchConfig {
  /** Cosine floor for the fast coarse pass (probe[0] vs template[0]). */
  coarseThreshold: number;
  /** Final acceptance threshold applied after score fusion. */
  fineThreshold: number;
  /** How many live probe frames to include in the cross-matrix. */
  probeFrameCount: number;
  /** Max enrolled frames per person to include in the cross-matrix. */
  templateFrameCount: number;
  /** Strategy for collapsing the M×K matrix into a single score. */
  fusionStrategy: FusionStrategy;
  /** K for `top_k` fusion — ignored for other strategies. */
  topK: number;
}

/** Result of a successful multi-frame match. */
export interface FinalMatch {
  personnelId: string;
  score: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Group a flat `Template[]` (as returned by `templateStore.loadAll()`) into
 * per-person `MultiFrameTemplate[]` for use with the cascading match engine.
 * Preserves enrollment order within each person's embeddings slice.
 */
export function groupTemplates(flat: Template[]): MultiFrameTemplate[] {
  const map = new Map<string, Float32Array[]>();
  for (let i = 0; i < flat.length; i++) {
    const t = flat[i]!;
    let arr = map.get(t.personnelId);
    if (!arr) { arr = []; map.set(t.personnelId, arr); }
    arr.push(t.embedding);
  }
  const result: MultiFrameTemplate[] = [];
  map.forEach((embeddings, personnelId) => result.push({ personnelId, embeddings }));
  return result;
}

/**
 * Collapse a score matrix into a single confidence value.
 * All computation is synchronous — `dotProduct` produces no I/O.
 */
function fuseScores(scores: number[], strategy: FusionStrategy, topK: number): number {
  if (scores.length === 0) return 0;
  switch (strategy) {
    case 'max': {
      let m = scores[0]!;
      for (let i = 1; i < scores.length; i++) if (scores[i]! > m) m = scores[i]!;
      return m;
    }
    case 'top_k': {
      const k = Math.min(topK, scores.length);
      // Full descending sort; k is typically small (≤ 5) so this is cheap.
      const sorted = scores.slice().sort((a, b) => b - a);
      let s = 0;
      for (let i = 0; i < k; i++) s += sorted[i]!;
      return s / k;
    }
    default: { // 'mean'
      let s = 0;
      for (let i = 0; i < scores.length; i++) s += scores[i]!;
      return s / scores.length;
    }
  }
}

// ── Single-frame matching (legacy / backward-compat) ─────────────────────────

/**
 * Best match across all templates (identify / 1:N).
 *
 * Returns the highest-scoring template and, separately, whether that score
 * clears the threshold.
 */
export function identify(
  probe: Float32Array,
  templates: Template[],
  threshold: number,
): { match: Match | null; best: Match | null } {
  let best: Match | null = null;
  for (let i = 0; i < templates.length; i++) {
    const t = templates[i]!;
    const score = dotProduct(probe, t.embedding);
    if (!best || score > best.score) {
      best = { personnelId: t.personnelId, score };
    }
  }
  const match = best && best.score >= threshold ? best : null;
  return { match, best };
}

/**
 * Verify a probe against one claimed identity (1:1).
 *
 * **`personnelTemplates` must already be filtered** to contain only the
 * embeddings that belong to the claimed identity.  Pass an empty array (not
 * the whole template pool) when no templates exist for that person.
 *
 * Returns `score: null` (not 0) when the slice is empty so callers can
 * distinguish "no template enrolled" from a genuine cosine score of 0.0
 * (orthogonal vectors), which is a real, if terrible, match result.
 */
export function verify(
  probe: Float32Array,
  personnelTemplates: Template[],
  threshold: number,
): { ok: boolean; score: number | null } {
  if (personnelTemplates.length === 0) {
    return { ok: false, score: null }; // no template enrolled — distinct from score 0.0
  }
  let best = -Infinity;
  for (let i = 0; i < personnelTemplates.length; i++) {
    const s = dotProduct(probe, personnelTemplates[i]!.embedding);
    if (s > best) best = s;
  }
  return { ok: best >= threshold, score: best };
}

// ── Multi-frame cascading match engine ────────────────────────────────────────

/**
 * 3-step cascading 1:N match engine.
 *
 * Step 1 — Coarse filter: probe[0] vs each person's first enrolled template.
 *           Identities below `cfg.coarseThreshold` are eliminated cheaply.
 * Step 2 — Cross-product score matrix: all active probe frames ×
 *           all enrolled frames for each surviving candidate.
 * Step 3 — Score fusion + fine threshold: the M×K matrix is collapsed by
 *           `cfg.fusionStrategy`; only candidates above `cfg.fineThreshold`
 *           are accepted.  The best-scoring identity wins.
 *
 * The entire function is synchronous — `dotProduct` produces no I/O and
 * Promise.all would only add microtask overhead for pure CPU work.
 */
export function identifyMultiFrame(
  probes: Float32Array[],
  registry: MultiFrameTemplate[],
  cfg: MultiFrameMatchConfig,
): { match: FinalMatch | null; candidatesEvaluated: number } {
  if (probes.length === 0 || registry.length === 0) {
    return { match: null, candidatesEvaluated: 0 };
  }

  const primaryProbe = probes[0]!;
  const activeProbes = probes.slice(0, cfg.probeFrameCount);

  // ── Step 1: Coarse filter ─────────────────────────────────────────────────
  const candidates: MultiFrameTemplate[] = [];
  for (let i = 0; i < registry.length; i++) {
    const r = registry[i]!;
    if (r.embeddings.length > 0 &&
        dotProduct(primaryProbe, r.embeddings[0]!) >= cfg.coarseThreshold) {
      candidates.push(r);
    }
  }
  if (candidates.length === 0) return { match: null, candidatesEvaluated: 0 };

  // ── Step 2: Cross-product score matrix per candidate ──────────────────────
  let best: FinalMatch | null = null;
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i]!;
    const templates = c.embeddings.slice(0, cfg.templateFrameCount);
    const matrix: number[] = [];
    for (let p = 0; p < activeProbes.length; p++) {
      for (let t = 0; t < templates.length; t++) {
        matrix.push(dotProduct(activeProbes[p]!, templates[t]!));
      }
    }
    const score = fuseScores(matrix, cfg.fusionStrategy, cfg.topK);
    if (!best || score > best.score) {
      best = { personnelId: c.personnelId, score };
    }
  }

  // ── Step 3: Fine threshold gate ───────────────────────────────────────────
  const match = best && best.score >= cfg.fineThreshold ? best : null;
  return { match, candidatesEvaluated: candidates.length };
}

/**
 * Multi-frame 1:1 verification.
 *
 * @param probes           Live probe frames (each L2-normalized).
 * @param personnelGroup   Pre-grouped templates for the claimed identity, or
 *                         `null` if the identity has no enrolled templates.
 * @param cfg              Match configuration (same object as `identifyMultiFrame`).
 *
 * Returns `score: null` when `personnelGroup` is null or empty — unambiguous
 * "not enrolled" signal, distinct from a real cosine score of 0.0.
 */
export function verifyMultiFrame(
  probes: Float32Array[],
  personnelGroup: MultiFrameTemplate | null,
  cfg: MultiFrameMatchConfig,
): { ok: boolean; score: number | null } {
  if (!personnelGroup || personnelGroup.embeddings.length === 0) {
    return { ok: false, score: null };
  }
  const { match } = identifyMultiFrame(probes, [personnelGroup], cfg);
  // If the coarse filter rejects (probe[0] vs template[0] < coarseThreshold),
  // match is null but we still want to report a real score for diagnostics.
  // Re-run without coarse gate by lowering threshold to -Infinity on a single entry.
  if (!match) {
    // compute fused score regardless of coarse threshold for the score field
    const activeProbes = probes.slice(0, cfg.probeFrameCount);
    const templates = personnelGroup.embeddings.slice(0, cfg.templateFrameCount);
    const matrix: number[] = [];
    for (let p = 0; p < activeProbes.length; p++)
      for (let t = 0; t < templates.length; t++)
        matrix.push(dotProduct(activeProbes[p]!, templates[t]!));
    const score = fuseScores(matrix, cfg.fusionStrategy, cfg.topK);
    return { ok: false, score };
  }
  return { ok: true, score: match.score };
}
