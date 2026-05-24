import { l2Normalize } from '../embeddings';

/**
 * Cancelable (revocable) biometric transform.
 *
 * Raw face embeddings are never stored or matched directly. Instead we apply a
 * keyed, irreversible-ish transform so that a stolen template is useless in any
 * other deployment and can be revoked by rotating the tenant key. We use a
 * seeded random projection followed by L2 renormalization — this preserves
 * cosine geometry (so matching still works) while binding the template to a
 * secret projection key.
 *
 * The SAME key + output dimension must be used for server enrollment and device
 * matching, or vectors won't be comparable.
 */

export interface CancelableParams {
  /** Tenant/deployment secret driving the projection (rotate to revoke). */
  key: string;
  /** Output dimensionality (<= input dims; typically equal). */
  outputDims: number;
}

/**
 * Deterministically build an outputDims × inputDims projection matrix from the
 * key using a fast seeded PRNG (xorshift128), entries in {-1,+1}/sqrt(dims)
 * (a sign-random / Achlioptas-style projection — cheap and norm-preserving).
 */
function buildProjection(key: string, outputDims: number, inputDims: number): Int8Array {
  const m = new Int8Array(outputDims * inputDims);
  const rng = makePrng(key);
  for (let i = 0; i < m.length; i++) {
    m[i] = rng() & 1 ? 1 : -1;
  }
  return m;
}

let cachedKey = '';
let cachedDims = 0;
let cachedInput = 0;
let cachedMatrix: Int8Array | null = null;

/** Apply the cancelable transform to an L2-normalized embedding. */
export function transform(embedding: Float32Array, params: CancelableParams): Float32Array {
  const inputDims = embedding.length;
  if (
    !cachedMatrix ||
    cachedKey !== params.key ||
    cachedDims !== params.outputDims ||
    cachedInput !== inputDims
  ) {
    cachedMatrix = buildProjection(params.key, params.outputDims, inputDims);
    cachedKey = params.key;
    cachedDims = params.outputDims;
    cachedInput = inputDims;
  }
  const scale = 1 / Math.sqrt(inputDims);
  const out = new Float32Array(params.outputDims);
  for (let o = 0; o < params.outputDims; o++) {
    let acc = 0;
    const row = o * inputDims;
    for (let i = 0; i < inputDims; i++) {
      acc += cachedMatrix[row + i]! * embedding[i]!;
    }
    out[o] = acc * scale;
  }
  return l2Normalize(out);
}

/** xorshift128 seeded from a string (FNV-1a → 4 lanes). Deterministic, fast. */
function makePrng(seed: string): () => number {
  let h = 0x811c9dc5 >>> 0;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  let x = h || 1;
  let y = Math.imul(h, 0x9e3779b1) >>> 0 || 2;
  let z = Math.imul(h, 0x85ebca6b) >>> 0 || 3;
  let w = Math.imul(h, 0xc2b2ae35) >>> 0 || 4;
  return () => {
    const t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    w = (w ^ (w >>> 19) ^ (t ^ (t >>> 8))) >>> 0;
    return w;
  };
}
