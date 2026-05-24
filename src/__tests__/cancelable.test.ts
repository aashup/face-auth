import { cosineSimilarity } from '../embeddings';
import { transform } from '../secure/cancelable';
import { seededEmbedding } from './_helpers';

describe('cancelable transform', () => {
  const DIMS = 512;
  const cp = { key: 'tenant-key', outputDims: DIMS };

  it('preserves cosine geometry', () => {
    const a = seededEmbedding(DIMS, 1);
    const noise = seededEmbedding(DIMS, 99);
    const probe = seededEmbedding(DIMS, 1); // same seed → identical
    // perturb slightly
    const perturbed = new Float32Array(DIMS);
    for (let i = 0; i < DIMS; i++) perturbed[i] = probe[i]! * 0.95 + noise[i]! * 0.05;

    const before = cosineSimilarity(a, perturbed);
    const after = cosineSimilarity(transform(a, cp), transform(perturbed, cp));
    // Sign-random projection preserves cosine up to JL variance (~1/sqrt(dims)).
    expect(Math.abs(before - after)).toBeLessThan(0.1);
  });

  it('isolates templates across different keys', () => {
    const a = seededEmbedding(DIMS, 7);
    const ta = transform(a, cp);
    const taOther = transform(a, { key: 'different-key', outputDims: DIMS });
    expect(cosineSimilarity(ta, taOther)).toBeLessThan(0.3);
  });

  it('is deterministic for the same key', () => {
    const a = seededEmbedding(DIMS, 3);
    expect(Array.from(transform(a, cp))).toEqual(Array.from(transform(a, cp)));
  });
});
