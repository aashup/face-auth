const mockPost = jest.fn();
jest.mock(
  'axios',
  () => ({ __esModule: true, default: { create: () => ({ post: mockPost }) } }),
  { virtual: true },
);

import {
  getCachedManifest,
  getLastProvisionedAt,
  provision,
} from '../sync/provisioning';
import { getConfig, setConfig } from '../runtime';
import { loadAll, openTemplateStore } from '../secure/templateStore';
import { __resetAsyncStorage } from '../../__mocks__/async-storage';
import { __resetKeychain } from '../../__mocks__/react-native-keychain';

/** Encode a numeric vector as base64 little-endian float32 (server wire format). */
function enc(v: number[]): string {
  const f = new Float32Array(v);
  return Buffer.from(f.buffer, f.byteOffset, f.byteLength).toString('base64');
}

const CANCELABLE = { key: 'k', outputDims: 4 };
const OPTS = { awsSyncUrl: 'https://x', deviceToken: 't', cancelable: CANCELABLE };

describe('provisioning', () => {
  beforeEach(async () => {
    __resetAsyncStorage();
    __resetKeychain();
    mockPost.mockReset();
    await openTemplateStore();
    setConfig({ awsSyncUrl: 'https://x', deviceToken: 't' });
  });

  it('applies a full roster, thresholds, manifest and cursor', async () => {
    const manifest = { 'm.tflite': { sha256: 'abc', version: '1', input: [1, 1] } };
    mockPost.mockResolvedValue({
      data: {
        mode: 'full',
        templates: [
          { personnelId: 'A', embedding: enc([1, 0, 0, 0]) },
          { personnelId: 'B', embedding: enc([0, 1, 0, 0]) },
        ],
        thresholds: { matchCosine: 0.82 },
        manifest,
        syncedAt: 1000,
      },
    });

    const r = await provision(OPTS);
    expect(r.ok).toBe(true);
    expect(r.mode).toBe('full');
    expect(r.count).toBe(2);
    expect(loadAll().map((t) => t.personnelId).sort()).toEqual(['A', 'B']);
    expect(getConfig().thresholds.matchCosine).toBe(0.82);
    expect(getCachedManifest()).toEqual(manifest);
    expect(getLastProvisionedAt()).toBe(1000);
  });

  it('sends the stored cursor and applies a delta (upsert + removal)', async () => {
    // First, a full sync sets the cursor to 1000.
    mockPost.mockResolvedValueOnce({
      data: { mode: 'full', templates: [{ personnelId: 'A', embedding: enc([1, 0, 0, 0]) }], syncedAt: 1000 },
    });
    await provision(OPTS);

    // Then a delta: add B, remove A.
    mockPost.mockResolvedValueOnce({
      data: {
        mode: 'delta',
        upserts: [{ personnelId: 'B', embeddings: [enc([0, 1, 0, 0])] }],
        removals: ['A'],
        syncedAt: 2000,
      },
    });
    const r = await provision(OPTS);

    expect(r.mode).toBe('delta');
    expect(mockPost.mock.calls[1][1]).toMatchObject({ since: 1000 });
    expect(loadAll().map((t) => t.personnelId)).toEqual(['B']);
    expect(getLastProvisionedAt()).toBe(2000);
  });

  it('forceFull ignores the stored cursor', async () => {
    mockPost.mockResolvedValueOnce({ data: { mode: 'full', templates: [], syncedAt: 1000 } });
    await provision(OPTS);
    mockPost.mockResolvedValueOnce({ data: { mode: 'full', templates: [], syncedAt: 3000 } });
    await provision({ ...OPTS, forceFull: true });
    expect(mockPost.mock.calls[1][1]).toMatchObject({ since: 0 });
  });

  it('does not advance the cursor on failure (resumable)', async () => {
    mockPost.mockResolvedValueOnce({ data: { mode: 'full', templates: [], syncedAt: 1000 } });
    await provision(OPTS);
    mockPost.mockRejectedValueOnce(new Error('offline'));
    const r = await provision(OPTS);
    expect(r.ok).toBe(false);
    expect(getLastProvisionedAt()).toBe(1000); // unchanged
  });
});
