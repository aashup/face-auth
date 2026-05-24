const mockPost = jest.fn();
jest.mock(
  'axios',
  () => ({ __esModule: true, default: { create: () => ({ post: mockPost }) } }),
  { virtual: true },
);

import { FaceAuth } from '../faceAuth';
import { sha256, toHex } from '../secure/sha256';
import { __resetAsyncStorage } from '../../__mocks__/async-storage';
import { __resetKeychain } from '../../__mocks__/react-native-keychain';

const BASE = { awsSyncUrl: 'https://x', deviceToken: 't', templateTtlMs: 1e12 };

describe('FaceAuth runtime (Phase 6 wiring)', () => {
  beforeEach(() => {
    __resetAsyncStorage();
    __resetKeychain();
    mockPost.mockReset();
  });

  afterEach(async () => {
    await FaceAuth.dispose();
  });

  it('needsReprovision is true before any provision', async () => {
    await FaceAuth.init(BASE);
    expect(FaceAuth.needsReprovision()).toBe(true);
  });

  it('needsReprovision is false right after a successful provision', async () => {
    await FaceAuth.init(BASE);
    mockPost.mockResolvedValue({ data: { mode: 'full', templates: [], syncedAt: Date.now() } });
    await FaceAuth.provision();
    expect(FaceAuth.needsReprovision()).toBe(false);
  });

  it('init throws when a model fails the integrity check', async () => {
    const manifest = { 'm.tflite': { sha256: 'deadbeef', version: '1', input: [1, 1] } };
    await expect(
      FaceAuth.init({ ...BASE, modelManifest: manifest, readModelBytes: async () => new Uint8Array([1, 2, 3]) }),
    ).rejects.toThrow(/integrity/);
  });

  it('init succeeds when the model digest matches the manifest', async () => {
    const bytes = new Uint8Array([1, 2, 3, 4]);
    const manifest = { 'm.tflite': { sha256: toHex(sha256(bytes)), version: '1', input: [1, 1] } };
    await expect(
      FaceAuth.init({ ...BASE, modelManifest: manifest, readModelBytes: async () => bytes }),
    ).resolves.toBeUndefined();
  });
});
