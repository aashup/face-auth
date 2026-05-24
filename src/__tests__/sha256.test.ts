import { hmacSha256, sha256, toHex, utf8 } from '../secure/sha256';

describe('sha256', () => {
  it('matches known vectors', () => {
    expect(toHex(sha256(utf8('')))).toBe(
      'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    );
    expect(toHex(sha256(utf8('abc')))).toBe(
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    );
    // 56 chars exercises the multi-block padding edge.
    expect(toHex(sha256(utf8('a'.repeat(56))))).toBe(
      'b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a',
    );
  });

  it('hmac matches a known vector', () => {
    expect(
      toHex(hmacSha256(utf8('key'), utf8('The quick brown fox jumps over the lazy dog'))),
    ).toBe('f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8');
  });
});
