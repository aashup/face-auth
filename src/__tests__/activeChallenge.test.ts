import { ActiveChallenge } from '../activeChallenge';
import { makeFace } from './_helpers';

function fixedChallenge(kind: 'blink' | 'smile' | 'turn_head', timeoutMs = 6000) {
  // rng → 0 picks the first pool entry deterministically.
  return new ActiveChallenge({ pool: [kind], timeoutMs }, 0, () => 0);
}

describe('active challenge', () => {
  it('blink passes on open→closed→open', () => {
    const c = fixedChallenge('blink');
    expect(c.feed(makeFace({ ear: 0.3 }), 10)).toBe('pending'); // open
    expect(c.feed(makeFace({ ear: 0.1 }), 20)).toBe('pending'); // closed
    expect(c.feed(makeFace({ ear: 0.3 }), 30)).toBe('passed'); // open again
  });

  it('smile passes on neutral→smiling', () => {
    const c = fixedChallenge('smile');
    expect(c.feed(makeFace({ mouthRatio: 0.4 }), 10)).toBe('pending');
    expect(c.feed(makeFace({ mouthRatio: 1.0 }), 20)).toBe('passed');
  });

  it('turn passes on yaw cross→return', () => {
    const c = fixedChallenge('turn_head');
    expect(c.feed(makeFace({ headPose: { yaw: 25 } }), 10)).toBe('pending');
    expect(c.feed(makeFace({ headPose: { yaw: 2 } }), 20)).toBe('passed');
  });

  it('fails on timeout', () => {
    const c = fixedChallenge('blink', 100);
    expect(c.feed(makeFace({ ear: 0.3 }), 50)).toBe('pending');
    expect(c.feed(makeFace({ ear: 0.1 }), 500)).toBe('failed');
  });

  it('fails on lost tracking', () => {
    const c = fixedChallenge('smile');
    expect(c.feed(null, 10)).toBe('failed');
  });
});
