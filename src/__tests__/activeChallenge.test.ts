import { ActiveChallenge } from '../activeChallenge';
import { DEFAULT_THRESHOLDS } from '../config';
import { makeFace } from './_helpers';

function fixedChallenge(
  kind: 'blink' | 'smile' | 'turn_head',
  timeoutMs = 6000,
  lostFrameTolerance = 0,  // 0 = fail on first missing frame (deterministic in tests)
) {
  return new ActiveChallenge(
    {
      pool: [kind],
      timeoutMs,
      thresholds: { ...DEFAULT_THRESHOLDS, challengeLostFrameTolerance: lostFrameTolerance },
    },
    0,
    () => 0,  // rng → 0 picks the first pool entry deterministically
  );
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

  it('fails on first missing frame when tolerance is 0', () => {
    const c = fixedChallenge('smile', 6000, 0);
    expect(c.feed(null, 10)).toBe('failed');
  });

  it('tolerates brief tracking loss within the grace window', () => {
    const c = fixedChallenge('smile', 6000, 2);  // tolerance = 2 → fail after 3rd miss
    expect(c.feed(null, 10)).toBe('pending');     // 1st miss — within tolerance
    expect(c.feed(null, 20)).toBe('pending');     // 2nd miss — within tolerance
    expect(c.feed(null, 30)).toBe('failed');      // 3rd miss — exceeds tolerance
  });

  it('resets lost-frame counter on face reacquire', () => {
    const c = fixedChallenge('smile', 6000, 1);  // tolerance = 1
    expect(c.feed(null, 10)).toBe('pending');     // 1 miss — tolerated
    expect(c.feed(makeFace({ mouthRatio: 0.4 }), 20)).toBe('pending');  // face back — counter reset
    expect(c.feed(null, 30)).toBe('pending');     // 1 miss again — tolerated (counter was reset)
    expect(c.feed(null, 40)).toBe('failed');      // 2nd consecutive miss — fails
  });
});
