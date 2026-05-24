import { DEFAULT_THRESHOLDS } from '../config';
import { evaluateQuality } from '../qualityGate';
import { makeFace, makeFrame } from './_helpers';

const T = DEFAULT_THRESHOLDS;

describe('quality gate', () => {
  it('flags no_face', () => {
    const r = evaluateQuality({ faces: [], frame: makeFrame() }, T);
    expect(r.ok).toBe(false);
    expect(r.issues).toContain('no_face');
  });

  it('flags multiple_faces', () => {
    const r = evaluateQuality({ faces: [makeFace(), makeFace()], frame: makeFrame() }, T);
    expect(r.issues).toContain('multiple_faces');
  });

  it('flags face_too_small', () => {
    const small = makeFace({ box: { x: 0.45, y: 0.45, width: 0.05, height: 0.05 } });
    const r = evaluateQuality({ faces: [small], frame: makeFrame() }, T);
    expect(r.issues).toContain('face_too_small');
  });

  it('flags off_center', () => {
    // Bypass minFaceArea so we can isolate the centering check.
    // Center at (0.025, 0.025) — inside the 0.10 margin → off_center.
    const T_noArea = { ...T, minFaceArea: 0 };
    const edge = makeFace({ box: { x: 0.0, y: 0.0, width: 0.05, height: 0.05 } });
    const r = evaluateQuality({ faces: [edge], frame: makeFrame() }, T_noArea);
    expect(r.issues).toContain('off_center');
  });

  it('flags bad_pose', () => {
    const turned = makeFace({ headPose: { yaw: 45 } });
    const r = evaluateQuality({ faces: [turned], frame: makeFrame() }, T);
    expect(r.issues).toContain('bad_pose');
  });

  it('flags too_dark', () => {
    const r = evaluateQuality({ faces: [makeFace()], frame: makeFrame(32, 10) }, T);
    expect(r.issues).toContain('too_dark');
  });

  it('uniform frame is blurry (low sharpness)', () => {
    const r = evaluateQuality({ faces: [makeFace()], frame: makeFrame(32, 128) }, T);
    expect(r.issues).toContain('blurry');
  });

  it('passes with relaxed thresholds on a clean face', () => {
    const relaxed = { ...T, minSharpness: 0, minBrightness: 0, minFaceArea: 0 };
    const r = evaluateQuality({ faces: [makeFace()], frame: makeFrame(32, 128) }, relaxed);
    expect(r.ok).toBe(true);
  });
});
