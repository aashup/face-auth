# Configuration Reference

Complete reference for `FaceAuthConfig` and `Thresholds`. Pass overrides to `FaceAuth.init()`.

---

## `FaceAuthConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `awsSyncUrl` | `string` | — | AWS endpoint for attendance sync and template provisioning. Omit for fully offline deployments. |
| `deviceToken` | `string` | — | Per-device bearer token. Required when `awsSyncUrl` is set. |
| `deviceId` | `string` | `'unknown-device'` | Stable identifier stamped onto every attendance event. |
| `debug` | `boolean` | `false` | Enable SDK debug logging to Metro / logcat / Xcode console. |
| `thresholds` | `Partial<Thresholds>` | see below | Override any subset of decision thresholds. |
| `challenges` | `Challenge[]` | `['blink','smile','turn_head']` | Pool from which one challenge is randomly selected per auth. |
| `syncIntervalMs` | `number` | `300000` (5 min) | Background sync cadence in ms. Also fires on network reconnect. |
| `purgeAfterSync` | `boolean` | `true` | Delete local attendance rows after the server acknowledges them. |
| `challengeTimeoutMs` | `number` | `600000` (10 min) | Max time in ms to complete the active challenge before timeout. |
| `cancelableKey` | `string` | hardware key | Secret driving the cancelable template transform. Rotate this key to revoke all enrolled templates. |
| `cancelableDims` | `number` | `512` | Output dimensionality of the cancelable biohash transform. |
| `templateTtlMs` | `number` | `604800000` (7 days) | `needsReprovision()` returns `true` when templates are older than this. |
| `getAttestationToken` | `() => Promise<string \| null>` | — | Returns a Play Integrity / App Attest token included in provisioning requests. |
| `modelManifest` | `ModelManifest` | — | Bundled model manifest (e.g. `require('../models/manifest.json')`). Required for boot-time integrity check. |
| `readModelBytes` | `ModelReader` | — | Host-supplied reader for bundled .tflite bytes. When provided with `modelManifest`, `init()` verifies all model SHA-256 hashes. |

---

## `Thresholds`

All thresholds can be overridden via `FaceAuthConfig.thresholds`. The defaults are conservative starting points — calibrate against a validation set for your deployment.

### Detection & Frame Quality

| Field | Default | Controls | Effect of raising | Effect of lowering |
|-------|---------|----------|-------------------|-------------------|
| `detectScore` | `0.8` | Minimum detector confidence to accept a face | Fewer false detections; may miss real faces at angle | More faces accepted; may include false positives |
| `minFaceArea` | `0.4` | Minimum fraction of frame area the face box must cover | User must be closer to camera | Accepts smaller/more distant faces |
| `centerMargin` | `0.10` | Normalized margin from each edge within which face center must fall (0.10 = middle 80% of frame) | Tighter centering required | Accepts off-center faces |
| `maxYaw` | `30` | Max head yaw (left/right rotation) in degrees | Stricter pose; must face more directly | Accepts more angled poses |
| `maxPitch` | `50` | Max head pitch (up/down tilt) in degrees | Must look more level | Accepts more vertical tilt |
| `maxRoll` | `50` | Max head roll (side tilt) in degrees | Must be more upright | Accepts more head tilt |
| `minBrightness` | `50` | Minimum mean luma (0–255) | Requires brighter lighting | Accepts dimmer frames |
| `maxBrightness` | `230` | Maximum mean luma (0–255) | Rejects fewer overexposed frames | Rejects overexposed frames sooner |
| `minSharpness` | `80` | Laplacian variance floor (higher = sharper) | Requires crisper frames | Accepts more blurry frames |

### Liveness

| Field | Default | Controls | Effect of raising | Effect of lowering |
|-------|---------|----------|-------------------|-------------------|
| `spoofReject` | `0.5` | Reject if spoof probability ≥ this value. Real faces score spoof ≈ 0.05–0.15; spoofs ≈ 0.85–0.99 | More aggressive spoof rejection; risk rejecting real faces | More permissive; accept marginal liveness scores |
| `livenessFrames` | `4` | Number of frames sampled before making a liveness decision (scores are averaged) | More stable decision; slightly slower | Faster decision; less stable |

> **Warning:** Never set `spoofReject` below `0.3` in production — legitimate live faces will be rejected.

### Multi-Frame Matching

| Field | Default | Controls | Effect of raising | Effect of lowering |
|-------|---------|----------|-------------------|-------------------|
| `coarseThreshold` | `0.55` | Cosine similarity floor for the fast 1:N pre-filter (probe[0] vs each person's first template) | Fewer candidates reach the cross-matrix; faster but higher false-reject risk | More candidates pass to step 2; more CPU, fewer false rejects |
| `fineThreshold` | `0.72` | Final acceptance threshold after score fusion | Stricter matching; fewer false accepts | More permissive; higher false-accept rate |
| `probeFrameCount` | `3` | Number of live probe frames fed into the cross-matrix | More stable probe; slightly more latency | Faster; less stable |
| `templateFrameCount` | `3` | Max enrolled frames per person used in cross-matrix | More enrolled angles used; more CPU | Fewer enrolled frames compared; faster |
| `fusionStrategy` | `'top_k'` | How to collapse the M×K score matrix: `'mean'`, `'max'`, or `'top_k'` | — | — |
| `topK` | `3` | Number of top scores averaged when `fusionStrategy` is `'top_k'` | More conservative average | Average of fewer best-case pairs |
| `matchCosine` | `0.65` | *(Deprecated)* Legacy single-frame cosine threshold | — | — |

> **FaceNet-512 note:** Same-person cosine scores typically fall between 0.55–0.90. Setting `fineThreshold` above 0.85 without per-deployment calibration will produce significant false-reject rates.

---

## Challenge Types

| Challenge | Trigger Condition | How Detected |
|-----------|-------------------|--------------|
| `'blink'` | Eyes close then reopen | Eye Aspect Ratio (EAR) drops below 0.18 (closed), then rises ≥ 0.22 (open) |
| `'smile'` | User smiles | `mouth_width / mouth_height ≥ 0.72` from landmark geometry |
| `'turn_head'` | Head turns then returns to center | `\|yaw\| > 15°` (turned), then `\|yaw\| < 5°` (returned) |

---

## Tuning Guide

### Security vs. False-Rejection Rate

Higher security = stricter thresholds = more false rejects. Find your balance:

- **Raise** `fineThreshold` (→ 0.80+) + **raise** `spoofReject` (→ 0.4) for high-security deployments (banking, access control).
- **Lower** `fineThreshold` (→ 0.65) + **lower** `minFaceArea` (→ 0.25) for high-throughput, controlled lighting (turnstile, office).
- Keep `coarseThreshold` at least 0.05–0.10 below `fineThreshold` to avoid filtering real matches in the pre-pass.

---

## Example Config Snippets

### Minimal — fully offline, no server

```typescript
await FaceAuth.init({
  deviceId: 'kiosk-001',
});
```

### With AWS sync

```typescript
await FaceAuth.init({
  awsSyncUrl:  'https://api.example.com',
  deviceToken: 'device-secret-token',
  deviceId:    'field-unit-42',
  syncIntervalMs: 3 * 60 * 1000, // sync every 3 min
});
```

### High-security mode

```typescript
await FaceAuth.init({
  thresholds: {
    fineThreshold:  0.80,
    coarseThreshold: 0.65,
    spoofReject:    0.35,
    detectScore:    0.85,
    minFaceArea:    0.50,
    maxYaw:         20,
    maxPitch:       20,
    maxRoll:        20,
  },
  challenges: ['blink', 'turn_head'], // exclude smile (easier to fake)
});
```

### Fast-auth — controlled indoor environment

```typescript
await FaceAuth.init({
  thresholds: {
    fineThreshold:   0.65,
    coarseThreshold: 0.50,
    minFaceArea:     0.25,
    maxYaw:          40,
    maxPitch:        60,
    livenessFrames:  2,
    probeFrameCount: 2,
  },
  challenges: ['blink'],
});
```
