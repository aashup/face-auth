# Architecture & Pipeline Logic

Deep-dive into how `react-native-offline-face-auth` works — from raw camera frame to signed attendance event.

---

## Pipeline Overview

```
Camera (30 fps)
    │
    ▼
frameSource.ts ─── worklet · 256×256 RGB · back-pressure drop
    │
    ▼
detector.ts ─────── BlazeFace (bbox) + FaceMesh (468 landmarks + head pose)
    │
    ▼
pipeline.ts  ◄──── FaceAuthSession state machine
    │
    ├─ [GATING] ────── qualityGate.ts ──── 7 checks, fast-fail
    │                      │ fail → guidance hint, loop back
    │                      │ pass ↓
    ├─ [CHALLENGE] ── activeChallenge.ts ── blink / smile / turn-head
    │                      │ fail / timeout → REJECT
    │                      │ pass ↓
    ├─ [ANALYZING] ── liveness.ts ────── MiniFASNet ×2 (passive spoof)
    │              ── embeddings.ts ──── FaceNet 512-d vector
    │                      │ collect N frames, average scores
    │                      │ pass ↓
    └─ [MATCH] ────── matcher.ts ────── 3-step cascade
                           │
                           ▼
                       AuthResult ──── auditLog.ts (signed hash-chained event)
```

---

## Stage 1 — Frame Capture (`src/frameSource.ts`)

Uses `react-native-vision-camera` + `vision-camera-resize-plugin` inside a VisionCamera **worklet** (runs on a dedicated native thread, never blocks JS).

- Downscales each frame to **256×256 RGB** before passing to JS.
- Back-pressure via a shared `isBusy` Reanimated shared value: if the previous frame is still processing the new one is **dropped**, not queued. Keeps latency low on 3 GB devices.
- Marshals the RGB buffer to the JS thread with `useRunOnJS`.
- Effective throughput: ~5–10 accepted frames per second.

---

## Stage 2 — Detection (`src/detector.ts`)

Two TFLite models run sequentially on each accepted frame:

| Model | Input size | Output |
|-------|-----------|--------|
| **BlazeFace** | 128×128 | Face bounding box + detection confidence |
| **FaceMesh** | 192×192 | 468 landmark points (normalized 0..1) + head pose |

Head pose (yaw, pitch, roll in degrees) is derived from landmark geometry via a simplified PnP approach — no separate model needed. `LANDMARK` constants map indices to named facial features (eye corners, nose tip, chin, mouth corners, etc.).

FaceMesh only runs during `challenge` and `analyzing` stages; the cheaper BlazeFace runs every frame for continuous face tracking.

---

## Stage 3 — Quality Gating (`src/qualityGate.ts`)

Checks execute in fast-fail order. The first failure returns a user-facing guidance message.

| # | Check | Condition | User hint |
|---|-------|-----------|-----------|
| 1 | `no_face` | No bbox detected | "Position your face in the frame" |
| 2 | `multiple_faces` | More than one bbox | "Only one person at a time" |
| 3 | `face_too_small` | Face area < `minFaceArea` × frame | "Move closer" |
| 4 | `off_center` | Face center outside `centerMargin` zone | "Center your face" |
| 5 | `bad_pose` | `\|yaw\|`>`maxYaw` or `\|pitch\|`>`maxPitch` or `\|roll\|`>`maxRoll` | "Look straight ahead" |
| 6 | `too_dark` / `too_bright` | Mean luma < `minBrightness` or > `maxBrightness` | "Improve lighting" |
| 7 | `blurry` | Laplacian variance < `minSharpness` | "Hold still" |

All thresholds are configurable — see [config.md](config.md).

---

## Stage 4 — Active Challenge (`src/activeChallenge.ts`)

A state machine that enforces one randomized liveness challenge per session.

```
WAITING ──► PROMPTED ──► VERIFYING ──► DONE (pass)
                │
                └── fail: timeout / face tracking lost
```

**Challenge mechanics:**

- **Blink** — Eye Aspect Ratio (EAR) computed from 6 eye landmarks (Soukupová & Čech formula).
  EAR dips below `0.18` (closed) then rises ≥ `0.22` (open) = pass.
- **Smile** — `mouth_width / mouth_height ≥ 0.72` from mouth landmark geometry = pass.
- **Turn head** — `|yaw| > 15°` (turned away), then `|yaw| < 5°` (returned to center) = pass.

Failure triggers: `challengeTimeoutMs` exceeded, or face tracking lost for 2+ consecutive frames.

---

## Stage 5 — Passive Liveness (`src/liveness.ts`)

**MiniFASNet** runs at two spatial scales to defeat different attack classes:

| Scale | Model | Targets |
|-------|-------|---------|
| 2.7× | `spoof_2_7.tflite` | Texture artifacts from printed photos, phone screens |
| 4.0× | `spoof_4_0.tflite` | Geometry anomalies from 3D masks, video replays |

Per scale: face crop → resize to 80×80 → TFLite inference → `[spoof_score, live_score]`.

Final liveness score = average `live_score` across both branches. Scores are averaged across `livenessFrames` frames (default 4) before the pass/fail decision. Rejected when `avg_score < (1 - spoofReject)`.

> **Warning:** Never set `spoofReject` below `0.3` in production — legitimate live faces will fail.

---

## Stage 6 — Embedding (`src/embeddings.ts`)

**MobileFaceNet (FaceNet-512)** produces a 512-dimensional face representation:

1. **Align** — rotate + scale using eye landmark midpoints so both eyes are level at a fixed position.
2. **Crop** — extract 112×112 pixels from the aligned face region.
3. **Normalize** — scale pixel values to `[-1.0, 1.0]`.
4. **TFLite inference** → raw 512-d float vector.
5. **L2 normalize** → unit-length vector (cosine similarity = dot product; no division needed at match time).

`averageEmbeddings()` combines multiple probe frames into a single representative vector.

---

## Stage 7 — Matching (`src/matcher.ts`)

Three-step cascade balances accuracy against CPU cost:

### Step 1 — Coarse Filter (fast 1:N)
`probe[0]` vs the first enrolled template per person. Identities scoring below `coarseThreshold` (default `0.55`) are dropped before the expensive cross-matrix step.

### Step 2 — Cross-Matrix
For each person that survives the coarse filter: build an M×K score matrix (M = probe frames, K = enrolled frames per person). Each cell = cosine similarity of one probe frame vs one template frame.

### Step 3 — Score Fusion
Collapse the M×K matrix into a single confidence value:

| Strategy | How | Best for |
|----------|-----|---------|
| `'mean'` | Arithmetic average of all M×K scores | Simple; sensitive to outliers |
| `'max'` | Single best-matching pair | Diverse enrollment angles |
| `'top_k'` **(default)** | Average of top `topK` scores | Robust against a single blurry/off-angle frame |

Final score ≥ `fineThreshold` (default `0.72`) → **AUTHENTICATED**.

> FaceNet-512 same-person cosine scores typically fall between 0.55–0.90. Setting `fineThreshold` above 0.85 without per-deployment calibration produces significant false-reject rates.

**Exports:** `identify()` (1:N), `verify()` (1:1), `identifyMultiFrame()`, `verifyMultiFrame()`, `groupTemplates()`.

---

## Stage 8 — Audit Log (`src/secure/auditLog.ts`)

Every authentication attempt is appended as a tamper-evident, hash-chained event:

```
Event = {
  id, ts, ok, personnelId?, matchScore, livenessScore,
  challengePassed, failureReason?, deviceId,
  prevHash,   ← SHA-256 of previous event ('GENESIS' for the first)
  hash,       ← SHA-256(prevHash ‖ JSON(event fields))
  sig         ← HMAC(deviceKey, hash)
}
```

- Modifying any field breaks the hash chain for all subsequent events.
- Events are pushed to AWS via `attendanceSync.ts` and purged **only** after server acknowledgement.
- `verifyChain()` recomputes all hashes and signatures to confirm integrity.

---

## Security Layer (`src/secure/`)

### `keystore.ts`
Derives or retrieves a hardware-backed AES-256 key:
- **Android**: Android Keystore (StrongBox if available)
- **iOS**: Keychain Services (Secure Enclave if available)

The key never leaves secure hardware. It seeds both the encrypted KV store and the default `cancelableKey`.

### `secureKv.ts`
Encrypted key-value store over AsyncStorage:
- At-rest: 8-byte random nonce + HMAC-SHA256 keystream cipher (32-byte blocks).
- In-memory Map provides synchronous reads — no `await` on hot paths.

### `templateStore.ts`
CRUD for enrolled face templates:
- Stored as `personnelId|base64(float32le)`.
- `replaceAll()` — full roster replacement (used after provisioning).
- `upsertPersonnel()` — incremental add/update.
- `loadAll()` — fetch all templates for matching.
- Raw face images are **never** stored.

### `cancelable.ts` — Revocable Templates (Biohashing)
Prevents raw biometrics from being usable even if storage is compromised:

1. Derive a deterministic random projection matrix from `cancelableKey`.
2. Project 512-d embedding → 256-d vector.
3. Binarize (>0 → 1, else → 0) → 256-bit biohash.

**Key rotation**: generate a new `cancelableKey` → all existing templates are invalidated → re-enroll from captured embeddings to restore access.

### `modelIntegrity.ts`
On every `FaceAuth.init()` (when `readModelBytes` + `modelManifest` are supplied):
- Reads each `.tflite` file, computes SHA-256, compares against `manifest.json`.
- Throws and refuses to start if any model has been tampered with.

---

## Sync Layer (`src/sync/`)

### `provisioning.ts`
Pulls enrolled templates from the server when online:
- `POST /provision { attestation, since }` → server replies with `full` or `delta` roster.
- **Incremental**: `since = lastProvisionedAt`; server sends only new/changed records.
- Applies the cancelable transform locally before storing → raw embeddings never touch AsyncStorage.
- Caches a server-signed model manifest for the next boot integrity check.

### `attendanceSync.ts`
Pushes queued events to AWS:
- `POST /attendance { events: [...] }` → `{ acked: [ids] }`.
- Local purge **only** for acknowledged IDs (no data loss on partial failure).
- Idempotent: each event has a UUID; server deduplicates.
- Re-entrancy guard prevents concurrent sync runs.

### `netStatus.ts`
Wraps `@react-native-community/netinfo`:
- Triggers `syncAttendance()` on every `false → true` connectivity transition.
- Fallback periodic timer fires every `syncIntervalMs` (default 5 min).

---

## Model Inventory

| Model | File | Size | License | Purpose |
|-------|------|------|---------|---------|
| BlazeFace | `models/blazeface.tflite` | ~0.5 MB | Apache-2.0 | Face detection (bbox + confidence) |
| Face Mesh | `models/face_mesh.tflite` | ~3 MB | Apache-2.0 | 468 landmark points + head pose |
| FaceNet-512 | `models/facenet_512.tflite` | ~4 MB | Apache/MIT | 512-d face embedding |
| MiniFASNet 2.7× | `models/spoof_2_7.tflite` | ~1 MB | Apache-2.0 | Passive liveness (texture scale) |
| MiniFASNet 4.0× | `models/spoof_4_0.tflite` | ~1 MB | Apache-2.0 | Passive liveness (geometry scale) |
| **Total** | | **~9.5 MB** | | |

SHA-256 digests are in `models/manifest.json`.

---

## Performance Targets

Measured on CPU-only (XNNPACK), 3 GB RAM device:

| Stage | Target latency |
|-------|----------------|
| Frame capture + resize | < 10 ms |
| BlazeFace detection | < 25 ms |
| FaceMesh landmarks | < 40 ms |
| Quality gate | < 5 ms |
| MiniFASNet ×2 (liveness) | < 40 ms |
| FaceNet embedding | < 80 ms |
| Active challenge (user-paced) | 1.5–3 s |
| **End-to-end decision** | **< 4 s** |
| Total model footprint | < 10 MB |

---

## Security Properties Summary

| Property | Mechanism |
|----------|-----------|
| No raw images stored | Only 512-d embeddings written to disk |
| Hardware-backed key | Android Keystore / iOS Keychain (StrongBox / Secure Enclave) |
| Encrypted at rest | HMAC-SHA256 keystream cipher per SecureKV entry |
| Template cancelability | Biohashing; rotate `cancelableKey` to revoke all templates |
| Tamper-evident log | SHA-256 hash chain + HMAC signature per audit event |
| Model integrity | SHA-256 of every `.tflite` verified vs signed manifest on boot |
| Passive anti-spoofing | MiniFASNet two-scale; defeats prints, screens, replays |
| Active anti-spoofing | Random blink/smile/turn; defeats static-photo fraud |
| Offline-first | Full pipeline runs with zero network; sync on reconnect |
| Purge on sync | Events deleted only after server acknowledgement |
