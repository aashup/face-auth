# `react-native-offline-face-auth` — Reusable RN Package (Android + iOS)

## Context

We are building a **brand-new, reusable React Native package** (publishable npm library, not tied to any
existing repo) that any host app can install to authenticate field personnel via **facial recognition +
offline liveness detection**, running **fully offline** on **standard mid-range phones**. The package ships
a TypeScript JS API + a native module (TFLite inference) + bundled open-source models, and behaves
identically on Android and iOS. A bundled **example app** demonstrates and tests it.

Primary use case: **field attendance / personnel check-in** — so anti-spoofing must defeat photo/screen fraud.

Confirmed requirements:
- **Liveness (hybrid):** passive silent anti-spoof on every frame **+ a basic active challenge** —
  the user is randomly prompted to **blink, smile, or turn their head slightly** to prevent attendance
  fraud via photographs or screens. Both run fully offline.
- **Enrollment:** templates are generated **centrally on the server** and synced to the device while online;
  **matching happens 100% offline** in the field.
- **Sync & purge:** authentication/attendance events are stored locally while offline, then **synced to an
  AWS server when connectivity is restored, after which the local copy is purged.**
- **Security bar:** **high assurance** — embeddings only (never raw images at rest), hardware-backed key
  storage, hardware attestation, template cancelability, signed model-integrity checks, and a tamper-evident
  audit log.

The hard part is not "detect a face" — it's doing accurate 1:1 (or 1:N) matching **plus** spoof rejection
**plus** template protection, all on-device, offline, on cheap hardware, with one shared codebase.

---

## Architecture Overview

Authentication pipeline (all on-device, offline):

```
Camera frame (VisionCamera)
   │
   ▼
[1] Face detect + landmarks  ── ML Kit face detector (frame processor)
   │   (bbox, eyes/nose/mouth landmarks, head angle, face size/centering gate)
   ▼
[2] Quality gate            ── reject blur / off-angle / too-small / multi-face / too-dark
   │
   ▼
[3] Passive liveness        ── Silent-Face-Anti-Spoofing (MiniFASNet, MobileNetV2-class) TFLite
   │   spoof score ≥ threshold → REJECT (screen/print/replay)
   ▼
[4] Active challenge        ── random prompt: BLINK / SMILE / TURN-HEAD
   │   verified from FaceMesh landmarks: Eye-Aspect-Ratio dip→rise (blink),
   │   mouth-aspect / lip-corner spread (smile), head yaw from landmarks (turn).
   │   timeout / wrong response → REJECT
   ▼
[5] Face embedding          ── MobileFaceNet TFLite → 128/512-d vector (CPU/XNNPACK)
   │
   ▼
[6] Match                   ── cosine similarity vs locally-stored enrolled templates
   │   ≥ threshold → AUTHENTICATED
   ▼
[7] Record attendance       ── signed local event (hash-chained); queued for AWS sync, then purged
```

Templates are **provisioned by the server** (generated from enrollment photos server-side using the *same*
embedding model), pushed to the device during an online sync, decrypted into memory only at match time.

---

## Technology Choices (grounded in current RN tooling)

**All technologies below are open-source (MIT / Apache-2.0) with no per-seat or runtime license fee.**
Proprietary SDKs (Google ML Kit, Banuba, MiniAI, Face++) are deliberately **avoided** — we use only
open models + open RN libraries so the full prototype source can be shared without licensing encumbrance.

| Concern | Library / Model | License | Why |
|---|---|---|---|
| Camera + frame processors | **react-native-vision-camera** + **react-native-worklets-core** | MIT | Real-time frame processors, runs JS worklets on frames, Android+iOS. |
| TFLite inference | **react-native-fast-tflite** | MIT (runtime: TFLite Apache-2.0) | JSI/zero-copy; **defaults to XNNPACK CPU** (no GPU needed); GPU delegate is optional acceleration. |
| Face detection | **BlazeFace** `.tflite` (from MediaPipe) | Apache-2.0 | Tiny (~0.5 MB) face detector → bbox; runs fast on CPU. |
| Landmarks (gate + challenge) | **MediaPipe Face Mesh / Face Landmark** `.tflite` | Apache-2.0 | 468 landmarks → eye/mouth points + head pose; no proprietary SDK. |
| Passive liveness model | **Silent-Face-Anti-Spoofing (MiniFASNet)** `.tflite` | Apache-2.0 | Tiny (<2 MB), RGB single-frame, ~98% PAD, mobile-grade. |
| Embedding model | **MobileFaceNet** `.tflite` (open weights) | Apache/MIT | ~1–4 MB, mobile-designed, 128/512-d embeddings, strong accuracy on CPU. |
| Active challenge signals | Derived from **Face Mesh landmarks** | — | Eye-Aspect-Ratio (blink), mouth/lip-corner geometry (smile), landmark-derived yaw (turn) — no extra model. |
| Local secure store | **react-native-mmkv** (with encryption key) | MIT | Fast key-value store, AES-encrypted at rest; holds templates + queued attendance events. |
| Network status | **@react-native-community/netinfo** | MIT | Detect connectivity restore to trigger AWS sync. |
| AWS sync | **axios** → AWS API Gateway / app endpoint | MIT | Push queued attendance events; purge on confirmed receipt. (Avoid heavy aws-sdk; plain HTTPS + token.) |
| Key management | **react-native-keychain** → Android **Keystore** / iOS **Keychain** | MIT | Hardware-backed AES key (StrongBox/Secure Enclave when present; falls back gracefully on 3GB devices). |
| Hardware attestation | **Play Integrity API** (Android) + **DeviceCheck/App Attest** (iOS) | Free platform API | Device/app integrity, verified server-side during provisioning sync (online only). |

> **One embedding model used everywhere** (server enrollment + device matching) — vectors are only
> comparable if produced by the identical model + preprocessing.
>
> **Deliverable:** the full source of the **RN package + example app + bundled open `.tflite` models**
> (with provenance/licenses) shared as a repository; no closed binaries, publishable to npm.

---

## Package Structure (scaffolded with `create-react-native-library`)

A library, not an app. Built with `create-react-native-library` (TurboModule template) so it ships a
JS/TS API, a native module, and an `example/` app for dev + manual testing.

```
react-native-offline-face-auth/
├── package.json                      # name, peerDeps, files whitelist (lib/, android/, ios/, models/)
├── react-native-offline-face-auth.podspec
├── src/                              # ── public + internal TS, compiled to lib/ ──
│   ├── index.ts                      # PUBLIC API barrel (the only supported entry point)
│   ├── FaceAuthView.tsx              # <FaceAuthView> camera component (VisionCamera-based)
│   ├── useFaceAuth.ts                # hook: start/cancel auth, status, result callbacks
│   ├── pipeline.ts                   # orchestrates detect→gate→liveness→challenge→embed→match
│   ├── detector.ts                   # BlazeFace + FaceMesh TFLite (bbox + landmarks)
│   ├── qualityGate.ts                # blur/angle/size/lighting/multi-face checks
│   ├── liveness.ts                   # MiniFASNet TFLite (passive) inference + threshold
│   ├── activeChallenge.ts            # blink/smile/head-turn state machine (landmark-driven)
│   ├── embeddings.ts                 # MobileFaceNet preprocessing + inference
│   ├── matcher.ts                    # cosine similarity, threshold, 1:1 / 1:N
│   ├── secure/
│   │   ├── keystore.ts               # hardware-backed key (drives MMKV encryption key)
│   │   ├── templateStore.ts          # encrypted MMKV CRUD for enrolled templates
│   │   ├── cancelable.ts             # biohashing / revocable template transform
│   │   └── auditLog.ts               # append-only signed attendance/auth events
│   ├── sync/
│   │   ├── provisioning.ts           # online: attest device, pull encrypted templates
│   │   ├── attendanceSync.ts         # push queued events to AWS, then purge local
│   │   ├── netStatus.ts              # NetInfo watcher → triggers sync on reconnect
│   │   └── modelIntegrity.ts         # verify .tflite sha256 vs signed manifest on boot
│   ├── config.ts                     # FaceAuthConfig type + sane defaults (thresholds, endpoints)
│   └── types.ts                      # public TS types (AuthResult, Challenge, SyncStatus, …)
├── android/                          # native module: TFLite frame-processor plugin, keystore
├── ios/                              # native module: TFLite frame-processor plugin, keychain
├── models/                           # bundled .tflite files + sha256 manifest + LICENSES
└── example/                          # demo app: enroll-status, run auth, attendance log, sync
```

Internal modules are **not** exported; only `src/index.ts` is the public surface.

---

## Public API Surface (what host apps consume)

```ts
// 1. Initialize the SDK once (loads models, opens encrypted store, starts net watcher)
await FaceAuth.init(config: FaceAuthConfig);
//    config: { awsSyncUrl, deviceToken, thresholds?, challenges?, syncIntervalMs?, purgeAfterSync? }

// 2. Provision templates from server while online (idempotent, resumable)
await FaceAuth.provision();        // attests device, pulls encrypted templates → local store

// 3. Drop-in UI component to run an authentication
<FaceAuthView
  mode="identify" | "verify"
  personnelId={id}                  // required for verify (1:1)
  onResult={(r: AuthResult) => …}   // { ok, personnelId?, matchScore, livenessScore, challengePassed }
  onGuidance={(g) => …}             // "center face", "blink now", etc.
/>

// 3b. Or headless via hook for custom UI
const { start, cancel, status } = useFaceAuth({ mode, personnelId, onResult });

// 4. Attendance + sync (also runs automatically on reconnect)
FaceAuth.getPendingCount();
await FaceAuth.syncNow();           // push queued events to AWS, purge acked rows
FaceAuth.onSyncStatus(cb);          // subscribe to SyncStatus changes

// 5. Lifecycle
await FaceAuth.dispose();
```

Peer dependencies (host installs these): `react-native-vision-camera`, `react-native-worklets-core`,
`react-native-fast-tflite`, `react-native-mmkv`, `react-native-keychain`,
`@react-native-community/netinfo`. The package declares them as `peerDependencies` so the host controls
versions and there's no duplication.

---

## Implementation Plan

### Phase 0 — Scaffold package + example app
- `npx create-react-native-library@latest react-native-offline-face-auth` (TurboModule + view template).
  Produces the `src/` + `android/` + `ios/` + `example/` layout above.
- Declare the open-source RN libs as **peerDependencies**; install them inside `example/` for dev.
- Set min targets in the podspec/gradle: Android `minSdkVersion = 26` (8.0), iOS deployment target `12.0`.
- Document required host config (camera permission strings, VisionCamera frame-processor setup) in README.
- fast-tflite: **default to XNNPACK CPU delegate** (works on every 3GB device, no GPU required);
  detect and use GPU delegate only as optional acceleration where available.

### Phase 1 — Camera + face detection + quality gate (open-source, no ML Kit)
- `detector.ts`: load **BlazeFace** + **Face Mesh** `.tflite` via fast-tflite; in the VisionCamera
  frame processor (throttled ~5–10 fps) run BlazeFace → bbox, then Face Mesh → 468 landmarks.
- Compute head pose (yaw/pitch/roll) and feature points from landmarks (no proprietary detector).
- `qualityGate.ts`: accept a frame only if exactly **one** face, bbox covers target % of frame,
  head pose within bounds, brightness in range, sharpness above threshold. Drives on-screen
  guidance ("center your face", "move closer").

### Phase 2 — Embedding model on-device
- Add MobileFaceNet `.tflite` to `src/models/`; load via fast-tflite (XNNPACK CPU).
- `embeddings.ts`: crop to face bbox, align (use eye landmarks), resize to model input (e.g. 112×112),
  normalize **exactly** as the server pipeline does, run inference → L2-normalized vector.
- Validate parity: same face image through server pipeline and device pipeline must yield
  cosine similarity ≈ 1.0. **This parity check is the make-or-break step.**

### Phase 3 — Passive liveness
- Add MiniFASNet `.tflite`; `liveness.ts` runs it on the (optionally multi-scale cropped) face region.
- Reject if spoof probability ≥ tuned threshold. Run **before** matching to fail fast on spoofs.
- Tune thresholds against a test set of print/screen/replay attacks; expose as remote-config-style
  constants provisioned during sync (so they can be tightened without an app update).

### Phase 3b — Active challenge (anti-fraud)
- `activeChallenge.ts`: a small state machine that, on each auth, picks a **random** challenge
  (BLINK / SMILE / TURN-HEAD) and a randomized order if multiple are required.
- Verification is computed from **Face Mesh landmarks** (no extra model, fully offline):
  - **Blink:** Eye-Aspect-Ratio (EAR) from eye landmarks goes high → low → high in a time window.
  - **Smile:** mouth-aspect / lip-corner spread ratio rises above threshold.
  - **Turn head:** landmark-derived yaw crosses a target (e.g. > +15° then back to center).
- Enforce a timeout and require the live face to remain the *same* tracked face throughout
  (prevents swapping a photo mid-challenge). Wrong/late response → REJECT.
- Defense-in-depth: passive model rejects static screens/prints instantly; the active challenge
  defeats high-quality replay and is the visible anti-fraud step personnel perform.

### Phase 4 — Matching
- `matcher.ts`: cosine similarity of probe embedding vs each enrolled template; authenticate on best
  match ≥ threshold (1:N) or against the claimed identity (1:1). Calibrate threshold (e.g. ~0.6–0.7 for
  MobileFaceNet) for target FAR/FRR using a validation set.
- Optional: average several gated frames' embeddings to reduce variance.

### Phase 5 — High-assurance security
- `keystore.ts`: create/fetch a **hardware-backed** AES-256 key (Android Keystore `StrongBox` if available
  / iOS Secure Enclave). Key never exported.
- `templateStore.ts`: store **embeddings only** (no raw face images, ever) in an **encrypted MMKV**
  instance whose encryption key comes from the hardware-backed key; load into memory only during a
  match, zeroize after.
- `cancelable.ts`: apply a revocable transform (biohashing / random-projection keyed per-tenant) so a
  stolen template is useless elsewhere and can be reissued by rotating the key.
- `auditLog.ts`: append-only, hash-chained, signed local log of every auth attempt (timestamp, result,
  liveness score, match score, device id) — uploaded opportunistically on next sync.
- `modelIntegrity.ts`: on every boot, verify each `.tflite` file's SHA-256 against a **server-signed
  manifest**; refuse to run if tampered.

### Phase 6 — Server provisioning sync (the only online part)
- `provisioning.ts`: when online, (a) run **Play Integrity / App Attest** and submit the attestation token
  to the server, (b) on success, pull this device's roster of **encrypted enrolled templates** + threshold
  config + signed model manifest, (c) store locally encrypted.
- Server generates templates from enrollment photos using the **identical** embedding model & preprocessing.
- Sync is incremental and resumable; after first successful sync the device authenticates indefinitely
  offline. Define template TTL / forced re-sync window for revocation hygiene.

### Phase 6b — Attendance sync to AWS + local purge
- Every successful (or failed) auth writes an **attendance event** to the encrypted local queue:
  personnel id, timestamp, geo (optional), liveness score, match score, device id, challenge passed —
  **embeddings/images are never uploaded**, only the decision record. Each event is hash-chained + signed.
- `netStatus.ts` watches connectivity via NetInfo. On reconnect (or periodic timer), `attendanceSync.ts`:
  1. Batches queued events and **POSTs to the AWS endpoint** (API Gateway → Lambda/app, TLS, device
     auth token from provisioning).
  2. On a **confirmed 2xx + server ack** for a batch, **purges** those local rows (delete from the
     encrypted DB). Unconfirmed events stay queued and retry with backoff (idempotency key per event
     so retries don't double-count attendance).
  3. Sync is transactional per-batch: purge only what the server acknowledged.
- Result: device works fully offline; the moment it has network, attendance flushes to AWS and local
  data is cleared, minimizing data-at-rest on field devices.

### Phase 7 — Public API, example app, docs
- Finalize `src/index.ts` API (`init`, `provision`, `FaceAuthView`, `useFaceAuth`, `syncNow`,
  `getPendingCount`, `onSyncStatus`, `dispose`) and the `FaceAuthConfig` / result types.
- `<FaceAuthView>` exposes live guidance overlay, success/failure states, retry, and a configurable
  **fallback** signal (host renders PIN/passcode) when auth fails N times or no template is provisioned.
- Build the **example app**: enroll-status, run identify/verify auth, view local attendance queue,
  trigger sync, and a diagnostics panel (model versions, last sync, attestation status, FAR/FRR harness).
- Write README: install, peerDeps, host config, API reference, model provenance + licenses.

---

# Follow-up Plan — Device Integration (Phases 8 & 9)

## Context

Phases 0–7 are implemented and verified in TypeScript: the full offline pipeline (detect → gate →
liveness → challenge → embed → match), security layer, provisioning, and AWS sync/purge all exist and
the core math is unit-verified. **What is missing is device integration** — the package cannot yet
authenticate a real face on a phone because (1) nothing converts live camera frames into the `RgbFrame`
the pipeline consumes, and (2) `example/` is only an `App.tsx`, not a runnable RN project, and there is
no automated test suite. This follow-up covers exactly those two gaps.

**Chosen approach for (a):** JS-thread inference. A Vision Camera frame-processor worklet uses
`vision-camera-resize-plugin` to extract a downscaled RGB buffer, marshals it to the JS thread, where the
**existing** `detector.ts` / `pipeline.ts` run the TFLite models via `react-native-fast-tflite`. This adds
**zero new native code**, reuses everything already built, and keeps session state on the JS thread where
`useFaceAuth` already lives. Frames are throttled to ~5–10 fps to fit the 3 GB CPU budget.

---

## Phase 8 — (a) Live camera → RgbFrame frame pipeline (no native code)

**Goal:** replace the placeholder `global.__faceAuthExtract` seam in
[`src/FaceAuthView.tsx`](D:\erp\pos\face-recognition\src\FaceAuthView.tsx) with a real worklet that feeds
the pipeline.

- **Add peer dep:** `vision-camera-resize-plugin` (MIT, mrousavy) in
  [`package.json`](D:\erp\pos\face-recognition\package.json) `peerDependencies`; document in README.
- **New file `src/frameSource.ts`** — a `useFrameSource(onRgbFrame)` hook that:
  - builds the resize-plugin (`useResizePlugin`) and a throttle (track `lastRun` timestamp in the worklet),
  - in the `useFrameProcessor` worklet, calls `resize(frame, { scale: { width: W, height: H }, pixelFormat: 'rgb', dataType: 'uint8' })` to get a `Uint8Array`,
  - marshals `{ data, width, height }` to JS via `useRunOnJS` (worklets-core), invoking `onRgbFrame(rgb)`.
  - Pick W×H so the largest model input (FaceMesh 192) is satisfied without over-sampling (e.g. 256×256);
    the existing `resampleToSquare` in
    [`src/detector.ts`](D:\erp\pos\face-recognition\src\detector.ts) handles per-model crops from there.
- **Refactor `FaceAuthView.tsx`:**
  - drop the `__faceAuthExtract` global and the `FrameExtract` type,
  - on each JS-thread `RgbFrame`, run `detectFace(frame, { blazeface, faceMesh }, thresholds.detectScore)`
    from `detector.ts` to get `DetectedFace[]`, then call the hook's `feed(frame, faces)`.
  - This means `FaceAuthView` now needs the **detector** models too — extend `PipelineModels` consumption:
    pass `{ blazeface, faceMesh, embedding, liveness }`. Update
    [`src/pipeline.ts`](D:\erp\pos\face-recognition\src\pipeline.ts) `PipelineModels` to carry the detector
    models (or add a separate `DetectorModels` prop on the view) and thread them through.
- **`useFaceAuth.feed` already exists** and runs the session on the JS thread — reuse unchanged.
- **Throttling + back-pressure:** skip new frames while a `feed` is mid-flight (a `processing` ref) so a
  slow embedding pass on a 3 GB device can't queue up frames.
- **Model loading stays in the host** (example) via `useTensorflowModel` — already shown in
  [`example/App.tsx`](D:\erp\pos\face-recognition\example\App.tsx); update it to also load BlazeFace +
  FaceMesh into the detector models and pass them to the view.

**Files:** `src/frameSource.ts` (new), `src/FaceAuthView.tsx`, `src/pipeline.ts`, `src/index.ts`
(export `DetectorModels` type), `package.json`, `example/App.tsx`, `README.md`.

> Note: this is JS-thread inference by design. If profiling on a 3 GB device misses the latency targets,
> the documented future optimization is to move `model.runSync` + decode into the worklet — but that is
> explicitly out of scope here.

## Phase 9 — (b) Runnable example app + Jest test suite

### 9a. Make `example/` a real RN project
Follow the standard `react-native-builder-bob` / `create-react-native-library` example wiring so the
example consumes the package source directly (hot-reloads on edits):

- Scaffold a bare RN app in `example/` (`npx @react-native-community/cli init` then keep `App.tsx`), or
  add the missing files by hand: `example/package.json`, `example/index.js`, `example/app.json`,
  `example/metro.config.js`, `example/babel.config.js`, `example/react-native.config.js`, plus the
  generated `example/android/` and `example/ios/` projects.
- **Link the package to source:** `metro.config.js` `watchFolders` includes the repo root + `extraNodeModules`
  maps `react-native-offline-face-auth` → `../src`; `babel.config.js` uses `babel-plugin-module-resolver`
  with the same alias (the bob convention). This lets the example import the live TS without a build.
- Install the peer deps in `example/`: vision-camera, worklets-core, fast-tflite, vision-camera-resize-plugin,
  mmkv, keychain, netinfo (+ `react-native-get-random-values` to back the CSPRNG used in
  [`src/secure/keystore.ts`](D:\erp\pos\face-recognition\src\secure\keystore.ts)).
- Add the iOS `NSCameraUsageDescription` + Android `CAMERA` permission and the VisionCamera/worklets babel
  plugin to the example's native config.
- Drop placeholder/real `.tflite` files into
  [`models/`](D:\erp\pos\face-recognition\models) and fill the SHA-256 fields in
  [`models/manifest.json`](D:\erp\pos\face-recognition\models\manifest.json) so the example boots (the
  integrity check currently fails on empty digests by design).

### 9b. Jest test suite (the part runnable here, no device)
Add `jest` + `babel-jest` (or `ts-jest`) + `@types/jest` to root `devDependencies`, a `jest.config.js`,
and a `jest.setup.js` that mocks the native peer modules. Tests cover the **pure logic** I built:

- `__tests__/sha256.test.ts` — empty / "abc" / 56-char / HMAC known vectors (already validated manually).
- `__tests__/cancelable.test.ts` — transform preserves cosine geometry (Δ<0.05) and cross-key isolation.
- `__tests__/matcher.test.ts` — `identify` picks the right id; `verify` accepts true / rejects false.
- `__tests__/qualityGate.test.ts` — no_face / multiple / too-small / off-center / pose / lighting branches.
- `__tests__/activeChallenge.test.ts` — blink (EAR dip→rise), smile, turn pass; timeout + lost-tracking fail.
- `__tests__/pipeline.test.ts` — feed synthetic frames/faces through `FaceAuthSession`; assert stage
  transitions and `AuthResult` for happy path, liveness-fail, challenge-fail, no-template, no-match.
- `__tests__/auditLog.test.ts` — append → chain links → `verifyChain` true; tamper → false; `purge` drops
  acked ids. (Mock `react-native-mmkv` with an in-memory Map and `react-native-keychain`.)
- `__tests__/attendanceSync.test.ts` — mock axios + auditLog: only acked ids purged; partial-ack leaves
  remainder; re-entrancy guard. (`syncAttendance` in
  [`src/sync/attendanceSync.ts`](D:\erp\pos\face-recognition\src\sync\attendanceSync.ts)).

Wire `npm test` (already declared) and add `npm run typecheck` against the full graph once example peer
deps resolve the React-layer files.

**Files:** `jest.config.js`, `jest.setup.js`, `__mocks__/` (mmkv, keychain, netinfo), `src/__tests__/*`,
`package.json` (devDeps), example project files listed above.

### Verification
- **Tests (here, no device):** `npm install` then `npm test` — all suites green; `npm run typecheck` clean.
- **Frame pipeline (device):** run `example/` on an Android 8 / 3 GB device and an iPhone; confirm the
  guidance overlay reacts (centering/lighting), the active challenge prompts, and a provisioned face
  authenticates while in airplane mode.
- **Sync/purge (device):** queue events offline, re-enable network, confirm `POST /attendance`, server ack,
  and local purge; kill network mid-batch and confirm no double-count.
- **Perf:** log per-stage timings on the 3 GB device against the targets below.

---

## Minimum Device Support
- **Android 8.0+ (API 26)**, **iOS 12+**, **≥ 3 GB RAM** — **no GPU required**.
- Inference path defaults to **XNNPACK CPU**; multi-threaded, sized for these devices.
- Keep all model inputs small (112×112 embed, BlazeFace 128×128) to fit RAM/compute budget.
- Hardware-backed keystore: use StrongBox/Secure Enclave **when present**, fall back to standard
  Keystore/Keychain on older 3GB devices (still hardware-backed on Android 8+).

## Performance Targets (CPU-only, mid-range / 3GB)
- Face detect (BlazeFace): <25 ms/frame, throttled.
- Liveness inference (MiniFASNet): <40 ms.
- Embedding inference (MobileFaceNet, XNNPACK): <80 ms.
- Active challenge: ~1.5–3 s (one prompt, user-paced).
- End-to-end decision: **<4 s** including the active challenge + a few gated frames.
- Total model footprint: **<10 MB** (BlazeFace + Face Mesh + MiniFASNet + MobileFaceNet combined).

---

## Key Risks & Mitigations
1. **Server/device embedding mismatch** → identical model + byte-identical preprocessing; automated parity
   test in CI (Phase 2). Highest-priority risk.
2. **Low-end / no-GPU performance** → XNNPACK CPU is the default path; small inputs, throttled frames,
   quantized (int8) `.tflite` variants if needed to hit latency on 3GB devices.
3. **Liveness false rejects in poor field lighting** → quality gate enforces lighting; thresholds tunable
   via provisioning, not hardcoded.
4. **Model theft / tampering** → signed SHA-256 manifest + integrity check at boot; consider model
   obfuscation/encryption-at-rest of the `.tflite` if threat model demands.
5. **iOS Secure Enclave vs Android Keystore API differences** → thin `keystore.ts` abstraction over
   react-native-keychain; test StrongBox/Enclave presence per device.

---

## Verification Plan
- **Parity test (CI):** server vs device embedding cosine ≈ 1.0 for the same image set.
- **Accuracy:** measure FAR/FRR on a labeled validation set; tune match threshold.
- **Liveness PAD:** run a print/screen/replay attack set; report spoof-acceptance rate (target ~0%).
- **Active challenge:** verify blink/smile/turn are reliably detected and that a static photo or a
  pre-recorded video held to the camera **fails** the challenge.
- **Offline test:** airplane mode after one sync — full enroll-roster auth must work; attendance events
  accumulate locally.
- **Sync & purge:** re-enable network — confirm queued attendance events POST to AWS, server acks, and
  local rows are purged; killing network mid-batch must retry without double-counting (idempotency).
- **Min-spec device:** validate the full flow on an Android 8.0 / 3GB device and an iOS 12 device,
  CPU-only, meeting the latency targets above.
- **Licensing:** confirm every dependency + model is MIT/Apache (or platform-free API) and that the
  package can be published with no additional license obligations.
- **Package integration:** in a clean host app, `npm install` the package + peerDeps, call the public API,
  and confirm a full enroll→auth→sync flow works end-to-end (the example app is the reference host).
- **Cross-platform:** identical thresholds/results on a mid-range Android (e.g. 2023 ~$200 device) and a
  base iPhone SE-class device.
- **Security:** confirm no raw images at rest, key is hardware-backed (not extractable), audit log is
  hash-chained, integrity check rejects a modified `.tflite`.
- **Perf:** measure per-stage latency on target hardware against the targets above.
```