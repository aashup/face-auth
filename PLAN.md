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
[1] Face detect + landmarks  ── BlazeFace + Face Mesh TFLite (frame processor)
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

## Technology Choices

All technologies are open-source (MIT / Apache-2.0) with no per-seat or runtime license fee. Proprietary
SDKs (Google ML Kit, Banuba, MiniAI, Face++) are deliberately avoided.

| Concern | Library / Model | License |
|---|---|---|
| Camera + frame processors | react-native-vision-camera + react-native-worklets-core | MIT |
| TFLite inference | react-native-fast-tflite (XNNPACK CPU default) | MIT / Apache-2.0 |
| Face detection | BlazeFace `.tflite` (MediaPipe) | Apache-2.0 |
| Landmarks | MediaPipe Face Mesh `.tflite` | Apache-2.0 |
| Passive liveness | Silent-Face-Anti-Spoofing (MiniFASNet) `.tflite` | Apache-2.0 |
| Embedding | MobileFaceNet `.tflite` | Apache/MIT |
| Local secure store | react-native-mmkv (encrypted) | MIT |
| Network status | @react-native-community/netinfo | MIT |
| AWS sync | axios | MIT |
| Key management | react-native-keychain | MIT |
| Hardware attestation | Play Integrity / DeviceCheck-App Attest | platform |

> One embedding model used everywhere (server enrollment + device matching) — vectors are only
> comparable if produced by the identical model + preprocessing.

---

## Implementation Phases

- **Phase 0** — Scaffold package + example app, peerDeps, min targets (Android 8.0 / iOS 12), README.
- **Phase 1** — Camera + BlazeFace/FaceMesh detection + quality gate.
- **Phase 2** — MobileFaceNet embedding + server/device parity test.
- **Phase 3** — Passive liveness (MiniFASNet).
- **Phase 3b** — Active challenge (blink/smile/turn from landmarks).
- **Phase 4** — Matching (cosine similarity, FAR/FRR calibration).
- **Phase 5** — High-assurance security (hardware key, encrypted MMKV, cancelable templates, audit log,
  model integrity).
- **Phase 6** — Server provisioning sync (attestation + encrypted template pull).
- **Phase 6b** — Attendance sync to AWS + local purge (NetInfo-triggered, idempotent, transactional).
- **Phase 7** — Public API, example app, docs.

---

## Minimum Device Support
- Android 8.0+ (API 26), iOS 12+, ≥ 3 GB RAM — no GPU required.
- Inference defaults to XNNPACK CPU; small model inputs (112×112 embed, 128×128 BlazeFace).

## Performance Targets (CPU-only, mid-range / 3GB)
- Face detect <25 ms/frame; liveness <40 ms; embedding <80 ms.
- End-to-end decision <4 s incl. active challenge. Model footprint <10 MB total.

See the canonical plan at the repo root planning notes for full risk/verification detail.
