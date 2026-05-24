# CLAUDE.md — Contributor & AI Guide

> **At the start of every session, read the persistent developer memory file before doing anything else:**
> `C:\Users\xtrox\.claude\projects\D--erp-face-recognition\memory\developer.md`
> It contains the authoritative architecture reference, confirmed model class orderings, fixed bugs, security invariants, and test coverage notes accumulated across sessions.

This file helps Claude (and human contributors) work effectively on `react-native-offline-face-auth` — an offline-first React Native package for facial recognition and attendance tracking.

---

## Project in Two Sentences

`react-native-offline-face-auth` provides a drop-in React Native camera component that authenticates field personnel entirely on-device using TFLite models (BlazeFace, FaceMesh, FaceNet, MiniFASNet). It combines passive anti-spoofing with a randomized active challenge, stores templates and attendance events in encrypted local storage, and syncs to AWS when connectivity is available.

---

## Directory Map

```
src/
├── index.ts              ← PUBLIC API barrel — only edit stable symbols here
├── types.ts              ← Shared TypeScript types (AuthResult, Challenge, etc.)
├── config.ts             ← FaceAuthConfig interface + DEFAULT_THRESHOLDS
├── runtime.ts            ← Module-level config singleton (setConfig / getConfig)
├── faceAuth.ts           ← FaceAuth static class (init / provision / sync / enroll)
├── FaceAuthView.tsx      ← Camera component (renders pipeline + UI guidance)
├── FaceAuthModal.tsx     ← Modal wrapper around FaceAuthView
├── FaceMaskOverlay.tsx   ← SVG oval face guide overlay
├── useFaceAuth.ts        ← Headless hook (start / feed / cancel) for custom UIs
├── frameSource.ts        ← Worklet: camera frame → 256×256 RGB bytes
├── detector.ts           ← BlazeFace + FaceMesh TFLite inference
├── qualityGate.ts        ← 7 frame quality checks, fast-fail
├── liveness.ts           ← MiniFASNet passive anti-spoof (two scales)
├── activeChallenge.ts    ← Blink / smile / turn-head state machine
├── embeddings.ts         ← FaceNet 512-d embedding extraction
├── matcher.ts            ← 3-step cosine similarity cascade
├── pipeline.ts           ← FaceAuthSession state machine (gating→challenge→analyzing→match)
├── logger.ts             ← Debug logging helpers
├── secure/
│   ├── sha256.ts         ← Pure-JS HMAC-SHA256 (no native deps)
│   ├── keystore.ts       ← Hardware-backed AES-256 key (Android Keystore / iOS Keychain)
│   ├── secureKv.ts       ← Encrypted AsyncStorage KV store
│   ├── templateStore.ts  ← Enrolled face template CRUD
│   ├── cancelable.ts     ← Biohashing / revocable template transform
│   ├── auditLog.ts       ← Hash-chained, HMAC-signed attendance log
│   └── modelIntegrity.ts ← TFLite SHA-256 boot-time verification
└── sync/
    ├── provisioning.ts   ← Pull templates from server (incremental)
    ├── attendanceSync.ts ← Push events to AWS + purge acked rows
    ├── netStatus.ts      ← Network watcher + periodic sync trigger
    ├── http.ts           ← Axios wrapper with device auth header
    └── modelIntegrity.ts ← Integrity check helper used by provisioning

models/               ← Bundled .tflite files + manifest.json
example/              ← Runnable React Native demo app
server/index.js       ← Mock Node.js backend (port 8080)
__mocks__/            ← Jest mocks for native modules
src/__tests__/        ← Jest unit tests
```

---

## Public API Entry Point

**`src/index.ts`** is the only stable surface. When adding new exports:
- Add to `src/index.ts` only when the symbol is production-ready.
- Internal modules (`detector`, `qualityGate`, `secure/*`, `sync/*`) may change without notice.

---

## Key Patterns — Follow These

### Thresholds always come from config
All numeric decision values live in `src/config.ts` `DEFAULT_THRESHOLDS`. Never hard-code a threshold value in pipeline code — always read from the resolved config via `getConfig()` or from the `Thresholds` object passed to the function.

### Worklet-safe frame processing
`src/frameSource.ts` runs inside a VisionCamera worklet on a native thread. Code that runs in a worklet must not close over JS objects, React state, or non-serializable values. Pass data via Reanimated shared values or `useRunOnJS`.

### Security layer invariants (never break)
- Raw face images must **never** be persisted to disk. Only 512-d float embeddings are stored.
- `templateStore.upsertPersonnel()` must **always** receive embeddings that have already had the cancelable transform applied (see `cancelable.ts`). Never store raw embeddings.
- `auditLog.ts` is append-only. Never delete or modify events before the server has acknowledged them.
- `FaceAuth.init()` runs model integrity checks when `readModelBytes` + `modelManifest` are supplied. Do not bypass or skip this check.

### Adding a new public symbol
1. Implement in the appropriate internal module.
2. Export from `src/index.ts`.
3. Add the corresponding type to `src/types.ts` if it is a shared type.

---

## Commands

```bash
npm test              # run Jest unit tests
npm run typecheck     # TypeScript strict-mode check
npm run lint          # ESLint
npm run prepare       # build to lib/ (commonjs + esm + typings)

node server/index.js  # start mock backend on http://localhost:8080
# Android ADB tunnel:
# adb reverse tcp:8080 tcp:8080
```

Tests run without a device — native modules are mocked in `__mocks__/`.

---

## Testing Approach

- **Unit tests** in `src/__tests__/` — one file per module.
- **Mocks** in `__mocks__/` for `async-storage`, `netinfo`, `react-native-keychain`.
- Test threshold behaviour by passing `Partial<Thresholds>` overrides directly to the functions under test — do not patch `DEFAULT_THRESHOLDS`.
- The mock backend (`server/index.js`) is for manual end-to-end testing with the example app, not for Jest tests.

---

## TypeScript

- Strict mode enabled in `tsconfig.json`.
- Run `npm run typecheck` before committing.
- Separate build config: `tsconfig.build.json` (used by `bob build`), `tsconfig.typecheck.json` (CI typecheck).
- Peer-dep type stubs live in `types/peer-stubs.d.ts`.

---

## Important Invariants

| Invariant | Where enforced |
|-----------|---------------|
| No raw images on disk | `templateStore.ts`, `embeddings.ts` |
| Cancelable transform before storage | `faceAuth.enrollLocal()`, `provisioning.ts` |
| Audit log is append-only | `auditLog.ts` — purge only after server ack |
| Model integrity on every boot | `faceAuth.init()` → `modelIntegrity.ts` |
| Coarse threshold < fine threshold | Caller responsibility; document in config |
| Hardware key never exported | `keystore.ts` — key stored in secure element |

---

## Further Reading

- [logic.md](logic.md) — Full pipeline walkthrough, stage by stage
- [config.md](config.md) — Every config field and threshold with defaults and tuning guidance
- [models/README.md](models/README.md) — Model provenance and licenses
- [example/App.tsx](example/App.tsx) — Complete reference host app
