# react-native-offline-face-auth — Example App

Runnable React Native demo that exercises the full `react-native-offline-face-auth` package:
multi-shot local enrollment, offline identification, duplicate-enrollment detection, and optional
AWS attendance sync.

---

## Screenshots

| Home Screen | Enrollment Dialog | Camera / Face Guide |
|:-----------:|:-----------------:|:-------------------:|
| ![Home screen showing main menu with 3 enrolled templates and 1 pending sync event](../screenshots/screenshot_20260524_202829.png) | ![Enrollment dialog prompting for person ID with Start Enroll button](../screenshots/screenshot_20260524_202855.png) | ![Live camera view with SVG oval face guide overlay and quality gate hint](../screenshots/screenshot_20260524_202959.png) |
| Main menu — 3 templates enrolled, 1 pending event | "Enter Person ID / Name" dialog before capture | Live camera + oval guide · *"Position your face in the frame"* |

---

## What the app demonstrates

| Feature | How |
|---------|-----|
| **Local enrollment** | Tap *Enroll* → enter a name → camera opens `ENROLL_SHOTS` times (default 3), capturing one face embedding per session |
| **Duplicate detection** | After the first shot, `FaceAuth.checkDuplicate()` runs to catch re-registration fraud before committing |
| **Offline identification** | Tap *Authenticate* → pipeline runs fully on-device, no network needed |
| **Server provisioning** | Tap *Provision* → pulls enrolled templates from `awsSyncUrl` (offline-only if not configured) |
| **Attendance sync** | Tap *Sync* → pushes queued attendance events to the mock backend and purges acknowledged rows |

---

## Quick Start

### 1. Install dependencies

```sh
# From the repo root
npm install

# From this directory
cd example
npm install

# iOS only
cd ios && pod install && cd ..
```

### 2. Verify TFLite models

The models must be in `example/assets/models/`. They are committed in this repo:

```
example/assets/models/
├── blazeface.tflite       ~0.5 MB   Face detection
├── face_mesh.tflite       ~3.0 MB   468 landmarks + head pose
├── facenet_512.tflite     ~4.0 MB   512-d face embedding
├── spoof_2_7.tflite       ~1.0 MB   Passive liveness (2.7× scale)
└── spoof_4_0.tflite       ~1.0 MB   Passive liveness (4.0× scale)
                          ─────────
                          ~9.5 MB   total
```

If missing, copy from the package root:

```sh
cp ../models/*.tflite assets/models/
```

### 3. Start Metro

```sh
npm start
```

### 4. Run on device

```sh
# Android
npm run android

# iOS
npm run ios
```

> **Physical device required** — the camera pipeline needs a real camera; emulators and simulators will not work.

---

## Configuration

All tuneable constants are at the top of [`App.tsx`](App.tsx):

```ts
// Number of face captures per enrollment (1 = fast, 3 = recommended)
const ENROLL_SHOTS = 3;

// Uncomment to enable server-based provisioning and attendance sync
// const AWS_URL      = 'http://10.0.2.2:8080';   // Android emulator → localhost
// const DEVICE_TOKEN = 'demo-device-token';
```

Pass `awsSyncUrl` and `deviceToken` into `FaceAuth.init()` to switch from fully-offline mode to
server sync mode.

---

## Mock Backend (optional)

A zero-dependency Node.js mock server is in `../server/index.js`:

```sh
# Terminal 1 — start the mock server
node ../server/index.js
# Listening on http://localhost:8080

# Terminal 2 — Android ADB tunnel (maps device port to your machine)
adb reverse tcp:8080 tcp:8080
```

Then set `AWS_URL = 'http://localhost:8080'` in `App.tsx`.

**Mock endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/provision` | Returns all enrolled templates as a full roster |
| `POST` | `/attendance` | Acknowledges all received event IDs |
| `POST` | `/enroll` | Stores a template (server-side enrollment) |
| `GET`  | `/roster` | Lists all enrolled people and template counts |
| `GET`  | `/health` | Health check |

---

## Project Structure

```
example/
├── App.tsx                  ← Main screen (enroll / identify / sync)
├── index.js                 ← Entry point
├── app.json                 ← App name and bundle ID
├── metro.config.js          ← Metro config (watchFolders, blockList, .tflite assets)
├── babel.config.js          ← Babel + module-resolver
├── react-native.config.js   ← Asset directories
├── assets/
│   └── models/              ← .tflite files loaded by useTensorflowModel
├── android/                 ← Android native project
└── ios/                     ← iOS native project (Xcode)
```

---

## Loaded Models

The app loads all five models via `useTensorflowModel` from `react-native-fast-tflite`.
The loading screen stays visible until all five report `state === 'loaded'`.

| # | Model file | Size | Pipeline stage | What it does |
|---|------------|------|----------------|--------------|
| 1 | `blazeface.tflite` | ~0.5 MB | Every frame | Face detection — bounding box + confidence |
| 2 | `face_mesh.tflite` | ~3.0 MB | Challenge + Analyze | 468 facial landmark points + head pose (yaw/pitch/roll) |
| 3 | `facenet_512.tflite` | ~4.0 MB | Analyze | 512-dimensional identity embedding for matching |
| 4 | `spoof_2_7.tflite` | ~1.0 MB | Analyze | Passive liveness — texture artifacts (prints, screens) |
| 5 | `spoof_4_0.tflite` | ~1.0 MB | Analyze | Passive liveness — geometry artifacts (3D masks, replays) |
| | **Total** | **~9.5 MB** | | |

---

## Enrollment Flow

```
Tap "Enroll"
    │
    ▼
Enter name / ID in dialog
    │
    ▼
Camera opens  (Shot 1 of ENROLL_SHOTS)
    │
    ├─ Fail → Retry or Cancel
    │
    └─ Pass → embeddingB64 captured
                │
                ▼  (first shot only)
         FaceAuth.checkDuplicate()
                │
                ├─ Duplicate found → Alert, abort
                │
                └─ No duplicate
                        │
                        ▼
                 More shots needed?
                    yes │  no
                        │   └─ FaceAuth.enrollLocal(name, [emb1…embN])
                        │      → cancelable transform applied
                        │      → encrypted on-device storage ✓
                        ▼
                 Camera opens (Shot N+1)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Loading models…" stuck | Confirm `.tflite` files exist in `assets/models/` and that `metro.config.js` has `.tflite` in `assetExts` |
| Metro crashes `ENOENT` on Windows | Already fixed — `metro.config.js` `blockList` excludes `android/app/build/` |
| Camera permission denied | Add `NSCameraUsageDescription` (iOS `Info.plist`) or `CAMERA` (Android `AndroidManifest.xml`) |
| `FaceAuth.init failed` | Ensure `react-native-keychain` native module is linked (`pod install` / Gradle sync) |
| Provision returns "no server configured" | Set `awsSyncUrl` + `deviceToken` in `FaceAuth.init()` and start the mock server |
| Build APK ~150 MB | Normal for debug — see root `README.md` for ABI split and ProGuard instructions |
