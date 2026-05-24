# Example app

Reference host for `react-native-offline-face-auth`. It consumes the package
**from source** (`../src`) via the Metro/Babel wiring in this folder, so library
edits hot-reload with no build step.

## What's here

- `App.tsx` — provision → authenticate → sync demo + diagnostics
- `index.js`, `app.json` — RN entry point
- `metro.config.js` — watches `../`, pins peer deps, registers `.tflite` assets
- `babel.config.js` — aliases the package name to `../src`, worklets plugin
- `react-native.config.js` — autolinks the package from root
- `package.json` — peer deps + `link:..` to the package

## One-time native scaffold (required before first run)

The native `android/` and `ios/` projects are **not** committed (they're large
and machine-generated). Generate them once, then keep this folder's `App.tsx`,
`index.js`, `app.json`, and the config files:

```sh
# from the repo root
npx @react-native-community/cli@latest init FaceAuthExample --version 0.75.5 --directory _tmp
# copy the generated android/ + ios/ into example/, then delete _tmp
mv _tmp/android example/android
mv _tmp/ios     example/ios
rm -rf _tmp
```

## Models

Drop the four open-source `.tflite` files into `../models/` and fill their
SHA-256 in `../models/manifest.json`. Until then the model load + boot integrity
check will fail by design. See `../models/README.md`.

## Run

```sh
cd example
npm install          # installs peer deps + links the package
npm run pods         # iOS only
npm run android      # or: npm run ios
```

## Permissions

- iOS: add `NSCameraUsageDescription` to `ios/FaceAuthExample/Info.plist`.
- Android: `<uses-permission android:name="android.permission.CAMERA" />` in
  `android/app/src/main/AndroidManifest.xml`.
