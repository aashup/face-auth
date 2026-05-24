const path = require('path');
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

const root = path.resolve(__dirname, '..');
const exampleModules = path.join(__dirname, 'node_modules');

/**
 * Watch the package root so Metro picks up live edits to ../src. The package
 * has no node_modules of its own, so resolve EVERY bare import (react,
 * react-native, peer deps, @babel/runtime helpers, …) from the example's
 * node_modules — a single, consistent copy. Standard create-react-native-library
 * setup.
 *
 * IMPORTANT: the root package's node_modules may contain transitive deps with
 * different versions (e.g. @react-native-async-storage/async-storage@3.x) that
 * differ from the peer-dep versions in example/node_modules. Metro's default
 * node_modules traversal finds the root version first (before extraNodeModules
 * kicks in), causing native-module-is-null crashes at runtime.
 *
 * Fix: use resolveRequest to explicitly redirect all peer-dep imports through
 * example/node_modules, regardless of where Metro's traversal would find them.
 */

// All packages that must resolve from the example's node_modules (i.e. the
// version the Android/iOS native APK was actually built against).
const PEER_DEPS = [
  '@react-native-async-storage/async-storage',
  '@react-native-community/netinfo',
  'axios',
  'react',
  'react-native',
  'react-native-fast-tflite',
  'react-native-get-random-values',
  'react-native-keychain',
  'react-native-mmkv',
  'react-native-vision-camera',
  'react-native-worklets-core',
  'vision-camera-resize-plugin',
];

const defaultConfig = getDefaultConfig(__dirname);

const config = {
  projectRoot: __dirname,
  watchFolders: [root],
  resolver: {
    // Never watch Android / iOS build artefacts — Metro crashes on Windows when
    // Gradle deletes and recreates these directories mid-build (ENOENT on watch).
    blockList: [
      /.*\/android\/\.gradle\/.*/,
      /.*\/android\/app\/build\/.*/,
      /.*\/android\/build\/.*/,
      /.*\/ios\/build\/.*/,
      /.*\/ios\/Pods\/.*/,
    ],
    // fast-tflite loads models via require(...) — register the asset extension.
    assetExts: [...defaultConfig.resolver.assetExts, 'tflite'],

    // Explicit redirect: any import whose module name starts with a peer-dep
    // prefix is resolved directly from example/node_modules, bypassing Metro's
    // normal upward traversal that would pick up wrong root-level versions.
    resolveRequest: (context, moduleName, platform) => {
      for (const dep of PEER_DEPS) {
        if (moduleName === dep || moduleName.startsWith(dep + '/')) {
          return context.resolveRequest(
            context,
            path.join(exampleModules, moduleName),
            platform,
          );
        }
      }
      return context.resolveRequest(context, moduleName, platform);
    },

    // Fallback for any other bare imports not handled above (e.g. @babel/runtime
    // helpers) — route them through example/node_modules too.
    extraNodeModules: new Proxy(
      {},
      { get: (_target, name) => path.join(exampleModules, String(name)) },
    ),
  },
};

module.exports = mergeConfig(defaultConfig, config);
