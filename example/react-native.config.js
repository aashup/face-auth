/**
 * The package is JS-only (no native module); resolution is handled by the
 * Babel alias + Metro config. Disable autolinking for it so the build doesn't
 * look for a non-existent android/ios native project.
 */
module.exports = {
  dependencies: {
    'react-native-offline-face-auth': {
      platforms: { android: null, ios: null },
    },
  },
};
