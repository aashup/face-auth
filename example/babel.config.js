const path = require('path');
const pkg = require('../package.json');

/**
 * Resolve the package name straight to its TypeScript source (../src) so edits
 * to the library hot-reload in the example with no build step. Peer deps are
 * pinned to the example's own node_modules to avoid duplicate React copies.
 */
module.exports = {
  presets: ['module:@react-native/babel-preset'],
  plugins: [
    'react-native-worklets-core/plugin',
    [
      'module-resolver',
      {
        extensions: ['.tsx', '.ts', '.js', '.json'],
        alias: {
          [pkg.name]: path.join(__dirname, '..', pkg.source || 'src'),
        },
      },
    ],
  ],
};
