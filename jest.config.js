/** Jest config — pure-logic unit tests for the package core.
 *  Native peer modules are mapped to in-memory mocks so the security/sync
 *  layers can be exercised without a device. */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.test.ts'],
  moduleNameMapper: {
    '^@react-native-async-storage/async-storage$': '<rootDir>/__mocks__/async-storage.ts',
    '^react-native-keychain$': '<rootDir>/__mocks__/react-native-keychain.ts',
    '^@react-native-community/netinfo$': '<rootDir>/__mocks__/netinfo.ts',
  },
  transform: {
    // Transpile-only: tests run on emitted JS; type-safety is enforced separately
    // by `npm run typecheck` against the strict tsconfig.
    '^.+\\.tsx?$': ['ts-jest', { isolatedModules: true }],
  },
};
