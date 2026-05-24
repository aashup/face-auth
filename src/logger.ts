/**
 * Central debug logger for react-native-offline-face-auth.
 *
 * Controlled by `FaceAuthConfig.debug` passed to `FaceAuth.init()`.
 * When `debug: false` (the default), all log/warn/error calls are no-ops —
 * zero console noise in production builds.
 * When `debug: true`, every message is forwarded to the native console with
 * a bracketed tag so it is easy to filter in Metro / logcat / Xcode Console.
 *
 * Usage:
 *   import { log, warn, error } from './logger';
 *   log('[FaceAuth]', 'stage=', stage);   // only prints when debug=true
 */

type LogFn = (...args: unknown[]) => void;

let _debug = false;

/** Enable or disable all log output. Called by `runtime.setConfig()`. */
export function setDebug(enabled: boolean): void {
  _debug = enabled;
}

/** Returns the current debug flag. */
export function isDebug(): boolean {
  return _debug;
}

/** Print an informational message (console.log). No-op when debug=false. */
export const log: LogFn = (...args) => {
  if (_debug) console.log(...args);
};

/** Print a warning (console.warn). No-op when debug=false. */
export const warn: LogFn = (...args) => {
  if (_debug) console.warn(...args);
};

/**
 * Print an error (console.error). No-op when debug=false.
 * Note: errors that would crash the SDK always throw — this is only for
 * non-fatal diagnostic messages (e.g. "failed to save debug frame").
 */
export const error: LogFn = (...args) => {
  if (_debug) console.error(...args);
};
