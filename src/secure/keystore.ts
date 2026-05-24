import * as Keychain from 'react-native-keychain';

/**
 * Hardware-backed key management.
 *
 * Generates a 256-bit data-encryption key on first run and stores it via
 * react-native-keychain, which is backed by the Android Keystore (StrongBox
 * when available) / iOS Keychain (Secure Enclave). The raw key bytes are kept
 * only transiently in JS to seed the encrypted MMKV store; access is gated by
 * the platform so the key is not extractable from a rooted/jailbroken dump as
 * easily as a plain file.
 */

const SERVICE = 'react-native-offline-face-auth.dek';

const BASE_OPTIONS: Keychain.SetOptions = {
  service: SERVICE,
  accessible: Keychain.ACCESSIBLE.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
};

/** Return the data-encryption key (base64), creating + persisting it if absent.
 *  Prefers hardware-backed storage (StrongBox/TEE) but degrades gracefully on
 *  devices without secure hardware (e.g. many 3 GB phones) rather than failing. */
export async function getOrCreateKey(): Promise<string> {
  const existing = await Keychain.getGenericPassword({ service: SERVICE });
  if (existing && existing.password) {
    return existing.password;
  }
  const key = generateKeyBase64(32); // 256-bit
  try {
    // Prefer secure hardware when the device truly supports it.
    await Keychain.setGenericPassword('dek', key, {
      ...BASE_OPTIONS,
      securityLevel: Keychain.SECURITY_LEVEL.SECURE_HARDWARE,
    });
  } catch {
    // Fall back to the best available (TEE-backed Keystore / software).
    await Keychain.setGenericPassword('dek', key, BASE_OPTIONS);
  }
  return key;
}

/** Rotate the key (revokes all cancelable templates derived under the old one). */
export async function rotateKey(): Promise<string> {
  await Keychain.resetGenericPassword({ service: SERVICE });
  return getOrCreateKey();
}

export async function clearKey(): Promise<void> {
  await Keychain.resetGenericPassword({ service: SERVICE });
}

function generateKeyBase64(bytes: number): string {
  const buf = new Uint8Array(bytes);
  const g: any = globalThis as any;
  if (g.crypto?.getRandomValues) {
    g.crypto.getRandomValues(buf);
  } else {
    // Fallback only — host should provide a CSPRNG (react-native-get-random-values).
    for (let i = 0; i < bytes; i++) buf[i] = Math.floor(Math.random() * 256);
  }
  return base64Encode(buf);
}

function base64Encode(bytes: Uint8Array): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let out = '';
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i]!;
    const b1 = i + 1 < bytes.length ? bytes[i + 1]! : 0;
    const b2 = i + 2 < bytes.length ? bytes[i + 2]! : 0;
    out += chars[b0 >> 2];
    out += chars[((b0 & 3) << 4) | (b1 >> 4)];
    out += i + 1 < bytes.length ? chars[((b1 & 15) << 2) | (b2 >> 6)] : '=';
    out += i + 2 < bytes.length ? chars[b2 & 63] : '=';
  }
  return out;
}
