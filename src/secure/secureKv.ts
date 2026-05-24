import AsyncStorage from '@react-native-async-storage/async-storage';

import { hmacSha256, utf8 } from './sha256';

/**
 * Encrypted, synchronous-read key-value store backed by AsyncStorage.
 *
 * AsyncStorage is async and unencrypted; this wrapper adds:
 *  - a synchronous in-memory cache (hydrated once at `open()`), so the rest of
 *    the package keeps its sync `getString`/`set`/`delete` API, and
 *  - at-rest encryption: each value is sealed with an HMAC-SHA256 keystream
 *    cipher keyed by the hardware-backed key (random nonce per write), so the
 *    on-disk AsyncStorage bytes are ciphertext, never plaintext embeddings.
 *
 * Chosen as a portable, broadly-compatible alternative to react-native-mmkv.
 */
export class SecureKV {
  private cache = new Map<string, string>();
  private readonly prefix: string;
  private readonly key: Uint8Array;

  constructor(namespace: string, keyB64: string) {
    this.prefix = `sfa:${namespace}:`;
    this.key = utf8(keyB64);
  }

  /** Hydrate the in-memory cache from disk. Call once before use. */
  async open(): Promise<void> {
    const all = await AsyncStorage.getAllKeys();
    const mine = all.filter((k) => k.startsWith(this.prefix));
    if (mine.length === 0) return;
    // multiGet returns [key, value][] (AsyncStorage has no `getMany`).
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const pairs = await (AsyncStorage as any).multiGet(mine) as [string, string | null][];
    for (const [k, v] of pairs) {
      if (v != null) {
        try {
          this.cache.set(k.slice(this.prefix.length), this.decrypt(v));
        } catch {
          // Corrupt/foreign value — skip.
        }
      }
    }
  }

  getString(key: string): string | undefined {
    return this.cache.get(key);
  }

  set(key: string, value: string): void {
    this.cache.set(key, value);
    void AsyncStorage.setItem(this.prefix + key, this.encrypt(value));
  }

  delete(key: string): void {
    this.cache.delete(key);
    void AsyncStorage.removeItem(this.prefix + key);
  }

  // ── HMAC-keystream cipher (nonce ‖ ciphertext, base64) ──────────────────

  private encrypt(plaintext: string): string {
    const data = utf8(plaintext);
    const nonce = randomBytes(8);
    const ks = this.keystream(nonce, data.length);
    const out = new Uint8Array(nonce.length + data.length);
    out.set(nonce, 0);
    for (let i = 0; i < data.length; i++) out[nonce.length + i] = data[i]! ^ ks[i]!;
    return bytesToBase64(out);
  }

  private decrypt(sealed: string): string {
    const all = base64ToBytes(sealed);
    const nonce = all.slice(0, 8);
    const ct = all.slice(8);
    const ks = this.keystream(nonce, ct.length);
    const pt = new Uint8Array(ct.length);
    for (let i = 0; i < ct.length; i++) pt[i] = ct[i]! ^ ks[i]!;
    return utf8Decode(pt);
  }

  /** keystream = HMAC(key, nonce ‖ counter) blocks (32 bytes each). */
  private keystream(nonce: Uint8Array, length: number): Uint8Array {
    const out = new Uint8Array(length);
    let offset = 0;
    let counter = 0;
    while (offset < length) {
      const block = new Uint8Array(nonce.length + 4);
      block.set(nonce, 0);
      block[nonce.length] = (counter >>> 24) & 0xff;
      block[nonce.length + 1] = (counter >>> 16) & 0xff;
      block[nonce.length + 2] = (counter >>> 8) & 0xff;
      block[nonce.length + 3] = counter & 0xff;
      const ks = hmacSha256(this.key, block);
      const n = Math.min(ks.length, length - offset);
      out.set(ks.subarray(0, n), offset);
      offset += n;
      counter += 1;
    }
    return out;
  }
}

function randomBytes(n: number): Uint8Array {
  const b = new Uint8Array(n);
  const g: any = globalThis as any;
  if (g.crypto?.getRandomValues) g.crypto.getRandomValues(b);
  else for (let i = 0; i < n; i++) b[i] = Math.floor(Math.random() * 256);
  return b;
}

function utf8Decode(bytes: Uint8Array): string {
  let s = '';
  let i = 0;
  while (i < bytes.length) {
    const b = bytes[i++]!;
    if (b < 0x80) s += String.fromCharCode(b);
    else if (b < 0xe0) s += String.fromCharCode(((b & 0x1f) << 6) | (bytes[i++]! & 0x3f));
    else if (b < 0xf0)
      s += String.fromCharCode(((b & 0x0f) << 12) | ((bytes[i++]! & 0x3f) << 6) | (bytes[i++]! & 0x3f));
    else {
      const cp =
        ((b & 0x07) << 18) | ((bytes[i++]! & 0x3f) << 12) | ((bytes[i++]! & 0x3f) << 6) | (bytes[i++]! & 0x3f);
      const c = cp - 0x10000;
      s += String.fromCharCode(0xd800 + (c >> 10), 0xdc00 + (c & 0x3ff));
    }
  }
  return s;
}

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

function bytesToBase64(bytes: Uint8Array): string {
  let out = '';
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i]!;
    const b1 = i + 1 < bytes.length ? bytes[i + 1]! : 0;
    const b2 = i + 2 < bytes.length ? bytes[i + 2]! : 0;
    out += B64[b0 >> 2];
    out += B64[((b0 & 3) << 4) | (b1 >> 4)];
    out += i + 1 < bytes.length ? B64[((b1 & 15) << 2) | (b2 >> 6)] : '=';
    out += i + 2 < bytes.length ? B64[b2 & 63] : '=';
  }
  return out;
}

function base64ToBytes(b64: string): Uint8Array {
  const clean = b64.replace(/[^A-Za-z0-9+/]/g, '');
  const len = Math.floor((clean.length * 3) / 4);
  const out = new Uint8Array(len);
  let o = 0;
  for (let i = 0; i < clean.length; i += 4) {
    const c0 = B64.indexOf(clean[i]!);
    const c1 = B64.indexOf(clean[i + 1]!);
    const c2 = i + 2 < clean.length ? B64.indexOf(clean[i + 2]!) : -1;
    const c3 = i + 3 < clean.length ? B64.indexOf(clean[i + 3]!) : -1;
    if (o < len) out[o++] = (c0 << 2) | (c1 >> 4);
    if (c2 >= 0 && o < len) out[o++] = ((c1 & 15) << 4) | (c2 >> 2);
    if (c3 >= 0 && o < len) out[o++] = ((c2 & 3) << 6) | c3;
  }
  return out;
}
