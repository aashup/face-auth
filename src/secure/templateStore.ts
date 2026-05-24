import type { Template } from '../matcher';
import { getOrCreateKey } from './keystore';
import { SecureKV } from './secureKv';

/**
 * Encrypted, on-device template store.
 *
 * Holds only cancelable-transformed embeddings (never raw images, never raw
 * embeddings) in a SecureKV instance (AsyncStorage + HMAC-keystream cipher)
 * keyed by the hardware-backed key. The embedding bytes are loaded into memory
 * only while matching and the instance is the single source of truth for the
 * provisioned roster.
 */

const INSTANCE_ID = 'templates';
const INDEX_KEY = '__index__'; // JSON array of template ids

let kv: SecureKV | null = null;

/** Open (or reopen) the encrypted store. Call once after init / provisioning. */
export async function openTemplateStore(): Promise<void> {
  const key = await getOrCreateKey();
  kv = new SecureKV(INSTANCE_ID, key);
  await kv.open();
}

function store(): SecureKV {
  if (!kv) throw new Error('[offline-face-auth] template store not opened');
  return kv;
}

/** Stable per-record id so one personnel can have multiple enrolled templates. */
function recordId(personnelId: string, idx: number): string {
  return `${personnelId}#${idx}`;
}

/** Replace the entire roster (typical after a full provisioning sync). */
export function replaceAll(templates: Template[]): void {
  const s = store();
  const ids: string[] = [];
  // wipe previous records
  for (const old of readIndex(s)) s.delete(old);
  const counters = new Map<string, number>();
  for (const t of templates) {
    const idx = counters.get(t.personnelId) ?? 0;
    counters.set(t.personnelId, idx + 1);
    const id = recordId(t.personnelId, idx);
    s.set(id, encodeTemplate(t));
    ids.push(id);
  }
  s.set(INDEX_KEY, JSON.stringify(ids));
}

/** Upsert templates for a single personnel (incremental provisioning). */
export function upsertPersonnel(personnelId: string, embeddings: Float32Array[]): void {
  const s = store();
  const ids = readIndex(s).filter((id) => !id.startsWith(`${personnelId}#`));
  for (const old of readIndex(s)) {
    if (old.startsWith(`${personnelId}#`)) s.delete(old);
  }
  embeddings.forEach((e, idx) => {
    const id = recordId(personnelId, idx);
    s.set(id, encodeTemplate({ personnelId, embedding: e }));
    ids.push(id);
  });
  s.set(INDEX_KEY, JSON.stringify(ids));
}

export function removePersonnel(personnelId: string): void {
  const s = store();
  const remaining: string[] = [];
  for (const id of readIndex(s)) {
    if (id.startsWith(`${personnelId}#`)) s.delete(id);
    else remaining.push(id);
  }
  s.set(INDEX_KEY, JSON.stringify(remaining));
}

/** Load all templates into memory for a match pass. */
export function loadAll(): Template[] {
  const s = store();
  const out: Template[] = [];
  for (const id of readIndex(s)) {
    const raw = s.getString(id);
    if (raw) out.push(decodeTemplate(raw));
  }
  return out;
}

export function count(): number {
  return readIndex(store()).length;
}

export function clearTemplates(): void {
  const s = store();
  for (const id of readIndex(s)) s.delete(id);
  s.delete(INDEX_KEY);
}

function readIndex(s: SecureKV): string[] {
  const raw = s.getString(INDEX_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw) as string[];
  } catch {
    return [];
  }
}

// ── provisioning metadata (lastProvisionedAt, cached signed manifest, …) ────
// Stored under a reserved prefix so it never collides with template ids and is
// untouched by replaceAll/clearTemplates.

const META_PREFIX = '__meta__';

export function setMeta(key: string, value: string): void {
  store().set(META_PREFIX + key, value);
}

export function getMeta(key: string): string | undefined {
  return store().getString(META_PREFIX + key);
}

// ── codec ────────────────────────────────────────────────────────────────
// Template → "personnelId|base64(float32le)". Compact + JSON-safe.

function encodeTemplate(t: Template): string {
  return `${t.personnelId}|${float32ToBase64(t.embedding)}`;
}

function decodeTemplate(raw: string): Template {
  const sep = raw.indexOf('|');
  const personnelId = raw.slice(0, sep);
  const embedding = base64ToFloat32(raw.slice(sep + 1));
  return { personnelId, embedding };
}

function float32ToBase64(v: Float32Array): string {
  const bytes = new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]!);
  return globalThis.btoa ? globalThis.btoa(bin) : bufferB64(bytes);
}

function base64ToFloat32(b64: string): Float32Array {
  const bin = globalThis.atob ? globalThis.atob(b64) : b64FromBuffer(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

// Minimal Buffer fallbacks for RN runtimes without atob/btoa.
function bufferB64(bytes: Uint8Array): string {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  return (globalThis as any).Buffer.from(bytes).toString('base64');
}
function b64FromBuffer(b64: string): string {
  return (globalThis as any).Buffer.from(b64, 'base64').toString('binary');
}
