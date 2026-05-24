import { getOrCreateKey } from './keystore';
import { SecureKV } from './secureKv';
import { hmacSha256, sha256, toHex, utf8 } from './sha256';

/**
 * Tamper-evident attendance / auth audit log.
 *
 * Every authentication attempt is appended as an event that (a) carries the
 * hash of the previous event (a hash chain, so deletion/reordering is
 * detectable) and (b) is HMAC-signed with the device key (so an attacker
 * without the hardware-backed key cannot forge entries). Events are uploaded by
 * the AWS sync pass (Phase 6b) and purged only after the server acks them.
 *
 * NOTE: embeddings/images are never written here — only the decision record.
 */

export interface AttendanceEvent {
  /** Idempotency key — also used by the server to de-dupe on retry. */
  id: string;
  ts: number;
  personnelId?: string;
  ok: boolean;
  matchScore: number;
  livenessScore: number;
  challengePassed: boolean;
  failureReason?: string;
  deviceId: string;
}

interface StoredEvent extends AttendanceEvent {
  prevHash: string;
  hash: string;
  sig: string;
}

const INSTANCE_ID = 'audit';
const INDEX_KEY = '__queue__';
const HEAD_KEY = '__head__'; // hash of the most recent event

let kv: SecureKV | null = null;
let signingKey: Uint8Array | null = null;

export async function openAuditLog(): Promise<void> {
  const key = await getOrCreateKey();
  kv = new SecureKV(INSTANCE_ID, key);
  await kv.open();
  signingKey = utf8(key);
}

function store(): SecureKV {
  if (!kv) throw new Error('[offline-face-auth] audit log not opened');
  return kv;
}

/** Append a signed, chained event. Returns the stored record. */
export function append(event: AttendanceEvent): StoredEvent {
  const s = store();
  const prevHash = s.getString(HEAD_KEY) ?? 'GENESIS';
  const canonical = canonicalize(event, prevHash);
  const hash = toHex(sha256(utf8(canonical)));
  const sig = toHex(hmacSha256(signingKey!, utf8(canonical)));
  const record: StoredEvent = { ...event, prevHash, hash, sig };

  s.set(event.id, JSON.stringify(record));
  const queue = readQueue(s);
  queue.push(event.id);
  s.set(INDEX_KEY, JSON.stringify(queue));
  s.set(HEAD_KEY, hash);
  return record;
}

/** Oldest-first batch of pending events for upload. */
export function pending(limit = 100): StoredEvent[] {
  const s = store();
  const ids = readQueue(s).slice(0, limit);
  const out: StoredEvent[] = [];
  for (const id of ids) {
    const raw = s.getString(id);
    if (raw) out.push(JSON.parse(raw) as StoredEvent);
  }
  return out;
}

export function pendingCount(): number {
  return readQueue(store()).length;
}

/** Purge the given event ids after the server has acked them. */
export function purge(ids: string[]): void {
  const s = store();
  const remove = new Set(ids);
  for (const id of ids) s.delete(id);
  const queue = readQueue(s).filter((id) => !remove.has(id));
  s.set(INDEX_KEY, JSON.stringify(queue));
}

/** Verify the chain + signatures of all queued events (integrity self-check). */
export function verifyChain(): boolean {
  const s = store();
  let prevHash = 'GENESIS';
  for (const id of readQueue(s)) {
    const raw = s.getString(id);
    if (!raw) return false;
    const rec = JSON.parse(raw) as StoredEvent;
    if (rec.prevHash !== prevHash) return false;
    const { hash, sig, prevHash: _p, ...event } = rec;
    const canonical = canonicalize(event as AttendanceEvent, prevHash);
    if (toHex(sha256(utf8(canonical))) !== hash) return false;
    if (toHex(hmacSha256(signingKey!, utf8(canonical))) !== sig) return false;
    prevHash = hash;
  }
  return true;
}

function canonicalize(e: AttendanceEvent, prevHash: string): string {
  // Stable field order — must match on verify.
  return JSON.stringify([
    prevHash,
    e.id,
    e.ts,
    e.personnelId ?? null,
    e.ok,
    e.matchScore,
    e.livenessScore,
    e.challengePassed,
    e.failureReason ?? null,
    e.deviceId,
  ]);
}

function readQueue(s: SecureKV): string[] {
  const raw = s.getString(INDEX_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw) as string[];
  } catch {
    return [];
  }
}
