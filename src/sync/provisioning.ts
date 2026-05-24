import type { Thresholds } from '../config';
import { transform, type CancelableParams } from '../secure/cancelable';
import type { ModelManifest } from '../secure/modelIntegrity';
import {
  getMeta,
  removePersonnel,
  replaceAll,
  setMeta,
  upsertPersonnel,
} from '../secure/templateStore';
import { updateThresholds } from '../runtime';
import type { Template } from '../matcher';
import { base64ToFloat32, makeClient } from './http';

/**
 * Server provisioning — the only online step.
 *
 * While connected, the device (1) submits a platform attestation token
 * (Play Integrity / App Attest) so the server can trust it, then (2) pulls the
 * roster of enrolled face embeddings for this device **plus** the tunable
 * threshold config and the signed model manifest. Raw embeddings arrive over
 * TLS, get the cancelable transform applied locally (binding them to the device
 * key), and are written to the encrypted template store. After this,
 * authentication runs fully offline.
 *
 * Sync is **incremental + resumable**: the device sends the timestamp of its
 * last successful provision (`since`); the server replies with either a `full`
 * roster or a `delta` (upserts + removals). `lastProvisionedAt` is only
 * advanced on success, so an interrupted sync simply re-requests the same delta
 * next time. The cancelable key and TTL are owned by the caller (`faceAuth`).
 */

export interface ServerTemplate {
  personnelId: string;
  /** base64 little-endian float32, produced by the SAME embedding model. */
  embedding: string;
}

export interface ServerUpsert {
  personnelId: string;
  /** All current templates for this personnel (replaces prior ones). */
  embeddings: string[];
}

export interface ProvisionResponse {
  mode: 'full' | 'delta';
  /** mode === 'full' */
  templates?: ServerTemplate[];
  /** mode === 'delta' */
  upserts?: ServerUpsert[];
  removals?: string[];
  /** Server-tunable decision thresholds (tighten without an app update). */
  thresholds?: Partial<Thresholds>;
  /** Signed model manifest cached for the next boot integrity check. */
  manifest?: ModelManifest;
  /** Server clock for this sync; becomes the next `since`. */
  syncedAt?: number;
}

export interface ProvisioningOptions {
  awsSyncUrl: string;
  deviceToken: string;
  cancelable: CancelableParams;
  /** Returns a fresh platform attestation token, or null if unavailable. */
  getAttestationToken?: () => Promise<string | null>;
  /** Force a full pull, ignoring the stored cursor. */
  forceFull?: boolean;
}

export interface ProvisioningResult {
  ok: boolean;
  mode?: 'full' | 'delta';
  /** Templates added/updated this sync (full count, or upserted personnel). */
  count: number;
  error?: string;
}

const LAST_PROVISIONED_AT = 'lastProvisionedAt';
const CACHED_MANIFEST = 'manifest';

/** Timestamp (ms) of the last successful provision, or 0 if never. */
export function getLastProvisionedAt(): number {
  const raw = getMeta(LAST_PROVISIONED_AT);
  return raw ? Number(raw) : 0;
}

/** The signed manifest cached from the most recent provision, if any. */
export function getCachedManifest(): ModelManifest | undefined {
  const raw = getMeta(CACHED_MANIFEST);
  if (!raw) return undefined;
  try {
    return JSON.parse(raw) as ModelManifest;
  } catch {
    return undefined;
  }
}

/** Fetch the roster (full or delta), apply it, and persist config + cursor. */
export async function provision(opts: ProvisioningOptions): Promise<ProvisioningResult> {
  try {
    const client = makeClient(opts.awsSyncUrl, opts.deviceToken);
    const attestation = opts.getAttestationToken ? await opts.getAttestationToken() : null;
    const since = opts.forceFull ? 0 : getLastProvisionedAt();

    const res = await client.post<ProvisionResponse>('/provision', { attestation, since });
    const data = res.data;
    const tx = (b64: string) => transform(base64ToFloat32(b64), opts.cancelable);

    let count = 0;
    if (data.mode === 'delta') {
      for (const up of data.upserts ?? []) {
        upsertPersonnel(up.personnelId, up.embeddings.map(tx));
        count += 1;
      }
      for (const id of data.removals ?? []) removePersonnel(id);
    } else {
      const all: Template[] = (data.templates ?? []).map((t) => ({
        personnelId: t.personnelId,
        embedding: tx(t.embedding),
      }));
      replaceAll(all);
      count = all.length;
    }

    // Apply server-tunable thresholds + cache the signed manifest.
    if (data.thresholds) updateThresholds(data.thresholds);
    if (data.manifest) setMeta(CACHED_MANIFEST, JSON.stringify(data.manifest));

    // Advance the cursor only after a fully-applied sync (resumability).
    setMeta(LAST_PROVISIONED_AT, String(data.syncedAt ?? Date.now()));

    return { ok: true, mode: data.mode, count };
  } catch (e: any) {
    return { ok: false, count: 0, error: e?.message ?? 'provisioning failed' };
  }
}
