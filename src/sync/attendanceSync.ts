import { pending, pendingCount, purge } from '../secure/auditLog';
import { makeClient } from './http';

/**
 * Attendance sync + purge (Phase 6b).
 *
 * Pushes queued, signed attendance events to AWS in oldest-first batches. Only
 * the events the server explicitly acknowledges (by id) are purged from the
 * local encrypted queue — so a dropped connection mid-batch never loses or
 * double-counts attendance (each event carries an idempotency id the server
 * de-dupes on). Embeddings/images are never uploaded, only decision records.
 */

export interface AttendanceSyncOptions {
  awsSyncUrl: string;
  deviceToken: string;
  /** Delete locally once acked. Off = keep a local copy after upload. */
  purgeAfterSync: boolean;
  /** Max events per request. */
  batchSize?: number;
}

export interface SyncOutcome {
  ok: boolean;
  uploaded: number;
  remaining: number;
  error?: string;
}

let inFlight = false;

/** Flush all pending events. Safe to call concurrently — re-entrancy guarded. */
export async function syncAttendance(opts: AttendanceSyncOptions): Promise<SyncOutcome> {
  if (inFlight) {
    return { ok: true, uploaded: 0, remaining: pendingCount() };
  }
  inFlight = true;
  const batchSize = opts.batchSize ?? 100;
  const client = makeClient(opts.awsSyncUrl, opts.deviceToken);
  let uploaded = 0;

  try {
    // Loop batches until the queue drains or a batch acks nothing.
    for (;;) {
      const batch = pending(batchSize);
      if (batch.length === 0) break;

      const res = await client.post<{ acked: string[] }>('/attendance', {
        events: batch,
      });

      const acked = res.data?.acked ?? [];
      if (acked.length === 0) {
        // Server accepted none — stop to avoid a hot retry loop.
        return { ok: false, uploaded, remaining: pendingCount(), error: 'no events acknowledged' };
      }

      if (opts.purgeAfterSync) purge(acked);
      uploaded += acked.length;

      if (acked.length < batch.length) {
        // Partial ack — remaining stay queued for the next run.
        break;
      }
    }
    return { ok: true, uploaded, remaining: pendingCount() };
  } catch (e: any) {
    return { ok: false, uploaded, remaining: pendingCount(), error: e?.message ?? 'sync failed' };
  } finally {
    inFlight = false;
  }
}
