import { sha256, toHex } from './sha256';

/**
 * Boot-time model integrity check.
 *
 * Each bundled `.tflite` file's SHA-256 is compared against the (server-signed)
 * manifest. If any model has been swapped or tampered with, we refuse to run —
 * a defense against an attacker substituting a permissive liveness/embedding
 * model on a compromised device.
 *
 * Reading the model bytes requires platform file access, so the host/native
 * layer injects a `readModelBytes` function; this module owns only the hashing
 * and comparison so it stays pure and testable.
 */

export interface ModelManifestEntry {
  sha256: string;
  version: string;
  input: number[];
  dims?: number;
}

export type ModelManifest = Record<string, ModelManifestEntry>;

export type ModelReader = (filename: string) => Promise<Uint8Array>;

export interface IntegrityResult {
  ok: boolean;
  /** filenames whose digest did not match (or could not be read). */
  failures: string[];
}

export async function verifyModels(
  manifest: ModelManifest,
  read: ModelReader,
): Promise<IntegrityResult> {
  const failures: string[] = [];
  for (const [filename, entry] of Object.entries(manifest)) {
    if (!entry.sha256) {
      // Empty digest = manifest not finalized; treat as a failure in production.
      failures.push(filename);
      continue;
    }
    try {
      const bytes = await read(filename);
      const digest = toHex(sha256(bytes));
      if (digest.toLowerCase() !== entry.sha256.toLowerCase()) {
        failures.push(filename);
      }
    } catch {
      failures.push(filename);
    }
  }
  return { ok: failures.length === 0, failures };
}
