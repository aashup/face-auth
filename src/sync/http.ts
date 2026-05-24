import axios, { type AxiosInstance } from 'axios';
import { error as logError, log } from '../logger';

/** Build an axios client bound to the AWS endpoint + device bearer token. */
export function makeClient(baseURL: string, deviceToken: string): AxiosInstance {
  const client = axios.create({
    baseURL,
    timeout: 20000,
    headers: {
      Authorization: `Bearer ${deviceToken}`,
      'Content-Type': 'application/json',
    },
  });

  // ── Request interceptor ────────────────────────────────────────────────────
  client.interceptors.request.use(
    (config) => {
      const url  = (config.baseURL ?? '') + (config.url ?? '');
      const body = config.data !== undefined
        ? JSON.stringify(config.data).slice(0, 500)  // cap at 500 chars
        : '(none)';
      log(
        `[HTTP] → ${(config.method ?? 'GET').toUpperCase()} ${url}\n` +
        `[HTTP]   headers: ${JSON.stringify(config.headers)}\n` +
        `[HTTP]   body:    ${body}`,
      );
      return config;
    },
    (err) => {
      logError('[HTTP] Request setup error:', err?.message ?? err);
      return Promise.reject(err);
    },
  );

  // ── Response interceptor ───────────────────────────────────────────────────
  client.interceptors.response.use(
    (response) => {
      const body = JSON.stringify(response.data).slice(0, 500);
      log(
        `[HTTP] ← ${response.status} ${response.config?.url ?? ''}\n` +
        `[HTTP]   response: ${body}`,
      );
      return response;
    },
    (err) => {
      if (err?.response) {
        // Server replied with a non-2xx status
        const { status, data, config } = err.response;
        logError(
          `[HTTP] ✗ ${status} ${config?.url ?? ''}\n` +
          `[HTTP]   baseURL:  ${config?.baseURL ?? ''}\n` +
          `[HTTP]   headers:  ${JSON.stringify(config?.headers)}\n` +
          `[HTTP]   req body: ${config?.data ? String(config.data).slice(0, 500) : '(none)'}\n` +
          `[HTTP]   res body: ${JSON.stringify(data).slice(0, 500)}`,
        );
      } else if (err?.request) {
        // Request was made but no response received (network error / timeout)
        logError(
          `[HTTP] ✗ No response received\n` +
          `[HTTP]   url:     ${err.config?.baseURL ?? ''}${err.config?.url ?? ''}\n` +
          `[HTTP]   message: ${err?.message ?? 'unknown'}`,
        );
      } else {
        logError('[HTTP] ✗ Unexpected error:', err?.message ?? err);
      }
      return Promise.reject(err);
    },
  );

  return client;
}

/** Decode a base64 little-endian float32 vector (server template wire format). */
export function base64ToFloat32(b64: string): Float32Array {
  const bin = globalThis.atob
    ? globalThis.atob(b64)
    : (globalThis as any).Buffer.from(b64, 'base64').toString('binary');
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
}
