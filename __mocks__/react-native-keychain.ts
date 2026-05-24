/** In-memory keychain mock. */
const creds = new Map<string, { username: string; password: string }>();

export enum ACCESSIBLE {
  WHEN_UNLOCKED_THIS_DEVICE_ONLY = 'AccessibleWhenUnlockedThisDeviceOnly',
}
export enum SECURITY_LEVEL {
  SECURE_HARDWARE = 'SECURE_HARDWARE',
}

export interface SetOptions {
  service?: string;
  accessible?: ACCESSIBLE;
  securityLevel?: SECURITY_LEVEL;
}

export async function getGenericPassword(o?: { service?: string }) {
  const v = creds.get(o?.service ?? 'default');
  return v ? v : false;
}

export async function setGenericPassword(username: string, password: string, o?: SetOptions) {
  creds.set(o?.service ?? 'default', { username, password });
  return true;
}

export async function resetGenericPassword(o?: { service?: string }) {
  return creds.delete(o?.service ?? 'default');
}

export function __resetKeychain(): void {
  creds.clear();
}
