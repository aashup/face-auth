/**
 * Minimal ambient declarations for native peer dependencies, used ONLY by the
 * strict typecheck (tsconfig.typecheck.json) so we can verify the non-React
 * graph without installing the full RN toolchain. The real types come from the
 * installed peer packages in a host/example app.
 */

declare module 'axios' {
  export interface AxiosInstance {
    post<T = unknown>(url: string, body?: unknown): Promise<{ data: T }>;
  }
  const axios: { create(config: unknown): AxiosInstance };
  export default axios;
}

declare module '@react-native-async-storage/async-storage' {
  const AsyncStorage: {
    getItem(k: string): Promise<string | null>;
    setItem(k: string, v: string): Promise<void>;
    removeItem(k: string): Promise<void>;
    getAllKeys(): Promise<string[]>;
    multiGet(keys: string[]): Promise<[string, string | null][]>;
  };
  export default AsyncStorage;
}

declare module 'react-native-keychain' {
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
  export function getGenericPassword(
    o?: { service?: string },
  ): Promise<{ username: string; password: string } | false>;
  export function setGenericPassword(
    username: string,
    password: string,
    o?: SetOptions,
  ): Promise<unknown>;
  export function resetGenericPassword(o?: { service?: string }): Promise<boolean>;
}

declare module '@react-native-community/netinfo' {
  interface NetInfoState {
    isConnected: boolean | null;
    isInternetReachable: boolean | null;
  }
  const NetInfo: {
    addEventListener(cb: (state: NetInfoState) => void): () => void;
  };
  export default NetInfo;
}
