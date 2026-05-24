/** Minimal NetInfo mock. */
type Listener = (s: { isConnected: boolean | null; isInternetReachable: boolean | null }) => void;
const listeners = new Set<Listener>();

const NetInfo = {
  addEventListener(cb: Listener): () => void {
    listeners.add(cb);
    return () => listeners.delete(cb);
  },
};

/** Test helper: simulate a connectivity change. */
export function __emit(isConnected: boolean, isInternetReachable = true): void {
  for (const cb of listeners) cb({ isConnected, isInternetReachable });
}

export default NetInfo;
