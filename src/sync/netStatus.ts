import NetInfo from '@react-native-community/netinfo';

/**
 * Connectivity watcher. Fires `onReconnect` when the device transitions from
 * offline → online (the moment to flush queued attendance to AWS), and runs a
 * periodic timer as a fallback so a long-lived connection still syncs.
 */

export interface NetWatcherOptions {
  intervalMs: number;
  onReconnect: () => void;
}

export interface NetWatcher {
  stop: () => void;
  /** Current best-known connectivity. */
  isOnline: () => boolean;
}

export function startNetWatcher(opts: NetWatcherOptions): NetWatcher {
  let online = true;
  let stopped = false;

  const unsubscribe = NetInfo.addEventListener((state) => {
    const nowOnline = Boolean(state.isConnected && state.isInternetReachable !== false);
    if (nowOnline && !online && !stopped) {
      opts.onReconnect();
    }
    online = nowOnline;
  });

  const timer = setInterval(() => {
    if (!stopped && online) opts.onReconnect();
  }, opts.intervalMs);

  return {
    isOnline: () => online,
    stop: () => {
      stopped = true;
      unsubscribe();
      clearInterval(timer);
    },
  };
}
