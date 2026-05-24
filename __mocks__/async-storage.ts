/** In-memory AsyncStorage mock (default export, async API). */
const store = new Map<string, string>();

const AsyncStorage = {
  async getItem(k: string): Promise<string | null> {
    return store.has(k) ? store.get(k)! : null;
  },
  async setItem(k: string, v: string): Promise<void> {
    store.set(k, v);
  },
  async removeItem(k: string): Promise<void> {
    store.delete(k);
  },
  async getAllKeys(): Promise<string[]> {
    return Array.from(store.keys());
  },
  async multiGet(keys: string[]): Promise<[string, string | null][]> {
    return keys.map((k) => [k, store.has(k) ? store.get(k)! : null]);
  },
};

export function __resetAsyncStorage(): void {
  store.clear();
}

export default AsyncStorage;
