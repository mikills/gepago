import type { CompileCapabilities } from "./index.d.ts";

export interface PersistenceCallbacks extends Pick<
  CompileCapabilities,
  "appendEvent" | "saveState" | "loadState"
> {}

export interface IndexedDBPersistenceOptions {
  name: string;
  runID?: string;
  storeName?: string;
  indexedDB?: IDBFactory;
}

export function createIndexedDBPersistence(
  options: IndexedDBPersistenceOptions,
): PersistenceCallbacks;
