import type { CompileCapabilities } from "./index.d.ts";

export interface FilePersistenceOptions {
  directory: string;
  stateFile?: string;
  eventFile?: string;
}

export interface PersistenceCallbacks extends Pick<
  CompileCapabilities,
  "appendEvent" | "saveState" | "loadState"
> {}

export function createFilePersistence(
  options: FilePersistenceOptions,
): PersistenceCallbacks & {
  readonly paths: Readonly<{
    directory: string;
    state: string;
    events: string;
  }>;
};

export function withFileTrainLock<T>(
  lockPath: string,
  operation: () => T | Promise<T>,
  options?: { timeoutMs?: number; pollMs?: number },
): Promise<T>;

export function createIndexedDBPersistence(options: {
  name: string;
  runID?: string;
  storeName?: string;
  indexedDB?: IDBFactory;
}): PersistenceCallbacks;
