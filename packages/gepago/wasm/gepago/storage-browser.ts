import type { CompileCapabilities, JSONValue } from "./public-types.ts";
import type { WireObject } from "./internal-types.ts";

type IndexedDBOptions = {
  name: string;
  runID?: string;
  storeName?: string;
  indexedDB?: IDBFactory;
};
type PersistenceCallbacks = Pick<
  CompileCapabilities,
  "appendEvent" | "saveState" | "loadState"
>;

function assertOptions(options: unknown): IndexedDBOptions {
  if (options === null || typeof options !== "object" || Array.isArray(options))
    throw new TypeError("IndexedDB persistence options must be an object");
  const object = options as WireObject;
  if (typeof object.name !== "string" || object.name.trim() === "")
    throw new TypeError("IndexedDB persistence name is required");
  return {
    name: object.name,
    ...(typeof object.runID === "string" ? { runID: object.runID } : {}),
    ...(typeof object.storeName === "string"
      ? { storeName: object.storeName }
      : {}),
    ...(object.indexedDB !== undefined
      ? { indexedDB: object.indexedDB as IDBFactory }
      : {}),
  };
}

function requestValue<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

/** Maps WASM persistence callbacks to browser IndexedDB. */
export function createIndexedDBPersistence(
  input: unknown,
): PersistenceCallbacks {
  const options = assertOptions(input);
  const indexedDB = options.indexedDB ?? globalThis.indexedDB;
  if (!indexedDB) throw new Error("IndexedDB is not available");
  const storeName = options.storeName ?? "gepago";
  const openDatabase: Promise<IDBDatabase> = new Promise((resolve, reject) => {
    const request = indexedDB.open(options.name, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(storeName);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error("IndexedDB open failed"));
  });
  const run = async <T>(
    mode: IDBTransactionMode,
    action: (store: IDBObjectStore) => Promise<T> | T,
  ): Promise<T> => {
    const database = await openDatabase;
    const transaction = database.transaction(storeName, mode);
    const result = await action(transaction.objectStore(storeName));
    await new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onerror = () =>
        reject(transaction.error ?? new Error("IndexedDB transaction failed"));
      transaction.onabort = () =>
        reject(transaction.error ?? new Error("IndexedDB transaction aborted"));
    });
    return result;
  };
  const runID = options.runID ?? "default";
  return {
    async appendEvent(request) {
      const key = `${runID}:events`;
      await run(
        "readwrite",
        (store) =>
          new Promise<void>((resolve, reject) => {
            const get = store.get(key);
            get.onsuccess = () => {
              const existing = Array.isArray(get.result) ? [...get.result] : [];
              existing.push(request.event);
              const put = store.put(existing, key);
              put.onsuccess = () => resolve();
              put.onerror = () =>
                reject(put.error ?? new Error("IndexedDB event write failed"));
            };
            get.onerror = () =>
              reject(get.error ?? new Error("IndexedDB event read failed"));
          }),
      );
    },
    async saveState(request) {
      await run("readwrite", (store) => {
        store.put(request.state, `${runID}:state`);
      });
    },
    async loadState() {
      return run<JSONValue>(
        "readonly",
        async (store) =>
          (await requestValue<JSONValue | undefined>(
            store.get(`${runID}:state`),
          )) ?? null,
      );
    },
  };
}
