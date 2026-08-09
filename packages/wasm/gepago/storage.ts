import {
  mkdir,
  open,
  readFile,
  rename,
  rm,
  appendFile,
  type FileHandle,
} from "node:fs/promises";
import { randomUUID } from "node:crypto";
import path from "node:path";
import type { CompileCapabilities, JSONValue } from "./public-types.ts";
import type { WireObject } from "./internal-types.ts";

type PersistenceRequest = WireObject & {
  kind?: unknown;
  event?: JSONValue;
  state?: JSONValue;
};
type PersistenceCallbacks = Pick<
  CompileCapabilities,
  "appendEvent" | "saveState" | "loadState"
> & {
  readonly paths?: Readonly<{
    directory: string;
    state: string;
    events: string;
  }>;
};
type LockOptions = { timeoutMs?: number; pollMs?: number };
type NodeError = Error & { code?: string };

function assertOptions(
  options: unknown,
  name: string,
): { directory: string; stateFile?: string; eventFile?: string } {
  if (options === null || typeof options !== "object" || Array.isArray(options))
    throw new TypeError(`${name} options must be an object`);
  const object = options as WireObject;
  if (typeof object.directory !== "string" || object.directory.trim() === "")
    throw new TypeError(`${name} directory is required`);
  return {
    directory: object.directory,
    ...(typeof object.stateFile === "string"
      ? { stateFile: object.stateFile }
      : {}),
    ...(typeof object.eventFile === "string"
      ? { eventFile: object.eventFile }
      : {}),
  };
}

function containedPath(directory: string, value: string, name: string): string {
  if (typeof value !== "string" || value.trim() === "")
    throw new TypeError(`${name} must be a non-empty string`);
  const resolved = path.resolve(directory, value);
  const relative = path.relative(directory, resolved);
  if (
    relative === "" ||
    path.isAbsolute(relative) ||
    relative === ".." ||
    relative.startsWith(`..${path.sep}`)
  ) {
    throw new TypeError(`${name} must stay inside the persistence directory`);
  }
  return resolved;
}

async function atomicWrite(filePath: string, contents: string): Promise<void> {
  const temporary = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
  let renamed = false;
  try {
    const handle = await open(temporary, "wx");
    try {
      await handle.writeFile(contents, "utf8");
      await handle.sync();
    } finally {
      await handle.close();
    }
    await rename(temporary, filePath);
    renamed = true;
  } finally {
    if (!renamed) await rm(temporary, { force: true });
  }
}

function assertRequest(request: unknown, kind: string): PersistenceRequest {
  if (
    request === null ||
    typeof request !== "object" ||
    Array.isArray(request) ||
    (request as WireObject).kind !== kind
  ) {
    throw new TypeError(
      `invalid ${kind.replace("optimizer.persistence.", "")} request`,
    );
  }
  return request as PersistenceRequest;
}

/** Maps the WASM persistence callbacks to atomic Node filesystem storage. */
export function createFilePersistence(input: unknown): PersistenceCallbacks {
  const options = assertOptions(input, "file persistence");
  const directory = path.resolve(options.directory);
  const statePath = containedPath(
    directory,
    options.stateFile ?? "optimization_state.json",
    "stateFile",
  );
  const eventPath = containedPath(
    directory,
    options.eventFile ?? "optimization_events.jsonl",
    "eventFile",
  );
  const ensure = (): Promise<string> =>
    mkdir(directory, { recursive: true }).then(() => directory);
  const ensureParent = (filePath: string): Promise<string> =>
    mkdir(path.dirname(filePath), { recursive: true }).then(() => filePath);
  const callbacks: PersistenceCallbacks = {
    async appendEvent(request) {
      const object = assertRequest(
        request,
        "optimizer.persistence.append_event",
      );
      await ensure();
      await ensureParent(eventPath);
      await appendFile(eventPath, `${JSON.stringify(object.event)}\n`, "utf8");
    },
    async saveState(request) {
      const object = assertRequest(request, "optimizer.persistence.save_state");
      await ensure();
      await ensureParent(statePath);
      await atomicWrite(statePath, JSON.stringify(object.state));
    },
    async loadState(request) {
      assertRequest(request, "optimizer.persistence.load_state");
      try {
        return JSON.parse(await readFile(statePath, "utf8")) as JSONValue;
      } catch (error) {
        if ((error as NodeError).code === "ENOENT") return null;
        throw error;
      }
    },
  };
  Object.defineProperty(callbacks, "paths", {
    value: Object.freeze({ directory, state: statePath, events: eventPath }),
  });
  return callbacks;
}

async function removeOwnedLock(lockPath: string, owner: string): Promise<void> {
  let contents: unknown;
  try {
    contents = JSON.parse(await readFile(lockPath, "utf8"));
  } catch {
    return;
  }
  if (
    contents === null ||
    typeof contents !== "object" ||
    (contents as WireObject).owner !== owner
  )
    return;
  await rm(lockPath, { force: true });
}

/** Runs one training operation under an O_EXCL cross-process lock. */
export async function withFileTrainLock<T>(
  lockPath: string,
  operation: () => T | Promise<T>,
  options: LockOptions = {},
): Promise<T> {
  if (typeof lockPath !== "string" || lockPath.trim() === "")
    throw new TypeError("lockPath must be a non-empty string");
  if (typeof operation !== "function")
    throw new TypeError("operation must be a function");
  if (options === null || typeof options !== "object" || Array.isArray(options))
    throw new TypeError("lock options must be an object");
  const timeout = options.timeoutMs ?? 30_000;
  const poll = options.pollMs ?? 50;
  if (!Number.isSafeInteger(timeout) || timeout < 0)
    throw new TypeError("lock timeoutMs must be a non-negative safe integer");
  if (!Number.isSafeInteger(poll) || poll <= 0)
    throw new TypeError("lock pollMs must be a positive safe integer");
  const resolvedLockPath = path.resolve(lockPath);
  const deadline = Date.now() + timeout;
  await mkdir(path.dirname(resolvedLockPath), { recursive: true });
  const owner = `${process.pid}:${randomUUID()}`;
  let handle: FileHandle | undefined;
  while (!handle) {
    try {
      const candidate = await open(resolvedLockPath, "wx");
      try {
        await candidate.writeFile(
          `${JSON.stringify({ owner, pid: process.pid, created_at: new Date().toISOString() })}\n`,
          "utf8",
        );
        await candidate.sync();
        handle = candidate;
      } catch (error) {
        await candidate.close();
        await removeOwnedLock(resolvedLockPath, owner);
        throw error;
      }
    } catch (error) {
      const nodeError = error as NodeError;
      if (nodeError.code !== "EEXIST" || Date.now() >= deadline) {
        throw nodeError.code === "EEXIST"
          ? new Error(`training lock timed out: ${resolvedLockPath}`)
          : error;
      }
      await new Promise<void>((resolve) => setTimeout(resolve, poll));
    }
  }
  try {
    return await operation();
  } finally {
    await handle.close();
    await removeOwnedLock(resolvedLockPath, owner);
  }
}
