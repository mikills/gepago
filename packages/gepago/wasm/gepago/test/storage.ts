import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile, readdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { indexedDB } from "fake-indexeddb";
import {
  createFilePersistence,
  withFileTrainLock,
} from "@mikills/wasm/gepago/storage";
import type {
  CapabilityPersistenceRequest,
  JSONObject,
} from "@mikills/wasm/gepago";
import { createIndexedDBPersistence } from "../storage-browser.ts";

test("file persistence and training lock provide host-side equivalents", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "gepago-storage-"));
  const persistence = createFilePersistence({ directory });
  await assert.rejects(
    Promise.resolve().then(() =>
      createFilePersistence({ directory, stateFile: "../outside.json" }),
    ),
    /inside the persistence directory/,
  );
  const saveState = persistence.saveState;
  const loadState = persistence.loadState;
  const appendEvent = persistence.appendEvent;
  assert.ok(saveState);
  assert.ok(loadState);
  assert.ok(appendEvent);
  assert.equal(
    await loadState({
      kind: "optimizer.persistence.load_state",
    } satisfies CapabilityPersistenceRequest),
    null,
  );
  const saveRequest = (run_id: string): CapabilityPersistenceRequest => ({
    kind: "optimizer.persistence.save_state",
    state: { run_id },
  });
  await Promise.all([
    saveState(saveRequest("run-1")),
    saveState(saveRequest("run-2")),
  ]);
  const state = await loadState({
    kind: "optimizer.persistence.load_state",
  } satisfies CapabilityPersistenceRequest);
  assert.ok(state && typeof state === "object" && !Array.isArray(state));
  const runID = (state as JSONObject).run_id;
  assert.equal(typeof runID === "string" && runID.startsWith("run-"), true);
  assert.deepEqual(
    (await readdir(directory)).filter((name) => name.endsWith(".tmp")),
    [],
  );
  await appendEvent({
    kind: "optimizer.persistence.append_event",
    event: { kind: "accepted" },
  } satisfies CapabilityPersistenceRequest);
  assert.match(await readFile(persistence.paths.events, "utf8"), /accepted/);
  const lockPath = path.join(directory, "train.lock");
  assert.equal(await withFileTrainLock(lockPath, () => "trained"), "trained");

  // A lock replaced by another owner must not be removed by the old owner.
  await withFileTrainLock(lockPath, async () => {
    await rm(lockPath);
    await writeFile(lockPath, JSON.stringify({ owner: "another-process" }));
  });
  assert.equal(
    JSON.parse(await readFile(lockPath, "utf8")).owner,
    "another-process",
  );
  await rm(lockPath);
});

test("IndexedDB persistence uses null for a fresh resumable state", async () => {
  const persistence = createIndexedDBPersistence({
    name: `gepago-storage-${crypto.randomUUID()}`,
    indexedDB,
  });
  const loadState = persistence.loadState;
  const saveState = persistence.saveState;
  assert.ok(loadState);
  assert.ok(saveState);

  const loadRequest = {
    kind: "optimizer.persistence.load_state",
  } satisfies CapabilityPersistenceRequest;
  assert.equal(await loadState(loadRequest), null);
  await saveState({
    kind: "optimizer.persistence.save_state",
    state: { run_id: "browser-run" },
  });
  assert.deepEqual(await loadState(loadRequest), { run_id: "browser-run" });
});
