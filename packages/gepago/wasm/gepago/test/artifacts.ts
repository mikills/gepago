import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  loadProgramArtifact,
  newProgramArtifact,
  parseProgramArtifact,
  saveProgramArtifact,
  type ProgramArtifact,
} from "@mikills/wasm/gepago/artifacts";

// goArtifact has every field of a ProgramArtifact saved by Go.
const goArtifact = {
  format_version: 1,
  name: "echo",
  version: "v2",
  program_version: "v1",
  fingerprint: "pipeline/v1:0123abcd",
  candidate: { "echo.instruction": "trained" },
  metadata: { score: 0.9 },
  created_at: "2026-01-02T03:04:05Z",
};

async function withDirectory(run: (directory: string) => Promise<void>) {
  const directory = await mkdtemp(
    path.join(os.tmpdir(), "gepago-artifacts-"),
  );
  try {
    await run(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

test("a Go artefact keeps its fingerprint through load and save", async () => {
  await withDirectory(async (directory) => {
    const goPath = path.join(directory, "go.json");
    const jsPath = path.join(directory, "js.json");
    await writeFile(goPath, JSON.stringify(goArtifact, null, 2));

    const loaded = await loadProgramArtifact(goPath);
    assert.deepEqual(loaded, goArtifact);
    await saveProgramArtifact(jsPath, loaded);
    assert.deepEqual(JSON.parse(await readFile(jsPath, "utf8")), goArtifact);
  });
});

test("an artefact without a fingerprint stays without one", async () => {
  await withDirectory(async (directory) => {
    const artifactPath = path.join(directory, "echo.json");
    await saveProgramArtifact(
      artifactPath,
      newProgramArtifact("echo", { "echo.instruction": "trained" }),
    );
    assert.doesNotMatch(await readFile(artifactPath, "utf8"), /fingerprint/);
    assert.equal(
      "fingerprint" in (await loadProgramArtifact(artifactPath)),
      false,
    );
    const parsed = parseProgramArtifact({ ...goArtifact, fingerprint: null });
    assert.equal("fingerprint" in parsed, false);
  });
});

test("a non-string fingerprint is rejected", async () => {
  const invalid = { ...goArtifact, fingerprint: 42 };
  assert.throws(
    () => parseProgramArtifact(invalid),
    new TypeError("artifact fingerprint must be a string"),
  );
  await withDirectory(async (directory) => {
    const artifactPath = path.join(directory, "echo.json");
    await assert.rejects(
      saveProgramArtifact(artifactPath, invalid as unknown as ProgramArtifact),
      /artifact fingerprint must be a string/,
    );
    await writeFile(artifactPath, JSON.stringify(invalid));
    await assert.rejects(
      loadProgramArtifact(artifactPath),
      /artifact fingerprint must be a string/,
    );
  });
});
