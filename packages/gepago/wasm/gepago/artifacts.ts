import { mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import path from "node:path";
import type { Candidate } from "./public-types.ts";
import {
  newProgramArtifact,
  parseProgramArtifact,
  candidateWithArtifact,
  PROGRAM_ARTIFACT_FORMAT_VERSION,
  type ProgramArtifact,
} from "./artifacts-core.ts";

export {
  newProgramArtifact,
  parseProgramArtifact,
  candidateWithArtifact,
  PROGRAM_ARTIFACT_FORMAT_VERSION,
};

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

export async function saveProgramArtifact(
  filePath: string,
  artifact: ProgramArtifact,
): Promise<void> {
  if (typeof filePath !== "string" || filePath.trim() === "")
    throw new TypeError("artifact path is required");
  const parsed = parseProgramArtifact(artifact);
  const resolvedPath = path.resolve(filePath);
  await mkdir(path.dirname(resolvedPath), { recursive: true });
  await atomicWrite(resolvedPath, `${JSON.stringify(parsed, null, 2)}\n`);
}

export async function loadProgramArtifact(
  filePath: string,
): Promise<ProgramArtifact> {
  return parseProgramArtifact(JSON.parse(await readFile(filePath, "utf8")));
}
