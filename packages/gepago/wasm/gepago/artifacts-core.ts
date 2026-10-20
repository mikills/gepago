import type { Candidate } from "./public-types.ts";
import type { WireObject } from "./internal-types.ts";

export const PROGRAM_ARTIFACT_FORMAT_VERSION = 1 as const;

interface ProgramArtifactValue extends WireObject {
  format_version?: unknown;
  name?: unknown;
  fingerprint?: unknown;
  candidate?: unknown;
  created_at?: unknown;
}

function assertCandidate(candidate: unknown, name = "candidate"): Candidate {
  if (
    candidate === null ||
    typeof candidate !== "object" ||
    Array.isArray(candidate) ||
    Object.keys(candidate).length === 0
  ) {
    throw new TypeError(`${name} must be a non-empty object`);
  }
  const result: Candidate = {};
  for (const [key, value] of Object.entries(candidate)) {
    if (typeof value !== "string")
      throw new TypeError(`${name} must map component names to strings`);
    result[key] = value;
  }
  return result;
}

export interface ProgramArtifact {
  format_version: 1;
  name: string;
  version?: string;
  program_version?: string;
  fingerprint?: string;
  candidate: Candidate;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export function newProgramArtifact(
  name: string,
  candidate: Candidate,
  metadata?: Record<string, unknown>,
): ProgramArtifact {
  if (typeof name !== "string" || name.trim() === "")
    throw new TypeError("artifact name is required");
  const parsedCandidate = assertCandidate(candidate);
  if (
    metadata !== undefined &&
    (metadata === null ||
      typeof metadata !== "object" ||
      Array.isArray(metadata))
  ) {
    throw new TypeError("artifact metadata must be an object");
  }
  return {
    format_version: PROGRAM_ARTIFACT_FORMAT_VERSION,
    name,
    candidate: { ...parsedCandidate },
    ...(metadata === undefined ? {} : { metadata }),
    created_at: new Date().toISOString(),
  };
}

export function parseProgramArtifact(value: unknown): ProgramArtifact {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    throw new TypeError("artifact must be an object");
  const object = value as ProgramArtifactValue;
  const format = object.format_version ?? PROGRAM_ARTIFACT_FORMAT_VERSION;
  if (format !== PROGRAM_ARTIFACT_FORMAT_VERSION)
    throw new Error(
      `unsupported program artifact format version ${String(format)}`,
    );
  if (typeof object.name !== "string" || object.name === "")
    throw new TypeError("artifact name is required");
  const candidate = assertCandidate(object.candidate, "artifact candidate");
  if (typeof object.created_at !== "string")
    throw new TypeError("artifact created_at is required");
  const result: ProgramArtifact = {
    format_version: PROGRAM_ARTIFACT_FORMAT_VERSION,
    name: object.name,
    candidate,
    created_at: object.created_at,
  };
  if (typeof object.version === "string") result.version = object.version;
  if (typeof object.program_version === "string")
    result.program_version = object.program_version;
  if (object.fingerprint !== undefined && object.fingerprint !== null) {
    if (typeof object.fingerprint !== "string")
      throw new TypeError("artifact fingerprint must be a string");
    result.fingerprint = object.fingerprint;
  }
  if (
    object.metadata !== undefined &&
    object.metadata !== null &&
    typeof object.metadata === "object" &&
    !Array.isArray(object.metadata)
  ) {
    result.metadata = { ...(object.metadata as Record<string, unknown>) };
  }
  return result;
}

export function candidateWithArtifact(
  candidate: Candidate,
  artifact: ProgramArtifact,
): Candidate {
  const parsed = parseProgramArtifact(artifact);
  return { ...assertCandidate(candidate), ...parsed.candidate };
}
