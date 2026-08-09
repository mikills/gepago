export const PROGRAM_ARTIFACT_FORMAT_VERSION: 1;
export interface ProgramArtifact {
  format_version: 1;
  name: string;
  version?: string;
  program_version?: string;
  candidate: Record<string, string>;
  metadata?: Record<string, unknown>;
  created_at: string;
}
export function newProgramArtifact(
  name: string,
  candidate: Record<string, string>,
  metadata?: Record<string, unknown>,
): ProgramArtifact;
export function parseProgramArtifact(value: unknown): ProgramArtifact;
export function candidateWithArtifact(
  candidate: Record<string, string>,
  artifact: ProgramArtifact,
): Record<string, string>;
export function saveProgramArtifact(
  path: string,
  artifact: ProgramArtifact,
): Promise<void>;
export function loadProgramArtifact(path: string): Promise<ProgramArtifact>;
