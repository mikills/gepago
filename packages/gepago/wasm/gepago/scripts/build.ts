import { execFileSync } from "node:child_process";
import { copyFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
const repositoryRoot = path.resolve(packageRoot, "../../../..");

function option(name: string, fallback: string): string {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

const source = option(
  "--source",
  process.env.GEPAGO_WASM_SOURCE || "./cmd/wasm/gepago",
);
const outputDirectory = path.resolve(
  option(
    "--out-dir",
    process.env.GEPAGO_WASM_OUT_DIR || path.join(packageRoot, "dist"),
  ),
);
const wasmName = option("--wasm-name", "gepago.wasm");
const wasmPath = path.join(outputDirectory, wasmName);

function run(
  command: string,
  args: string[],
  options: { cwd?: string; env?: NodeJS.ProcessEnv } = {},
): string {
  try {
    return execFileSync(command, args, {
      encoding: "utf8",
      stdio: "pipe",
      ...options,
    });
  } catch (error) {
    const stderr = (error as NodeJS.ErrnoException & { stderr?: Buffer }).stderr
      ?.toString()
      .trim();
    const detail = stderr ? `\n${stderr}` : "";
    throw new Error(`${command} ${args.join(" ")} failed${detail}`, {
      cause: error,
    });
  }
}

await mkdir(outputDirectory, { recursive: true });
run("go", ["build", "-o", wasmPath, source], {
  cwd: repositoryRoot,
  env: { ...process.env, GOOS: "js", GOARCH: "wasm" },
});

const goroot = (
  process.env.GOROOT || run("go", ["env", "GOROOT"], { cwd: repositoryRoot })
).trim();
const wasmExecSource = path.join(goroot, "lib", "wasm", "wasm_exec.js");
const bundledRuntimeDirectory = path.join(packageRoot, "runtime", "dist");
const bundledRuntimePath = path.join(bundledRuntimeDirectory, "wasm_exec.js");
const publishedRuntimePath = path.join(outputDirectory, "wasm_exec.js");
await mkdir(bundledRuntimeDirectory, { recursive: true });
await copyFile(wasmExecSource, bundledRuntimePath);
await copyFile(wasmExecSource, publishedRuntimePath);

for (const declaration of [
  "index.d.ts",
  "browser.d.ts",
  "providers.d.ts",
  "providers-browser.d.ts",
  "storage.d.ts",
  "storage-browser.d.ts",
  "artifacts.d.ts",
  "artifacts-core.d.ts",
]) {
  await copyFile(
    path.join(packageRoot, declaration),
    path.join(outputDirectory, declaration),
  );
}

console.log(`built ${path.relative(repositoryRoot, wasmPath)}`);
console.log(`copied ${path.relative(repositoryRoot, bundledRuntimePath)}`);
console.log(`source: ${source}`);
