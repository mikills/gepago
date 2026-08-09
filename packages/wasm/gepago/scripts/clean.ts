import { rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
const mode = process.argv[2];

if (mode !== "build" && mode !== "test" && mode !== "all") {
  throw new Error("clean mode must be build, test, or all");
}

const directories =
  mode === "build"
    ? [
        path.join(packageRoot, "dist"),
        path.join(packageRoot, "runtime", "dist"),
      ]
    : mode === "test"
      ? [path.join(packageRoot, ".test-dist")]
      : [
          path.join(packageRoot, "dist"),
          path.join(packageRoot, "runtime", "dist"),
          path.join(packageRoot, ".test-dist"),
        ];

for (const directory of directories) {
  const relative = path.relative(packageRoot, directory);
  if (
    relative === "" ||
    path.isAbsolute(relative) ||
    relative === ".." ||
    relative.startsWith(`..${path.sep}`)
  ) {
    throw new Error(
      `refusing to clean directory outside the Gepago package: ${directory}`,
    );
  }
  await rm(directory, { recursive: true, force: true });
}
