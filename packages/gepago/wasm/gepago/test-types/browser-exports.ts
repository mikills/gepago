import {
  createClaudeChat,
  createGoogleChat,
  createOpenAIChat,
  type HTTPProviderOptions,
} from "@mikills/wasm/gepago/providers";
import {
  createIndexedDBPersistence,
  type IndexedDBPersistenceOptions,
} from "@mikills/wasm/gepago/storage";
import {
  candidateWithArtifact,
  newProgramArtifact,
  type ProgramArtifact,
} from "@mikills/wasm/gepago/artifacts";

type Assert<T extends true> = T;
type BrowserProviders = typeof import("@mikills/wasm/gepago/providers");
type BrowserStorage = typeof import("@mikills/wasm/gepago/storage");
type BrowserArtifacts = typeof import("@mikills/wasm/gepago/artifacts");
type PiIsNodeOnly = Assert<
  "createPiModel" extends keyof BrowserProviders ? false : true
>;
type FilePersistenceIsNodeOnly = Assert<
  "createFilePersistence" extends keyof BrowserStorage ? false : true
>;
type ArtifactIOIsNodeOnly = Assert<
  "saveProgramArtifact" extends keyof BrowserArtifacts ? false : true
>;
const browserConditionsAreSafe:
  PiIsNodeOnly | FilePersistenceIsNodeOnly | ArtifactIOIsNodeOnly = true;

const providerOptions: HTTPProviderOptions = {
  model: "browser-model",
  apiKey: "key",
};
const indexedDBOptions: IndexedDBPersistenceOptions = {
  name: "gepago-browser",
};
const artifact: ProgramArtifact = newProgramArtifact("browser", {
  instruction: "classify",
});

void createOpenAIChat(providerOptions);
void createClaudeChat(providerOptions);
void createGoogleChat(providerOptions);
void createIndexedDBPersistence(indexedDBOptions);
void candidateWithArtifact({ instruction: "base" }, artifact);
void browserConditionsAreSafe;
