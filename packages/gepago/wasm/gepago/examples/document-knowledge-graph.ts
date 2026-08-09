import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  createGepagoWasm,
  createOpenAIModel,
  createPiModel,
} from "@mikills/wasm/gepago";
import type {
  Candidate,
  CapabilityMetricRequest,
  CompileExample,
  CompileModel,
  CompileResult,
  GepagoWasm,
  MetricResult,
  PredictModel,
  PredictProgram,
  Prediction,
  Signature,
} from "@mikills/wasm/gepago";

type Args = Record<string, string | boolean | undefined> & {
  compile?: boolean;
  help?: boolean;
  input?: string;
  mock?: boolean;
  model?: string;
  out?: string;
  provider?: string;
  "max-chars"?: string;
};
type Chunk = { id: string; text: string };
type RawEntity = {
  id?: unknown;
  name?: unknown;
  type?: unknown;
  description?: unknown;
};
type RawRelationship = {
  source?: unknown;
  type?: unknown;
  target?: unknown;
  evidence?: unknown;
};
type ModelRequest = { prompt: string };
type Model = PredictModel & CompileModel;
type NormalizedEntity = {
  id: string;
  name: string;
  type: string;
  description: string;
};
type GraphPrediction = {
  entities: RawEntity[];
  relationships: RawRelationship[];
};
type GraphNode = {
  id: string;
  stableId?: string;
  name: string;
  type: string;
  description?: string;
  mentions: string[];
};
type GraphEdge = {
  id: string;
  source: string;
  type: string;
  target: string;
  evidence: string[];
  mentions: string[];
};
type Graph = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  diagnostics: { unresolvedRelationships: number };
};
type FactSet = { chunkId: string; prediction: GraphPrediction };

const here = path.dirname(fileURLToPath(import.meta.url));
const defaultInput = path.join(here, "fixtures", "acme-robotics.md");

const extractionInstruction = `
Extract a knowledge graph from the document chunk.

Return only one JSON object with exactly these top-level fields:
{
  "entities": [
    {"id": "stable-slug", "name": "canonical name", "type": "person|organization|product|place|date|concept|other", "description": "short description"}
  ],
  "relationships": [
    {"source": "entity id", "type": "relationship type", "target": "entity id", "evidence": "short supporting quote"}
  ]
}

Rules:
- Extract only facts stated in the chunk; do not infer missing facts.
- Emit one entity per real-world entity and use the same stable slug whenever it is mentioned again.
- Relationship source and target must refer to an emitted entity id.
- Keep names and relationship types concise.
- If the chunk contains no facts, return empty arrays.
`.trim();

const signature: Signature = {
  name: "document_knowledge_graph_extraction",
  inputs: [{ name: "document" }, { name: "chunk_id" }],
  outputs: [{ name: "entities" }, { name: "relationships" }],
};

function parseArgs(argv: string[]): Args {
  const args: Args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") args.help = true;
    else if (arg === "--compile") args.compile = true;
    else if (arg === "--mock") args.mock = true;
    else if (arg === "--no-compile") args.compile = false;
    else if (arg.startsWith("--")) {
      const key = arg.slice(2);
      const value = argv[i + 1];
      if (!value || value.startsWith("--"))
        throw new Error(`${arg} requires a value`);
      args[key] = value;
      i += 1;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function usage() {
  return `Usage: node examples/document-knowledge-graph.ts [options]

Options:
  --input <path>       Markdown/text document (default: built-in fixture)
  --out <path>         Write graph JSON to this path instead of stdout
  --provider <name>    pi or openai (default: pi)
  --model <name>       Provider model; Pi uses its configured default when omitted
  --compile            Run GEPA compile first to optimize the extraction instruction
  --no-compile         Skip compile and use the seed instruction
  --max-chars <n>      Chunk size (default: 12000)
  --mock               Use a deterministic local model for smoke testing
  --help               Show this help
`;
}

function slug(value: unknown): string {
  return (
    String(value ?? "")
      .normalize("NFKD")
      .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
      .replace(/^-+|-+$/g, "")
      .toLowerCase() || "unknown"
  );
}

function label(value: unknown): string {
  return String(value ?? "")
    .trim()
    .replace(/\s+/g, " ");
}

function keyForEntity(entity: NormalizedEntity): string {
  return `${label(entity.type).toLowerCase()}|${label(entity.name).toLowerCase()}`;
}

function splitDocument(text: string, maxChars: number): Chunk[] {
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((part) => part.trim())
    .filter(Boolean);
  const chunks = [];
  let current = "";

  for (const paragraph of paragraphs) {
    if (paragraph.length > maxChars) {
      if (current) {
        chunks.push(current);
        current = "";
      }
      for (let start = 0; start < paragraph.length; start += maxChars) {
        chunks.push(paragraph.slice(start, start + maxChars));
      }
      continue;
    }
    const candidate = current ? `${current}\n\n${paragraph}` : paragraph;
    if (candidate.length > maxChars && current) {
      chunks.push(current);
      current = paragraph;
    } else {
      current = candidate;
    }
  }
  if (current) chunks.push(current);
  return chunks.map((text, index) => ({ id: `chunk-${index + 1}`, text }));
}

function safeFacts(prediction: Prediction): GraphPrediction {
  if (
    !Array.isArray(prediction.entities) ||
    !Array.isArray(prediction.relationships)
  ) {
    throw new Error(
      "model output must contain entities and relationships arrays",
    );
  }
  return {
    entities: prediction.entities as unknown as RawEntity[],
    relationships: prediction.relationships as unknown as RawRelationship[],
  };
}

function graphFromFacts(factSets: FactSet[]): Graph {
  const nodes = new Map<string, GraphNode>();
  const edges = new Map<string, GraphEdge>();
  const stableIdAliases = new Map<string, string>();
  const nameAliases = new Map<string, string>();
  const nodeIDs = new Set<string>();
  let unresolvedRelationships = 0;

  for (const { chunkId, prediction } of factSets) {
    const localRefs = new Map<string, string>();
    for (const raw of prediction.entities) {
      const entity = {
        stableId: label(raw.id) ? slug(label(raw.id)) : "",
        name: label(raw.name),
        type: label(raw.type).toLowerCase() || "other",
        description: label(raw.description),
      };
      if (!entity.name) continue;
      const stableKey = entity.stableId ? `id|${entity.stableId}` : undefined;
      const nameKey = `name|${entity.type}|${slug(entity.name)}`;
      let key = stableKey ? stableIdAliases.get(stableKey) : undefined;
      if (!key) {
        const nameMatch = nameAliases.get(nameKey);
        const nameNode = nameMatch ? nodes.get(nameMatch) : undefined;
        // A stable model ID wins over a name match. We only merge a newly
        // identified entity into a name-only node, never across two IDs.
        if (nameMatch && (!entity.stableId || !nameNode?.stableId))
          key = nameMatch;
      }
      if (!key) key = stableKey || nameKey;

      let existing = nodes.get(key);
      if (existing && entity.stableId && !existing.stableId)
        existing.stableId = entity.stableId;
      if (existing && entity.description && !existing.description)
        existing.description = entity.description;
      if (!existing) {
        const baseID = `entity-${slug(entity.stableId || `${entity.type}-${entity.name}`)}`;
        let nodeID = baseID;
        for (let suffix = 2; nodeIDs.has(nodeID); suffix += 1)
          nodeID = `${baseID}-${suffix}`;
        nodeIDs.add(nodeID);
        existing = {
          id: nodeID,
          ...(entity.stableId ? { stableId: entity.stableId } : {}),
          name: entity.name,
          type: entity.type,
          ...(entity.description ? { description: entity.description } : {}),
          mentions: [chunkId],
        };
        nodes.set(key, existing);
      } else if (!existing.mentions.includes(chunkId)) {
        existing.mentions.push(chunkId);
      }
      if (stableKey) stableIdAliases.set(stableKey, key);
      nameAliases.set(nameKey, key);
      for (const ref of [entity.stableId, entity.name, slug(entity.name)]) {
        if (ref) localRefs.set(label(ref).toLowerCase(), key);
      }
    }

    for (const raw of prediction.relationships) {
      const sourceRef = label(raw.source);
      const targetRef = label(raw.target);
      const sourceKey =
        localRefs.get(sourceRef.toLowerCase()) ||
        localRefs.get(slug(sourceRef));
      const targetKey =
        localRefs.get(targetRef.toLowerCase()) ||
        localRefs.get(slug(targetRef));
      if (
        !sourceKey ||
        !targetKey ||
        !nodes.has(sourceKey) ||
        !nodes.has(targetKey)
      ) {
        unresolvedRelationships += 1;
        continue;
      }
      const sourceNode = nodes.get(sourceKey);
      const targetNode = nodes.get(targetKey);
      if (!sourceNode || !targetNode) {
        unresolvedRelationships += 1;
        continue;
      }
      const source = sourceNode.id;
      const target = targetNode.id;
      const type = label(raw.type).toLowerCase() || "related_to";
      const edgeKey = `${source}|${type}|${target}`;
      const edge = edges.get(edgeKey);
      if (edge) {
        if (label(raw.evidence) && !edge.evidence.includes(label(raw.evidence)))
          edge.evidence.push(label(raw.evidence));
        if (!edge.mentions.includes(chunkId)) edge.mentions.push(chunkId);
      } else {
        edges.set(edgeKey, {
          id: `edge-${slug(edgeKey)}`,
          source,
          type,
          target,
          evidence: label(raw.evidence) ? [label(raw.evidence)] : [],
          mentions: [chunkId],
        });
      }
    }
  }

  return {
    nodes: [...nodes.values()].sort((a, b) => a.id.localeCompare(b.id)),
    edges: [...edges.values()].sort((a, b) => a.id.localeCompare(b.id)),
    diagnostics: { unresolvedRelationships },
  };
}

function graphMetric(request: CapabilityMetricRequest): MetricResult {
  const expected = graphFromFacts([
    { chunkId: "training", prediction: safeFacts(request.expected) },
  ]);
  const actual = graphFromFacts([
    { chunkId: "training", prediction: safeFacts(request.actual) },
  ]);
  const expectedNodes = new Set(
    expected.nodes.map((node) => `${node.type}|${node.name.toLowerCase()}`),
  );
  const actualNodes = new Set(
    actual.nodes.map((node) => `${node.type}|${node.name.toLowerCase()}`),
  );
  const expectedEdges = new Set(
    expected.edges.map((edge) => `${edge.source}|${edge.type}|${edge.target}`),
  );
  const actualEdges = new Set(
    actual.edges.map((edge) => `${edge.source}|${edge.type}|${edge.target}`),
  );

  const f1 = (want: Set<string>, got: Set<string>): number => {
    if (want.size === 0 && got.size === 0) return 1;
    if (want.size === 0 || got.size === 0) return 0;
    let matches = 0;
    for (const item of got) if (want.has(item)) matches += 1;
    const precision = matches / got.size;
    const recall = matches / want.size;
    return precision + recall === 0
      ? 0
      : (2 * precision * recall) / (precision + recall);
  };

  const nodeScore = f1(expectedNodes, actualNodes);
  const edgeScore = f1(expectedEdges, actualEdges);
  const score = nodeScore * 0.6 + edgeScore * 0.4;
  return {
    score,
    feedback: `Graph score: nodes ${(nodeScore * 100).toFixed(0)}%, relationships ${(edgeScore * 100).toFixed(0)}%. Preserve stated facts, stable entity ids, and explicit source/target relationships.`,
    details: {
      nodeScore,
      edgeScore,
      expectedNodes: expectedNodes.size,
      actualNodes: actualNodes.size,
      expectedEdges: expectedEdges.size,
      actualEdges: actualEdges.size,
    },
  };
}

function trainingExamples(): CompileExample[] {
  return [
    {
      id: "ada-lovelace",
      input: {
        chunk_id: "training-1",
        document:
          "Ada Lovelace wrote notes on Charles Babbage's Analytical Engine.",
      },
      expected: {
        entities: [
          {
            id: "ada-lovelace",
            name: "Ada Lovelace",
            type: "person",
            description: "writer",
          },
          {
            id: "charles-babbage",
            name: "Charles Babbage",
            type: "person",
            description: "engineer",
          },
          {
            id: "analytical-engine",
            name: "Analytical Engine",
            type: "concept",
            description: "a computing machine",
          },
        ],
        relationships: [
          {
            source: "ada-lovelace",
            type: "wrote-notes-on",
            target: "analytical-engine",
            evidence: "wrote notes on",
          },
          {
            source: "charles-babbage",
            type: "created",
            target: "analytical-engine",
            evidence: "Babbage's Analytical Engine",
          },
        ],
      },
    },
    {
      id: "acme-hq",
      input: {
        chunk_id: "training-2",
        document:
          "Acme Robotics, headquartered in Boston, launched the Atlas robot in 2024.",
      },
      expected: {
        entities: [
          {
            id: "acme-robotics",
            name: "Acme Robotics",
            type: "organization",
            description: "robotics company",
          },
          {
            id: "boston",
            name: "Boston",
            type: "place",
            description: "headquarters location",
          },
          { id: "atlas", name: "Atlas", type: "product", description: "robot" },
          {
            id: "2024",
            name: "2024",
            type: "date",
            description: "launch year",
          },
        ],
        relationships: [
          {
            source: "acme-robotics",
            type: "headquartered-in",
            target: "boston",
            evidence: "headquartered in Boston",
          },
          {
            source: "acme-robotics",
            type: "launched",
            target: "atlas",
            evidence: "launched the Atlas robot",
          },
          {
            source: "acme-robotics",
            type: "launched-in",
            target: "2024",
            evidence: "launched ... in 2024",
          },
        ],
      },
    },
  ];
}

function makeMockModel(): Model {
  return async ({ prompt }: ModelRequest) => {
    if (prompt.includes("Evaluation records and traces")) {
      return {
        text: "Extract only explicit people, organizations, products, places, dates, and their stated relationships. Return strict JSON with entities and relationships.",
      };
    }
    if (prompt.includes("Ada Lovelace")) {
      return { text: JSON.stringify(trainingExamples()[0].expected) };
    }
    if (prompt.includes("Acme Robotics")) {
      return { text: JSON.stringify(trainingExamples()[1].expected) };
    }
    return {
      text: JSON.stringify({
        entities: [
          {
            id: "acme-robotics",
            name: "Acme Robotics",
            type: "organization",
            description: "robotics company",
          },
          { id: "atlas", name: "Atlas", type: "product", description: "robot" },
          {
            id: "boston",
            name: "Boston",
            type: "place",
            description: "headquarters location",
          },
          {
            id: "2024",
            name: "2024",
            type: "date",
            description: "launch year",
          },
        ],
        relationships: [
          {
            source: "acme-robotics",
            type: "headquartered-in",
            target: "boston",
            evidence: "headquartered in Boston",
          },
          {
            source: "acme-robotics",
            type: "launched",
            target: "atlas",
            evidence: "launched the Atlas robot",
          },
          {
            source: "acme-robotics",
            type: "launched-in",
            target: "2024",
            evidence: "launched ... in 2024",
          },
        ],
      }),
    };
  };
}

function createModel(args: Args): Model {
  if (args.mock) return makeMockModel();
  const provider = args.provider ?? process.env.GEPA_PROVIDER ?? "pi";
  const model = args.model ?? process.env.GEPA_MODEL;
  if (provider === "pi")
    return createPiModel(model ? { model } : {}) as unknown as Model;
  if (provider === "openai")
    return createOpenAIModel({
      model: model ?? process.env.OPENAI_MODEL ?? "gpt-4.1-mini",
    }) as unknown as Model;
  throw new Error(`Unsupported provider ${provider}; use pi or openai`);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }
  const maxChars = Number(args["max-chars"] ?? 12000);
  if (!Number.isSafeInteger(maxChars) || maxChars < 500)
    throw new Error("--max-chars must be an integer >= 500");
  const inputPath = path.resolve(args.input ?? defaultInput);
  const text = await readFile(inputPath, "utf8");
  const chunks = splitDocument(text, maxChars);
  const model = createModel(args);
  const gepago = (await createGepagoWasm()) as GepagoWasm;
  const program: PredictProgram = {
    kind: "predict",
    signature,
    instruction: extractionInstruction,
    maxParseRetries: 2,
  };
  let candidate: Candidate | undefined;
  let compileResult: CompileResult | undefined;

  try {
    if (args.compile) {
      compileResult = await gepago.compile(
        {
          program,
          trainset: trainingExamples(),
          valset: trainingExamples(),
          metric: { kind: "host" },
          components: ["instruction"],
          maxMetricCalls: 8,
          minibatchSize: 1,
          seed: 7,
          acceptance: "equal",
        },
        model,
        { metric: graphMetric },
      );
      candidate = compileResult.candidate;
    }

    const factSets: FactSet[] = [];
    for (const chunk of chunks) {
      const result = candidate
        ? await gepago.runCompiled(
            program,
            candidate,
            { document: chunk.text, chunk_id: chunk.id },
            model,
          )
        : await gepago.predict(
            {
              program,
              inputs: { document: chunk.text, chunk_id: chunk.id },
            },
            model,
          );
      factSets.push({
        chunkId: chunk.id,
        prediction: safeFacts(result.prediction),
      });
    }

    const graph = graphFromFacts(factSets);
    const completedCompile = compileResult;
    const output = {
      schema: "gepago.knowledge-graph.v1",
      source: inputPath,
      chunks: chunks.map(({ id, text: chunkText }) => ({
        id,
        characters: chunkText.length,
      })),
      optimized: Boolean(candidate),
      ...(completedCompile
        ? {
            compile: {
              candidate,
              score: completedCompile.state.candidates.find(
                (item) => item.id === completedCompile.state.best_candidate_id,
              )?.validation_score,
            },
          }
        : {}),
      graph,
    };
    const encoded = `${JSON.stringify(output, null, 2)}\n`;
    if (args.out) await writeFile(path.resolve(args.out), encoded, "utf8");
    else process.stdout.write(encoded);
  } finally {
    gepago.close();
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  main().catch((error) => {
    console.error(error.stack ?? error);
    process.exitCode = 1;
  });
}

export {
  graphFromFacts,
  graphMetric,
  splitDocument,
  extractionInstruction,
  signature,
  trainingExamples,
};
