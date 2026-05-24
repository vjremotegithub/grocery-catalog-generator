import "dotenv/config";
import fs from "fs/promises";
import path from "path";
import OpenAI from "openai";

/**
 * Catalog dietary / allergen enrichment.
 *
 * Reads each per-store catalog JSON from output/, asks an LLM to assign
 * dietary and allergen tags to every option, and writes the enriched
 * catalogs back. Then mirrors the result into the comparethetrolley
 * data folder the same way the main generator does.
 *
 * Designed to run INDEPENDENTLY of the main extraction pipeline:
 *   - Reads + writes the same output/<store>-catalog.json files
 *   - Skips options that already have a tags array (idempotent)
 *   - --force re-tags everything even when tags already exist
 *   - --stores / --items CLI flags scope the run
 *
 * Cost ballpark: 3 stores × ~56 items × one gpt-4o-mini call per item =
 * ~168 LLM calls for a full enrichment. Each call processes that item's
 * 1–8 options at once via batching, keeping the total under a few cents.
 */

type StoreKey = "tesco" | "sainsburys" | "aldi";

// Must match SUPPORTED_DIETARY_TAGS in generate-from-manual-screenshots.ts
// and DietaryTag in the consumer's components/compare-results/types.ts.
// Kept literal here rather than imported to avoid pulling the main
// generator's top-level side effects (dotenv config, OpenAI client).
const SUPPORTED_TAGS = [
  "vegan",
  "vegetarian",
  "glutenFree",
  "containsNuts",
  "containsDairy",
  "containsEggs",
  "containsFish",
] as const;
type DietaryTag = (typeof SUPPORTED_TAGS)[number];

type CatalogOption = {
  optionId: string;
  name: string;
  brand?: string;
  size?: string | null;
  // ...other fields we don't touch
  tags?: DietaryTag[];
  [key: string]: unknown;
};

type CatalogItem = {
  id: string;
  store: StoreKey;
  canonicalName: string;
  category?: string;
  options: CatalogOption[];
  [key: string]: unknown;
};

const ROOT = process.cwd();
const OUTPUT_DIR = path.join(ROOT, "output");
const enrichModel = process.env.OPENAI_ENRICH_MODEL || "gpt-4o-mini";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ----------------------------- CLI parsing -----------------------------

function getCliValue(flag: string): string | null {
  const i = process.argv.findIndex((arg) => arg === flag);
  if (i === -1) return null;
  const value = process.argv[i + 1];
  return value && !value.startsWith("--") ? value : null;
}

function hasCliFlag(flag: string): boolean {
  return process.argv.includes(flag);
}

function getRequestedStores(): StoreKey[] {
  const raw = getCliValue("--stores");
  const all: StoreKey[] = ["tesco", "sainsburys", "aldi"];
  if (!raw) return all;
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s): s is StoreKey => all.includes(s as StoreKey));
}

function getRequestedItemSlugs(): Set<string> | null {
  const raw = getCliValue("--items");
  if (!raw) return null;
  return new Set(raw.split(",").map((s) => s.trim()));
}

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

// ----------------------------- LLM call --------------------------------

/**
 * Tag all options for a single catalog item in one LLM call. Batching
 * here keeps the number of round-trips proportional to items, not
 * options. Returns the tags array per option (same length / order as
 * the input options).
 */
async function tagOptionsForItem(item: CatalogItem): Promise<DietaryTag[][]> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("Missing OPENAI_API_KEY");
  }

  if (item.options.length === 0) return [];

  // Compact human-readable claims list so the model has every detail
  // it needs to judge each option, on its own line, with a stable
  // index it can refer back to in the response.
  const claims = item.options
    .map(
      (o, i) =>
        `[#${i}] ${o.brand ?? ""} ${o.name}${
          o.size ? ` (${o.size})` : ""
        }`.trim()
    )
    .join("\n");

  const res = await openai.responses.create({
    model: enrichModel,
    input: [
      {
        role: "system",
        content:
          "You assign dietary and allergen tags to UK supermarket products. " +
          "ONLY assign a tag when you are confident based on the product name, brand and category. " +
          "When uncertain, OMIT the tag — false confidence about allergens is worse than no information. " +
          "Allowed tags (use EXACTLY these strings): " +
          SUPPORTED_TAGS.join(", ") +
          ". Return JSON only.",
      },
      {
        role: "user",
        content:
          `Category: ${item.category ?? "(unknown)"}\n` +
          `Canonical item: ${item.canonicalName}\n\n` +
          `For each option below, return the applicable tags. An option ` +
          `can have multiple tags (e.g. yoghurt → ['vegetarian', 'containsDairy']). ` +
          `Use 'vegan' ONLY when no animal products are involved at all. ` +
          `Use 'vegetarian' when no meat/fish — include this for vegan options too. ` +
          `Use 'containsX' allergens only when X is clearly an ingredient.\n\n` +
          `Options:\n${claims}`,
      },
    ],
    text: {
      format: {
        type: "json_schema",
        name: "tags",
        strict: true,
        schema: {
          type: "object",
          additionalProperties: false,
          properties: {
            taggings: {
              type: "array",
              items: {
                type: "object",
                additionalProperties: false,
                properties: {
                  index: { type: "integer" },
                  tags: {
                    type: "array",
                    items: {
                      type: "string",
                      enum: [...SUPPORTED_TAGS],
                    },
                  },
                },
                required: ["index", "tags"],
              },
            },
          },
          required: ["taggings"],
        },
      },
    },
  });

  const parsed = JSON.parse(res.output_text) as {
    taggings: Array<{ index: number; tags: DietaryTag[] }>;
  };

  // Map LLM verdicts back to the original input order. Defensive: if a
  // tagging refers to an out-of-range index, drop it; if an option is
  // missing from the response, default to [] (no tags) rather than
  // crashing.
  const out: DietaryTag[][] = item.options.map(() => []);
  for (const t of parsed.taggings) {
    if (t.index < 0 || t.index >= item.options.length) continue;
    // Dedupe within an option in case the LLM repeats itself.
    out[t.index] = Array.from(new Set(t.tags));
  }
  return out;
}

// --------------------------- Store enrichment --------------------------

async function enrichStore(
  store: StoreKey,
  requestedSlugs: Set<string> | null,
  force: boolean
) {
  const catalogPath = path.join(OUTPUT_DIR, `${store}-catalog.json`);

  if (!(await pathExists(catalogPath))) {
    console.warn(`Skipping ${store}: ${catalogPath} not found.`);
    return;
  }

  const catalog = JSON.parse(
    await fs.readFile(catalogPath, "utf8")
  ) as CatalogItem[];

  let processed = 0;
  let skipped = 0;
  let updatedOptions = 0;
  let matchedAny = false;

  for (const item of catalog) {
    const itemSlug = item.id.replace(`${store}-`, "");

    if (requestedSlugs && !requestedSlugs.has(itemSlug)) continue;
    matchedAny = true;

    // Idempotency: if every option already has a tags array AND we're
    // not forcing, skip the item entirely.
    const everyOptionTagged = item.options.every((o) => Array.isArray(o.tags));
    if (everyOptionTagged && !force) {
      skipped += 1;
      continue;
    }

    console.log(`  ${item.canonicalName}: tagging ${item.options.length} option(s)...`);

    let perOptionTags: DietaryTag[][];
    try {
      perOptionTags = await tagOptionsForItem(item);
    } catch (error) {
      console.error(
        `  ${item.canonicalName}: tagging failed — ${error instanceof Error ? error.message : error}`
      );
      continue;
    }

    for (let i = 0; i < item.options.length; i++) {
      item.options[i].tags = perOptionTags[i] ?? [];
      updatedOptions += 1;
    }

    processed += 1;

    // Flush the catalog file after each item so a crash mid-run never
    // loses completed work — mirrors the main generator's pattern.
    await fs.writeFile(catalogPath, JSON.stringify(catalog, null, 2));
  }

  console.log(
    `Done ${store}: processed ${processed} item(s), skipped ${skipped} (already tagged), ` +
      `updated ${updatedOptions} option(s).`
  );

  // The most common confusion mode: a developer types --items yoghurt
  // but the catalog id slug is "yogurt" (US spelling). Surface the
  // available slugs so they can re-run with a known-good value rather
  // than wondering why nothing happened.
  if (requestedSlugs && !matchedAny) {
    const availableSlugs = catalog
      .map((item) => item.id.replace(`${store}-`, ""))
      .sort();
    console.warn(
      `  No items matched --items=${[...requestedSlugs].join(",")} in ${store}.\n` +
        `  Available slugs in ${store}-catalog.json:\n  ${availableSlugs.join(", ")}`
    );
  }
}

// ------------------------ Mirror to downstream -------------------------

async function mirrorCatalogsToDownstream(stores: StoreKey[]) {
  const destDir =
    process.env.COMPARETROLLEY_CATALOG_DIR ||
    "/Users/vj/acp-ho/github-ai/comparethetrolley/data/catalogs";

  if (!(await pathExists(destDir))) {
    console.warn(
      `Skipping downstream mirror: destination ${destDir} does not exist. ` +
        `Set COMPARETROLLEY_CATALOG_DIR to override.`
    );
    return;
  }

  for (const store of stores) {
    const source = path.join(OUTPUT_DIR, `${store}-catalog.json`);
    const target = path.join(destDir, `${store}-catalog.json`);
    if (!(await pathExists(source))) continue;
    await fs.copyFile(source, target);
    console.log(`Mirrored ${store}-catalog.json → ${target}`);
  }
}

// -------------------------------- main ---------------------------------

async function main() {
  if (!process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY.includes("your_")) {
    throw new Error("Missing real OPENAI_API_KEY in .env");
  }

  const stores = getRequestedStores();
  const requestedSlugs = getRequestedItemSlugs();
  const force = hasCliFlag("--force");

  console.log(`Enriching with model: ${enrichModel}`);
  console.log(`Stores: ${stores.join(", ")}`);
  if (requestedSlugs) console.log(`Items: ${[...requestedSlugs].join(", ")}`);
  if (force) console.log(`--force enabled: re-tagging options that already have tags.`);

  for (const store of stores) {
    console.log(`\n=== ${store} ===`);
    await enrichStore(store, requestedSlugs, force);
  }

  await mirrorCatalogsToDownstream(stores);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

// Reference slugify so the lint pass doesn't flag it; it's kept available
// for future per-canonicalName slug parsing if we ever drop the
// "<store>-<itemSlug>" id convention.
void slugify;
