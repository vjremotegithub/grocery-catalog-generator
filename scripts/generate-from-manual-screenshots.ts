import "dotenv/config";
import fs from "fs/promises";
import path from "path";
import OpenAI from "openai";
import sharp from "sharp";

type StoreKey = "tesco" | "sainsburys" | "aldi";

type CommonItem = {
  slug: string;
  canonicalName: string;
  category: string;
  keywords: string[];
};

type StoreConfig = {
  brand: string;
  searchUrl: string;
};

type CatalogOption = {
  name: string;
  brand: string;
  size: string | null;
  unit: string | null;
  price: number | null;
  loyaltyPrice: number | null;
  buy2Price: number | null;
  confidence?: number;
  evidenceText?: string;
};

type CatalogItem = {
  id: string;
  store: StoreKey;
  canonicalName: string;
  keywords: string[];
  category: string;
  brand: string;
  size: string | null;
  unit: string | null;
  name: string;
  price: number | null;
  loyaltyPrice: number | null;
  options: CatalogOption[];
  lastUpdated: string;
};

const ROOT = process.cwd();
const MANUAL_ROOT = path.join(ROOT, "manual-screenshots");
const OUTPUT_DIR = path.join(ROOT, "output");

function processedFilePath(store: StoreKey) {
  return path.join(OUTPUT_DIR, `${store}.processed.json`);
}

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const model = process.env.OPENAI_VISION_MODEL || "gpt-4o";
const minConfidence = Number(process.env.MIN_CONFIDENCE || 0.65);

function slugify(value: string) {
  return value
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function getRequestedStores(allStores: StoreKey[]) {
  const index = process.argv.findIndex((arg) => arg === "--stores");
  if (index === -1) return allStores;

  const value = process.argv[index + 1];
  if (!value) return allStores;

  return value
    .split(",")
    .map((s) => s.trim())
    .filter((s): s is StoreKey => allStores.includes(s as StoreKey));
}

function getLimit() {
  const index = process.argv.findIndex((arg) => arg === "--limit");
  if (index === -1) return null;

  const value = Number(process.argv[index + 1]);
  return Number.isNaN(value) ? null : value;
}

function getRequestedItemSlugs(): string[] | null {
  const index = process.argv.findIndex((arg) => arg === "--items");
  if (index === -1) return null;

  const value = process.argv[index + 1];
  if (!value) return null;

  return value
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function singularise(value: string) {
  return value
    .toLowerCase()
    .replace(/\bpotatoes\b/g, "potato")
    .replace(/\btomatoes\b/g, "tomato")
    .replace(/\bblueberries\b/g, "blueberry")
    .replace(/\bstrawberries\b/g, "strawberry")
    .replace(/\braspberries\b/g, "raspberry")
    .replace(/\beggs\b/g, "egg")
    .replace(/\bbananas\b/g, "banana")
    .replace(/\bapples\b/g, "apple")
    .replace(/\bcarrots\b/g, "carrot")
    .replace(/\bonions\b/g, "onion")
    .replace(/\bpeppers\b/g, "pepper")
    .replace(/\bchillies\b/g, "chilli")
    .replace(/\bberries\b/g, "berry");
}

function getVariantTags(value: string) {
  const text = value.toLowerCase();
  const tags = new Set<string>();

  if (text.includes("organic")) tags.add("organic");
  if (text.includes("lactose free") || text.includes("lactose-free") || text.includes("lactofree")) tags.add("lactose-free");
  if (text.includes("gluten free") || text.includes("gluten-free")) tags.add("gluten-free");
  if (text.includes("vegan")) tags.add("vegan");
  if (text.includes("vegetarian")) tags.add("vegetarian");
  if (text.includes("plant based") || text.includes("plant-based") || text.includes("meat free") || text.includes("meat-free")) tags.add("plant-based");

  if (
    text.includes("finest") ||
    text.includes("taste the difference") ||
    text.includes("specially selected") ||
    text.includes("extra special") ||
    text.includes("the best") ||
    text.includes("premium")
  ) {
    tags.add("premium");
  }

  if (
    text.includes("everyday essentials") ||
    text.includes("stockwell") ||
    text.includes("hubbards") ||
    text.includes("just essentials") ||
    text.includes("value") ||
    text.includes("basics") ||
    text.includes("wonky")
  ) {
    tags.add("value");
  }

  if (text.includes("frozen")) tags.add("frozen");
  if (text.includes("fresh")) tags.add("fresh");
  if (text.includes("free range") || text.includes("free-range")) tags.add("free-range");

  if (text.includes("semi skimmed") || text.includes("semi-skimmed")) {
    tags.add("semi-skimmed");
  } else if (text.includes("skimmed")) {
    tags.add("skimmed");
  }

  if (
    text.includes("whole milk") ||
    text.includes("whole pasteurised milk") ||
    text.includes("whole 3.5") ||
    text.includes("whole 3.6") ||
    text.includes("whole 3.7")
  ) {
    tags.add("whole");
  }

  return [...tags];
}

function variantsCompatible(canonical: string, productName: string) {
  const canonicalTags = getVariantTags(canonical);
  const productTags = getVariantTags(productName);

  const protectedTags = [
    "organic",
    "lactose-free",
    "gluten-free",
    "vegan",
    "vegetarian",
    "plant-based",
    "premium",
    "value",
    "frozen",
    "fresh",
    "free-range",
    "skimmed",
    "semi-skimmed",
    "whole",
  ];

  for (const tag of protectedTags) {
    const canonicalHas = canonicalTags.includes(tag);
    const productHas = productTags.includes(tag);

    if (canonicalHas !== productHas) return false;
  }

  return true;
}

function hasChickenBreastMatch(product: string, base: string) {
  if (!base.includes("chicken breast")) return false;

  return (
    product.includes("chicken breast") ||
    product.includes("breast fillet") ||
    product.includes("breast fillets") ||
    product.includes("chicken fillet") ||
    product.includes("chicken fillets")
  );
}

function hasWashingUpLiquidMatch(product: string, base: string) {
  if (!base.includes("washing up liquid")) return false;

  return (
    product.includes("washing up liquid") ||
    product.includes("washing-up liquid") ||
    product.includes("dishwashing liquid") ||
    product.includes("fairy liquid") ||
    product.includes("fairy original")
  );
}

function isComparableProduct(name: string, canonical: string) {
  const product = singularise(name);
  const base = singularise(canonical);

  const blocked = ["recipe", "ready meal", "baby", "pouch"];

  if (blocked.some((blockedWord) => product.includes(blockedWord))) {
    return false;
  }

  if (!variantsCompatible(base, product)) {
    return false;
  }

  if (hasChickenBreastMatch(product, base)) return true;
  if (hasWashingUpLiquidMatch(product, base)) return true;

  return base
    .split(" ")
    .filter((word) => word.length > 2)
    .some((word) => product.includes(word));
}

async function assertImageClearEnough(imagePath: string) {
  const image = sharp(imagePath);
  const metadata = await image.metadata();

  if (!metadata.width || !metadata.height) {
    throw new Error(`Image unreadable: ${imagePath}`);
  }

  if (metadata.width < 600 || metadata.height < 500) {
    throw new Error(`Image too small/unclear: ${imagePath}`);
  }

  const stats = await image.greyscale().stats();
  const sharpness = stats.channels[0]?.stdev ?? 0;

  if (sharpness < 12) {
    throw new Error(`Image looks blurry/low contrast: ${imagePath}`);
  }
}

async function createImageCrops(
  imagePath: string,
  store: StoreKey,
  item: CommonItem
) {
  await assertImageClearEnough(imagePath);

  const metadata = await sharp(imagePath).metadata();

  if (!metadata.width || !metadata.height) {
    return [imagePath];
  }

  const cropDir = path.join(
    OUTPUT_DIR,
    "crops",
    item.slug || slugify(item.canonicalName),
    store
  );

  await fs.mkdir(cropDir, { recursive: true });

  const cropPaths: string[] = [];

  const cropAreas = [
    {
      name: "full",
      left: 0,
      top: 0,
      width: metadata.width,
      height: metadata.height,
    },
    {
      name: "top",
      left: 0,
      top: 0,
      width: metadata.width,
      height: Math.round(metadata.height * 0.6),
    },
    {
      name: "middle",
      left: 0,
      top: Math.round(metadata.height * 0.2),
      width: metadata.width,
      height: Math.round(metadata.height * 0.6),
    },
    {
      name: "bottom",
      left: 0,
      top: Math.round(metadata.height * 0.4),
      width: metadata.width,
      height: Math.round(metadata.height * 0.6),
    },
  ];

  for (const crop of cropAreas) {
    const safeTop = Math.min(crop.top, metadata.height - 1);
    const safeHeight = Math.min(crop.height, metadata.height - safeTop);

    const cropPath = path.join(cropDir, `${crop.name}.jpg`);

    await sharp(imagePath)
      .extract({
        left: crop.left,
        top: safeTop,
        width: crop.width,
        height: safeHeight,
      })
      .jpeg({ quality: 95 })
      .toFile(cropPath);

    cropPaths.push(cropPath);
  }

  return cropPaths;
}

async function imagePathToInput(imagePath: string) {
  const img = await fs.readFile(imagePath);

  return {
    type: "input_image" as const,
    image_url: `data:image/jpeg;base64,${img.toString("base64")}`,
    detail: "high" as const,
  };
}

async function extractProducts(
  imagePath: string,
  item: CommonItem,
  store: StoreKey,
  brand: string
): Promise<CatalogOption[]> {
  const cropPaths = await createImageCrops(imagePath, store, item);
  const imageInputs = await Promise.all(cropPaths.map(imagePathToInput));

  const res = await openai.responses.create({
    model,
    input: [
      {
        role: "system",
        content:
          "You extract visible UK supermarket product cards from screenshots. You must never guess. Return JSON only.",
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: `
You are reading UK supermarket product listing screenshot crops.

STRICT RULES:
- Only extract products clearly visible in the screenshots.
- Do not guess product names, brands, sizes, prices, loyalty prices or offers.
- Do not infer anything from memory.
- If the price is not clearly visible, use null.
- Only include products matching: "${item.canonicalName}".
- Include singular/plural equivalents, e.g. potato/potatoes, tomato/tomatoes, egg/eggs.
- Exclude sponsored, recipe, non-grocery and non-comparable products.
- Exclude lactose-free, organic, premium, frozen, value or free-range variants unless the canonical item asks for that variant.
- For premium equivalence only:
  Tesco Finest = Sainsbury's Taste the Difference = Aldi Specially Selected.
- For standard items, do NOT include premium ranges.
- Return maximum 8 options.
- Prices must be numbers, for example 1.25 not "£1.25".
- If Clubcard or Nectar price exists and is clearly visible, set loyaltyPrice.
- Aldi usually has no loyalty price; use null unless clearly visible.
- If multi-buy like "2 for £3" exists and is clearly visible, set buy2Price to 3.
- Add confidence from 0 to 1.
- Add evidenceText: the exact visible text you used for the product and price.
- If you are unsure, lower confidence below ${minConfidence}.

Store: ${store}
Brand fallback: ${brand}
Canonical item: ${item.canonicalName}
Category: ${item.category}
Keywords: ${item.keywords.join(", ")}
`,
          },
          ...imageInputs,
        ],
      },
    ],
    text: {
      format: {
        type: "json_schema",
        name: "products",
        strict: true,
        schema: {
          type: "object",
          additionalProperties: false,
          properties: {
            options: {
              type: "array",
              items: {
                type: "object",
                additionalProperties: false,
                properties: {
                  name: { type: "string" },
                  brand: { type: "string" },
                  size: { type: ["string", "null"] },
                  unit: { type: ["string", "null"] },
                  price: { type: ["number", "null"] },
                  loyaltyPrice: { type: ["number", "null"] },
                  buy2Price: { type: ["number", "null"] },
                  confidence: { type: "number", minimum: 0, maximum: 1 },
                  evidenceText: { type: "string" },
                },
                required: [
                  "name",
                  "brand",
                  "size",
                  "unit",
                  "price",
                  "loyaltyPrice",
                  "buy2Price",
                  "confidence",
                  "evidenceText",
                ],
              },
            },
          },
          required: ["options"],
        },
      },
    },
  });

  const parsed = JSON.parse(res.output_text) as { options: CatalogOption[] };

  const itemSlug = item.slug || slugify(item.canonicalName);

  await fs.mkdir(path.join(OUTPUT_DIR, "debug", itemSlug), {
    recursive: true,
  });

  await fs.writeFile(
    path.join(OUTPUT_DIR, "debug", itemSlug, `${store}.raw.json`),
    JSON.stringify(parsed, null, 2)
  );

  return parsed.options
    .filter((option) => (option.confidence ?? 0) >= minConfidence)
    .filter((option) => option.price !== null)
    .filter((option) => option.evidenceText && option.evidenceText.length > 5)
    .filter((option) => isComparableProduct(option.name, item.canonicalName));
}

function cleanTextValue(value: string | null | undefined) {
  const text = String(value ?? "").trim();
  if (!text || text.toLowerCase() === "null") return null;
  return text;
}

function cleanOption(option: CatalogOption, store: StoreKey): CatalogOption {
  return {
    ...option,
    unit: cleanTextValue(option.unit),
    size: cleanTextValue(option.size),
    loyaltyPrice: store === "aldi" ? null : option.loyaltyPrice,
    buy2Price: option.buy2Price ?? null,
  };
}

async function pathExists(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function listImages(folderPath: string) {
  const entries = await fs.readdir(folderPath, { withFileTypes: true });

  return entries
    .filter((entry) => entry.isFile())
    .map((entry) => path.join(folderPath, entry.name))
    .filter((filePath) =>
      /\.(jpg|jpeg|png|webp)$/i.test(filePath)
    )
    .sort();
}

async function discoverManualItems(
  allItems: CommonItem[],
  requestedStores: StoreKey[]
) {
  const discovered: Array<{
    item: CommonItem;
    store: StoreKey;
    imagePaths: string[];
  }> = [];

  const missing: Array<{
    item: string;
    slug: string;
    store: StoreKey;
    reason: string;
    expectedPath: string;
  }> = [];

  for (const item of allItems) {
    const itemSlug = item.slug || slugify(item.canonicalName);
    const itemDir = path.join(MANUAL_ROOT, itemSlug);
    const itemDirExists = await pathExists(itemDir);

    for (const store of requestedStores) {
      const storeDir = path.join(itemDir, store);

      if (!itemDirExists) {
        missing.push({
          item: item.canonicalName,
          slug: itemSlug,
          store,
          reason: "item folder missing",
          expectedPath: itemDir,
        });
        continue;
      }

      if (!(await pathExists(storeDir))) {
        missing.push({
          item: item.canonicalName,
          slug: itemSlug,
          store,
          reason: "store folder missing",
          expectedPath: storeDir,
        });
        continue;
      }

      const imagePaths = await listImages(storeDir);

      if (!imagePaths.length) {
        missing.push({
          item: item.canonicalName,
          slug: itemSlug,
          store,
          reason: "folder empty (no .png/.jpg/.jpeg/.webp)",
          expectedPath: storeDir,
        });
        continue;
      }

      discovered.push({ item, store, imagePaths });
    }
  }

  if (missing.length) {
    const lines = missing.map(
      (m) => `  - ${m.item} [${m.store}] — ${m.reason}: ${m.expectedPath}`
    );
    throw new Error(
      `Missing screenshots for ${missing.length} item/store combination(s):\n${lines.join("\n")}`
    );
  }

  return discovered;
}

type ProcessedMap = Record<string, string>;
type ProcessedByStore = Partial<Record<StoreKey, ProcessedMap>>;

async function loadProcessedForStore(store: StoreKey): Promise<ProcessedMap> {
  const file = processedFilePath(store);
  if (!(await pathExists(file))) return {};
  try {
    return JSON.parse(await fs.readFile(file, "utf8")) as ProcessedMap;
  } catch {
    return {};
  }
}

async function markProcessed(
  processedByStore: ProcessedByStore,
  store: StoreKey,
  slug: string
) {
  const map = processedByStore[store] ?? {};
  map[slug] = new Date().toISOString();
  processedByStore[store] = map;
  await fs.writeFile(processedFilePath(store), JSON.stringify(map, null, 2));
}

async function flushCatalog(store: StoreKey, newItems: CatalogItem[]) {
  const catalogPath = path.join(OUTPUT_DIR, `${store}-catalog.json`);

  let existing: CatalogItem[] = [];
  if (await pathExists(catalogPath)) {
    try {
      existing = JSON.parse(await fs.readFile(catalogPath, "utf8"));
    } catch {
      existing = [];
    }
  }

  const merged = new Map<string, CatalogItem>();
  for (const item of existing) merged.set(item.id, item);
  for (const item of newItems) merged.set(item.id, item);

  const catalog = [...merged.values()].sort((a, b) =>
    a.canonicalName.localeCompare(b.canonicalName)
  );

  await fs.writeFile(catalogPath, JSON.stringify(catalog, null, 2));
  return catalog.length;
}

async function main() {
  if (!process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY.includes("your_")) {
    throw new Error("Missing real OPENAI_API_KEY in .env");
  }

  const allItems: CommonItem[] = JSON.parse(
    await fs.readFile("manual-screenshots/items.json", "utf8")
  );

  const stores: Record<StoreKey, StoreConfig> = JSON.parse(
    await fs.readFile("data/store-search-urls.json", "utf8")
  );

  await fs.mkdir(OUTPUT_DIR, { recursive: true });

  const requestedStores = getRequestedStores(Object.keys(stores) as StoreKey[]);
  const limit = getLimit();
  const requestedItemSlugs = getRequestedItemSlugs();

  let itemsToProcess = allItems;
  if (requestedItemSlugs) {
    const allSlugs = new Set(
      allItems.map((i) => i.slug || slugify(i.canonicalName))
    );
    const unknown = requestedItemSlugs.filter((s) => !allSlugs.has(s));
    if (unknown.length) {
      throw new Error(
        `Unknown item slug(s) in --items: ${unknown.join(", ")}.\n` +
          `Valid slugs are listed in manual-screenshots/items.json.`
      );
    }
    const wanted = new Set(requestedItemSlugs);
    itemsToProcess = allItems.filter((i) =>
      wanted.has(i.slug || slugify(i.canonicalName))
    );
  }

  const discovered = await discoverManualItems(itemsToProcess, requestedStores);

  const processedByStore: ProcessedByStore = {};
  for (const store of requestedStores) {
    processedByStore[store] = await loadProcessedForStore(store);
  }
  const explicitItemFilter = requestedItemSlugs !== null;

  const eligible = discovered.filter((entry) => {
    const slug = entry.item.slug || slugify(entry.item.canonicalName);
    if (explicitItemFilter) return true; // --items always reprocesses
    return !processedByStore[entry.store]?.[slug];
  });

  const skippedCount = discovered.length - eligible.length;
  const selected = limit ? eligible.slice(0, limit) : eligible;
  const now = new Date().toISOString();

  const newItemsByStore: Record<StoreKey, CatalogItem[]> = {
    tesco: [],
    sainsburys: [],
    aldi: [],
  };

  console.log(`Using model: ${model}`);
  console.log(`Minimum confidence: ${minConfidence}`);
  console.log(`Manual screenshots root: ${MANUAL_ROOT}`);
  if (requestedItemSlugs) {
    console.log(`Filtered to ${itemsToProcess.length} item(s): ${requestedItemSlugs.join(", ")}`);
  }
  console.log(`Found ${discovered.length} item/store screenshot sets`);
  if (skippedCount) {
    const files = requestedStores.map((s) => processedFilePath(s)).join(" ");
    console.log(
      `Skipping ${skippedCount} already-processed (delete ${files} to reprocess all)`
    );
  }
  console.log(`Will process ${selected.length} this run`);

  for (const entry of selected) {
    const { item, store, imagePaths } = entry;
    const config = stores[store];
    const itemSlug = item.slug || slugify(item.canonicalName);
    const catalogId = `${store}-${itemSlug}`;

    console.log(`Processing ${item.canonicalName} / ${store}`);

    const allOptions: CatalogOption[] = [];

    for (const imagePath of imagePaths) {
      console.log(`  Image: ${imagePath}`);

      const extracted = await extractProducts(
        imagePath,
        item,
        store,
        config.brand
      );

      allOptions.push(...extracted);
    }

    const cleanedOptions = allOptions
      .map((option) => cleanOption(option, store))
      .filter((option) => option.price !== null)
      .sort((a, b) => {
        const aPrice = a.loyaltyPrice ?? a.price ?? Number.MAX_SAFE_INTEGER;
        const bPrice = b.loyaltyPrice ?? b.price ?? Number.MAX_SAFE_INTEGER;
        return aPrice - bPrice;
      });

    const dedupedOptions = Array.from(
      new Map(
        cleanedOptions.map((option) => [
          `${option.name}-${option.size}-${option.price}-${option.loyaltyPrice}`,
          option,
        ])
      ).values()
    );

    if (!dedupedOptions.length) {
      throw new Error(
        `No reliable options found for ${item.canonicalName} / ${store}. ` +
          `Check screenshots in manual-screenshots/${itemSlug}/${store}/ ` +
          `or review output/debug/${itemSlug}/${store}.raw.json`
      );
    }

    const best = dedupedOptions[0];

    const newItem: CatalogItem = {
      id: catalogId,
      store,
      canonicalName: item.canonicalName,
      keywords: item.keywords,
      category: item.category,
      brand: best.brand || config.brand,
      size: best.size,
      unit: best.unit,
      name: best.name,
      price: best.price,
      loyaltyPrice: best.loyaltyPrice,
      options: dedupedOptions,
      lastUpdated: now,
    };

    newItemsByStore[store].push(newItem);

    // Flush catalog + checkpoint after every successful item so a crash mid-run
    // never loses completed work.
    const total = await flushCatalog(store, [newItem]);
    await markProcessed(processedByStore, store, itemSlug);

    console.log(
      `  OK: ${best.name} | £${best.price}` +
        (best.loyaltyPrice ? ` | loyalty £${best.loyaltyPrice}` : "") +
        ` | options ${dedupedOptions.length} | catalog now ${total} items`
    );
  }

  for (const store of requestedStores) {
    console.log(
      `Done ${store}: ${newItemsByStore[store].length} new/updated this run`
    );
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});