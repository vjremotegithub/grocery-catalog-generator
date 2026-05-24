import "dotenv/config";
import fs from "fs/promises";
import path from "path";
import OpenAI from "openai";
import sharp from "sharp";

type StoreKey = "tesco" | "sainsburys" | "aldi" | "lidl" | "asda" | "morrisons";

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

// The raw shape returned by the LLM vision extractor. It does NOT include
// optionId — that's assigned post-extraction once the option has been
// cleaned and deduped. Keeping these two shapes apart prevents accidental
// "we forgot to set optionId" bugs from compiling.
type LlmExtractedOption = {
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

// The shape we actually emit to the per-store catalog JSON. The required
// optionId is the stable per-SKU identifier the downstream app uses to
// attach a unit to a basket-level Deal (e.g. "any 3 for 2"). It MUST be
// derived deterministically from content so a re-scrape of the same
// product lands on the same id — see makeOptionId().
//
// `tags` is populated by the separate enrich-catalog-tags.ts pass —
// optional here so the main generator can emit options without doing the
// extra LLM call, and a follow-up enrichment run fills them in.
type CatalogOption = LlmExtractedOption & {
  optionId: string;
  tags?: DietaryTag[];
};

// Closed set of supported dietary / allergen tags. Two intents combined
// in one list so a single tags[] field covers both filters (vegan,
// vegetarian, glutenFree) and allergens (containsNuts, containsDairy,
// containsEggs, containsFish). The consumer can split them in the UI.
export const SUPPORTED_DIETARY_TAGS = [
  "vegan",
  "vegetarian",
  "glutenFree",
  "containsNuts",
  "containsDairy",
  "containsEggs",
  "containsFish",
] as const;
export type DietaryTag = (typeof SUPPORTED_DIETARY_TAGS)[number];

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

// Cross-SKU promotional deal, surfaced by the LLM from the same
// screenshots it's reading product cards from. The LLM returns
// memberOptionNames (free-text product names visible on the banner);
// post-processing resolves those against the just-emitted options to
// populate the canonical memberOptionIds the consumer expects.
type LlmExtractedDeal = {
  label: string;            // "Any 3 for £6"
  kind: "n_for_m" | "bundle_price";
  requiredQty: number;      // >= 2
  paidQty: number | null;   // for n_for_m
  bundlePrice: number | null; // for bundle_price
  memberOptionNames: string[]; // names from the SAME response
  confidence: number;
  evidenceText: string;
};

// Output shape — matches the consumer's Deal type in
// components/compare-results/types.ts. Keep these in sync.
type CatalogDeal = {
  id: string;
  store: StoreKey;
  label: string;
  rule:
    | { kind: "n_for_m"; requiredQty: number; paidQty: number }
    | { kind: "bundle_price"; requiredQty: number; bundlePrice: number };
  memberOptionIds: string[];
  // Provenance — useful for QA / debugging the LLM extraction. Not
  // consumed by the runtime deal engine.
  evidenceText?: string;
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

// QA pass — second LLM call that re-reads each screenshot and validates
// the just-extracted options/deals against what's actually visible. Off
// by default so existing `npm run generate` cost doesn't double silently;
// set QA_PASS=on (or =1, =true) to enable. Cheaper model than the main
// vision extractor because it only has to judge yes/no against a small,
// concrete claim — verification is much easier than extraction.
const qaPassEnabled = ((process.env.QA_PASS || "").toLowerCase() ===
  "on" ||
  process.env.QA_PASS === "1" ||
  process.env.QA_PASS === "true");
const qaModel = process.env.OPENAI_QA_MODEL || "gpt-4o-mini";

function slugify(value: string) {
  return value
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

/**
 * Stable per-option identifier. Built from content (catalog id, brand,
 * name, size) so a re-scrape that produces the same product lands on the
 * same id automatically — no preservation logic needed for the common
 * case.
 *
 * Notes:
 *   - PRICE IS DELIBERATELY EXCLUDED. We want price changes (a routine
 *     event) to leave the id stable, so any cross-SKU Deal that referenced
 *     this option keeps working through normal price churn.
 *   - Brand is included so different brands of the same product (e.g.
 *     "Cravendale Whole Milk 2pt" vs "Tesco Whole Milk 2pt" both appearing
 *     under the canonical "milk" item) don't collide on id.
 *   - Empty brand/size segments are skipped — concatenating them produces
 *     double hyphens otherwise. Catalog id + name alone is still unique
 *     enough in the empty-brand edge case.
 *   - If the LLM ever renames a product between runs (e.g. "British
 *     Apples" → "Apples British"), the id will drift. That's acceptable
 *     for v1; we can add a name-similarity reconciliation pass against
 *     the existing catalog if drift becomes a real problem in practice.
 */
function makeOptionId(
  catalogId: string,
  option: { name: string; brand: string; size: string | null }
): string {
  const parts = [
    catalogId,
    option.brand ? slugify(option.brand) : null,
    slugify(option.name),
    option.size ? slugify(option.size) : null,
  ].filter((part): part is string => Boolean(part));
  return parts.join("-");
}

/**
 * Stable deal identifier. Built from the deal's PROMO SHAPE, NOT the
 * catalog id of the item that surfaced it — so the same banner observed
 * across multiple item screenshots (e.g. "Any 3 for 2" visible on both
 * milk and yoghurt pages) deduplicates to a single deal with merged
 * memberOptionIds.
 */
function makeDealId(
  store: StoreKey,
  deal: { label: string; kind: string; requiredQty: number; paidQty: number | null; bundlePrice: number | null }
): string {
  const qty = deal.paidQty ?? deal.bundlePrice ?? 0;
  return `${store}-${slugify(deal.label)}-${deal.kind}-${deal.requiredQty}-${qty}`;
}

/**
 * Resolve LLM-reported memberOptionNames against the optionIds we
 * actually emitted for this catalog item, producing CatalogDeals ready
 * for the consumer.
 *
 * Strict-acceptance gate: a deal survives only if at least 2 of its
 * member names resolve to real optionIds. Hallucinated deals (where
 * names don't appear in our options list) collapse to zero members
 * and get dropped. The "at least 2" rule reflects that these are
 * meant to be CROSS-SKU deals; a single-member result usually means
 * the LLM misread a per-product offer.
 */
function resolveDealMembers(
  store: StoreKey,
  llmDeals: LlmExtractedDeal[],
  options: CatalogOption[]
): CatalogDeal[] {
  // Index by normalised name for fast lookup, plus a case-insensitive
  // contains fallback for slight wording variations.
  const byNorm = new Map<string, string>();
  for (const opt of options) {
    byNorm.set(opt.name.toLowerCase().trim(), opt.optionId);
  }

  function resolveName(name: string): string | null {
    const norm = name.toLowerCase().trim();
    const exact = byNorm.get(norm);
    if (exact) return exact;
    // Fuzzy fallback: pick the option whose name contains the query or
    // vice-versa. Conservative — only used when the exact lookup fails.
    for (const opt of options) {
      const optNorm = opt.name.toLowerCase();
      if (optNorm.includes(norm) || norm.includes(optNorm)) {
        return opt.optionId;
      }
    }
    return null;
  }

  const resolved: CatalogDeal[] = [];

  for (const deal of llmDeals) {
    const ids = Array.from(
      new Set(
        deal.memberOptionNames
          .map((n) => resolveName(n))
          .filter((id): id is string => id !== null)
      )
    );

    // Cross-SKU deals need ≥ 2 members to be meaningful. Single-member
    // "deals" are almost certainly per-product offers that belong on
    // buy2Price, not here.
    if (ids.length < 2) continue;

    const rule =
      deal.kind === "n_for_m"
        ? {
            kind: "n_for_m" as const,
            requiredQty: deal.requiredQty,
            paidQty: deal.paidQty as number,
          }
        : {
            kind: "bundle_price" as const,
            requiredQty: deal.requiredQty,
            bundlePrice: deal.bundlePrice as number,
          };

    resolved.push({
      id: makeDealId(store, deal),
      store,
      label: deal.label,
      rule,
      memberOptionIds: ids,
      evidenceText: deal.evidenceText,
    });
  }

  return resolved;
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

function getRequestedCategory(): string | null {
  const index = process.argv.findIndex((arg) => arg === "--category");
  if (index === -1) return null;

  const value = process.argv[index + 1];
  return value?.trim() || null;
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
    text.includes("premium") ||
    // Lidl premium
    text.includes("deluxe") ||
    // Asda premium
    text.includes("asda extra special") ||
    // Morrisons premium
    text.includes("morrisons the best")
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
    text.includes("wonky") ||
    // Asda value
    text.includes("smart price") ||
    // Morrisons value
    text.includes("morrisons savers") ||
    text.includes("savers")
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
): Promise<{ options: LlmExtractedOption[]; deals: LlmExtractedDeal[] }> {
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

CROSS-SKU PROMOTIONAL DEALS (separate "deals" array in your response):
- ONLY include a deal if a promotional banner / badge is clearly visible
  on the screenshot alongside the product cards. Examples: "Any 3 for £6",
  "3 for 2", "Buy 2 save £1", "Mix & Match — any 3 for £5".
- Use kind="n_for_m" with paidQty for offers like "3 for 2" (buy 3, pay
  for 2). Set bundlePrice to null.
- Use kind="bundle_price" with bundlePrice for offers like "Any 3 for £6".
  Set paidQty to null.
- memberOptionNames MUST list the EXACT names from THIS response's options
  array that participate in the deal. If you can't be sure which on-screen
  products the banner covers, DO NOT include the deal — better to skip than
  guess. Single-SKU "buy 2 for £3" stickers belong on buy2Price, NOT here.
- Return an empty deals array if no banner is visible. Never invent deals.
- Add confidence + evidenceText for each deal exactly like options.

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
            deals: {
              type: "array",
              items: {
                type: "object",
                additionalProperties: false,
                properties: {
                  label: { type: "string" },
                  kind: {
                    type: "string",
                    enum: ["n_for_m", "bundle_price"],
                  },
                  requiredQty: {
                    type: "integer",
                    minimum: 2,
                  },
                  // Exactly ONE of paidQty / bundlePrice is non-null
                  // depending on kind. Schema permits both as nullable
                  // for simplicity; post-processing enforces the rule.
                  paidQty: { type: ["integer", "null"] },
                  bundlePrice: { type: ["number", "null"] },
                  memberOptionNames: {
                    type: "array",
                    items: { type: "string" },
                  },
                  confidence: { type: "number", minimum: 0, maximum: 1 },
                  evidenceText: { type: "string" },
                },
                required: [
                  "label",
                  "kind",
                  "requiredQty",
                  "paidQty",
                  "bundlePrice",
                  "memberOptionNames",
                  "confidence",
                  "evidenceText",
                ],
              },
            },
          },
          required: ["options", "deals"],
        },
      },
    },
  });

  const parsed = JSON.parse(res.output_text) as {
    options: LlmExtractedOption[];
    deals: LlmExtractedDeal[];
  };

  const itemSlug = item.slug || slugify(item.canonicalName);

  await fs.mkdir(path.join(OUTPUT_DIR, "debug", itemSlug), {
    recursive: true,
  });

  await fs.writeFile(
    path.join(OUTPUT_DIR, "debug", itemSlug, `${store}.raw.json`),
    JSON.stringify(parsed, null, 2)
  );

  const options = parsed.options
    .filter((option) => (option.confidence ?? 0) >= minConfidence)
    .filter((option) => option.price !== null)
    .filter((option) => option.evidenceText && option.evidenceText.length > 5)
    .filter((option) => isComparableProduct(option.name, item.canonicalName));

  // Light filtering of deals at the extractor boundary; the heavy lifting
  // (resolving member names → optionIds) happens later, once all options
  // for this item have been cleaned and deduped.
  const deals = (parsed.deals ?? []).filter((deal) => {
    if ((deal.confidence ?? 0) < minConfidence) return false;
    if (!Array.isArray(deal.memberOptionNames) || deal.memberOptionNames.length === 0) {
      return false;
    }
    if (deal.requiredQty < 2) return false;
    if (deal.kind === "n_for_m") {
      if (deal.paidQty == null) return false;
      if (deal.paidQty < 1) return false;
      if (deal.paidQty >= deal.requiredQty) return false;
    }
    if (deal.kind === "bundle_price") {
      if (deal.bundlePrice == null) return false;
      if (deal.bundlePrice <= 0) return false;
    }
    return true;
  });

  return { options, deals };
}

// -----------------------------------------------------------------------
// QA pass: re-read the screenshot and verify the extracted claims.
//
// The main extractor (extractProducts) self-reports a confidence score
// per option, but those scores are not very reliable in practice — the
// model is generally over-confident about prices it can read but
// mis-paired with the wrong product card. Verification is a much easier
// task than extraction: we hand the model a CONCRETE claim ("Tesco
// British Apples Min 5 Pack £1.80, loyalty £1.20") and ask whether that
// exact text appears on the visible card. A small model handles this
// reliably and at a fraction of the cost of the main extractor.
//
// Returned verdicts:
//   - "ok"        → the claim matches what's on screen
//   - "wrong"     → the model is confident the claim is false (e.g. the
//                   price was misread, the product card isn't on this
//                   image at all, or the size doesn't match)
//   - "uncertain" → the image is ambiguous (cropped, blurred, partial).
//                   Kept by default; the caller can choose stricter
//                   filtering if desired.
//
// The caller is responsible for dropping "wrong" verdicts before they
// reach the persisted catalog.
// -----------------------------------------------------------------------

type QaVerdictKind = "ok" | "wrong" | "uncertain";

type QaVerdict = {
  index: number;
  verdict: QaVerdictKind;
  reason: string;
};

type QaResult = {
  options: QaVerdict[];
  deals: QaVerdict[];
};

async function validateExtraction(
  imagePath: string,
  item: CommonItem,
  store: StoreKey,
  options: LlmExtractedOption[],
  deals: LlmExtractedDeal[]
): Promise<QaResult> {
  // Nothing to validate; short-circuit so we don't burn an API call on
  // an empty extraction.
  if (options.length === 0 && deals.length === 0) {
    return { options: [], deals: [] };
  }

  const cropPaths = await createImageCrops(imagePath, store, item);
  const imageInputs = await Promise.all(cropPaths.map(imagePathToInput));

  // Build the claims list the model has to judge. Keep each claim on
  // one line and prefixed with its index so the model's JSON output
  // can refer back to it unambiguously.
  const optionClaims = options
    .map(
      (o, i) =>
        `[opt#${i}] ${o.brand || ""} ${o.name} | size=${o.size ?? "n/a"} | ` +
        `price=${o.price ?? "null"} | loyaltyPrice=${o.loyaltyPrice ?? "null"} | ` +
        `buy2Price=${o.buy2Price ?? "null"}`
    )
    .join("\n");

  const dealClaims = deals
    .map(
      (d, i) =>
        `[deal#${i}] "${d.label}" kind=${d.kind} requiredQty=${d.requiredQty} ` +
        `paidQty=${d.paidQty ?? "null"} bundlePrice=${d.bundlePrice ?? "null"} ` +
        `members=[${d.memberOptionNames.join(", ")}]`
    )
    .join("\n");

  const res = await openai.responses.create({
    model: qaModel,
    input: [
      {
        role: "system",
        content:
          "You are a careful verifier of UK supermarket product extractions. " +
          "For each claim, look at the screenshot and decide if the claim matches what is " +
          "ACTUALLY visible. Be conservative: if the relevant card isn't fully visible or " +
          "the price/size is partly obscured, return 'uncertain' rather than 'ok' or 'wrong'. " +
          "Return JSON only.",
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text:
              `Store: ${store}\n` +
              `Canonical item: ${item.canonicalName}\n\n` +
              `OPTIONS to verify (one per line):\n` +
              (optionClaims || "(none)") +
              `\n\nDEALS to verify (one per line):\n` +
              (dealClaims || "(none)") +
              `\n\nFor each option AND each deal, return a verdict:\n` +
              `- "ok"        → the exact product/banner is visible AND the price/size/qty match\n` +
              `- "wrong"     → the claim contradicts what's visible (wrong price, wrong product, hallucinated)\n` +
              `- "uncertain" → image is ambiguous, cropped, or partly obscured\n` +
              `Give a short reason (one sentence) referring to the visible text.`,
          },
          ...imageInputs,
        ],
      },
    ],
    text: {
      format: {
        type: "json_schema",
        name: "qa",
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
                  index: { type: "integer" },
                  verdict: { type: "string", enum: ["ok", "wrong", "uncertain"] },
                  reason: { type: "string" },
                },
                required: ["index", "verdict", "reason"],
              },
            },
            deals: {
              type: "array",
              items: {
                type: "object",
                additionalProperties: false,
                properties: {
                  index: { type: "integer" },
                  verdict: { type: "string", enum: ["ok", "wrong", "uncertain"] },
                  reason: { type: "string" },
                },
                required: ["index", "verdict", "reason"],
              },
            },
          },
          required: ["options", "deals"],
        },
      },
    },
  });

  const parsed = JSON.parse(res.output_text) as QaResult;
  return parsed;
}

function cleanTextValue(value: string | null | undefined) {
  const text = String(value ?? "").trim();
  if (!text || text.toLowerCase() === "null") return null;
  return text;
}

function cleanOption(
  option: LlmExtractedOption,
  store: StoreKey
): LlmExtractedOption {
  return {
    ...option,
    unit: cleanTextValue(option.unit),
    size: cleanTextValue(option.size),
    loyaltyPrice: (store === "aldi" || store === "lidl") ? null : option.loyaltyPrice,
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
    // Folders are now organised as manual-screenshots/{category}/{slug}/
    const itemDir = path.join(MANUAL_ROOT, item.category, itemSlug);
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

/**
 * After a generation run completes, copy the produced catalog JSON for
 * each requested store into the downstream `comparethetrolley` app's
 * data folder so the site picks up fresh prices on its next build.
 *
 * Destination resolution:
 *   1. COMPARETROLLEY_CATALOG_DIR env var (preferred for CI / other machines)
 *   2. Default: /Users/vj/acp-ho/github-ai/comparethetrolley/data/catalogs
 *      (matches the layout on the maintainer's local machine where the two
 *      repos sit side-by-side under ~/acp-ho/github-ai)
 *
 * This step is intentionally NOT fatal: if the destination doesn't exist
 * (e.g. running on a CI box, or the consumer repo isn't checked out), we
 * log a warning and continue. The canonical catalogs in output/ have
 * already been written by this point, so a failed mirror never loses
 * data — the maintainer can copy them manually if they need to.
 *
 * We only mirror the stores actually processed in this run: a `--stores
 * aldi` invocation should never overwrite the consumer's tesco or
 * sainsburys catalogs with stale data.
 */
async function mirrorCatalogsToDownstream(stores: StoreKey[]) {
  const destDir =
    process.env.COMPARETROLLEY_CATALOG_DIR ||
    "/Users/vj/acp-ho/github-ai/comparethetrolley/data/catalogs";

  if (!(await pathExists(destDir))) {
    console.warn(
      `Skipping downstream catalog copy: destination ${destDir} does not exist. ` +
        `Set COMPARETROLLEY_CATALOG_DIR to override, or copy the files manually.`
    );
    return;
  }

  for (const store of stores) {
    const source = path.join(OUTPUT_DIR, `${store}-catalog.json`);
    const target = path.join(destDir, `${store}-catalog.json`);

    if (!(await pathExists(source))) {
      console.warn(
        `Skipping ${store}: source ${source} was not produced this run.`
      );
      continue;
    }

    await fs.copyFile(source, target);
    console.log(`Mirrored ${store}-catalog.json → ${target}`);

    // Sibling deals file. Always copy when present, even if zero deals
    // were emitted, so the downstream's static import always sees the
    // freshest state (an empty array overwrites a stale populated one).
    const dealsSource = path.join(OUTPUT_DIR, `${store}-deals.json`);
    const dealsTarget = path.join(destDir, `${store}-deals.json`);
    if (await pathExists(dealsSource)) {
      await fs.copyFile(dealsSource, dealsTarget);
      console.log(`Mirrored ${store}-deals.json → ${dealsTarget}`);
    }
  }
}

// -------------------------------------------------------------------------
// Change-detection / changelog.
//
// After every generation run we diff the post-run catalog + deals state
// against the pre-run snapshot and write a per-store report describing
// what moved. Two outputs per store:
//   - output/changes/<timestamp>-<store>.json  (structured, for tooling)
//   - output/changes/<timestamp>-<store>.md    (human-readable digest)
//
// Mechanical diff only — no LLM. Cheap, deterministic, easy to verify.
// -------------------------------------------------------------------------

// Surface a price change only when it's at least one of: 5% of the old
// price OR an absolute 10p movement. Small absolute deltas on cheap
// items and small percentage deltas on expensive items both pass — but
// rounding-ripple noise (1-2p, < 5%) is filtered out.
const PRICE_CHANGE_PCT_THRESHOLD = 0.05;
const PRICE_CHANGE_ABS_THRESHOLD = 0.1;

type ChangeReport = {
  store: StoreKey;
  generatedAt: string;
  itemsAdded: Array<{ id: string; canonicalName: string }>;
  itemsRemoved: Array<{ id: string; canonicalName: string }>;
  optionsAdded: Array<{
    itemId: string;
    canonicalName: string;
    optionName: string;
    price: number | null;
  }>;
  optionsRemoved: Array<{
    itemId: string;
    canonicalName: string;
    optionName: string;
    previousPrice: number | null;
  }>;
  priceChanges: Array<{
    itemId: string;
    canonicalName: string;
    optionName: string;
    previousPrice: number;
    newPrice: number;
    deltaAbs: number;
    deltaPct: number;
  }>;
  dealsAdded: CatalogDeal[];
  dealsRemoved: CatalogDeal[];
};

function effectivePrice(opt: CatalogOption): number | null {
  return opt.loyaltyPrice ?? opt.price ?? null;
}

function isSignificantPriceChange(before: number, after: number): boolean {
  const deltaAbs = Math.abs(after - before);
  const deltaPct = before > 0 ? deltaAbs / before : 0;
  return (
    deltaAbs >= PRICE_CHANGE_ABS_THRESHOLD ||
    deltaPct >= PRICE_CHANGE_PCT_THRESHOLD
  );
}

function computeChangelog(
  store: StoreKey,
  before: CatalogItem[],
  after: CatalogItem[],
  beforeDeals: CatalogDeal[],
  afterDeals: CatalogDeal[]
): ChangeReport {
  const beforeById = new Map(before.map((i) => [i.id, i]));
  const afterById = new Map(after.map((i) => [i.id, i]));

  const report: ChangeReport = {
    store,
    generatedAt: new Date().toISOString(),
    itemsAdded: [],
    itemsRemoved: [],
    optionsAdded: [],
    optionsRemoved: [],
    priceChanges: [],
    dealsAdded: [],
    dealsRemoved: [],
  };

  for (const [id, item] of afterById) {
    if (!beforeById.has(id)) {
      report.itemsAdded.push({ id, canonicalName: item.canonicalName });
    }
  }
  for (const [id, item] of beforeById) {
    if (!afterById.has(id)) {
      report.itemsRemoved.push({ id, canonicalName: item.canonicalName });
    }
  }

  // For items present in both: diff their options by optionId. Falling
  // back to position-based keys for legacy data without optionIds —
  // this only matters for catalogs generated before step 2 (the
  // optionId rollout), and accepts that those items can't be diffed
  // precisely.
  for (const [id, afterItem] of afterById) {
    const beforeItem = beforeById.get(id);
    if (!beforeItem) continue;

    const beforeOpts = beforeItem.options ?? [];
    const afterOpts = afterItem.options ?? [];
    const beforeOptKey = (o: CatalogOption, i: number) =>
      o.optionId ?? `__pos__${i}`;
    const afterOptKey = (o: CatalogOption, i: number) =>
      o.optionId ?? `__pos__${i}`;

    const beforeByKey = new Map(beforeOpts.map((o, i) => [beforeOptKey(o, i), o]));
    const afterByKey = new Map(afterOpts.map((o, i) => [afterOptKey(o, i), o]));

    for (const [key, opt] of afterByKey) {
      if (!beforeByKey.has(key)) {
        report.optionsAdded.push({
          itemId: id,
          canonicalName: afterItem.canonicalName,
          optionName: opt.name,
          price: opt.price,
        });
      }
    }
    for (const [key, opt] of beforeByKey) {
      if (!afterByKey.has(key)) {
        report.optionsRemoved.push({
          itemId: id,
          canonicalName: beforeItem.canonicalName,
          optionName: opt.name,
          previousPrice: opt.price,
        });
      }
    }
    for (const [key, afterOpt] of afterByKey) {
      const beforeOpt = beforeByKey.get(key);
      if (!beforeOpt) continue;
      const bp = effectivePrice(beforeOpt);
      const ap = effectivePrice(afterOpt);
      if (bp === null || ap === null) continue;
      if (!isSignificantPriceChange(bp, ap)) continue;
      report.priceChanges.push({
        itemId: id,
        canonicalName: afterItem.canonicalName,
        optionName: afterOpt.name,
        previousPrice: bp,
        newPrice: ap,
        deltaAbs: ap - bp,
        deltaPct: ((ap - bp) / bp) * 100,
      });
    }
  }

  const beforeDealIds = new Set(beforeDeals.map((d) => d.id));
  const afterDealIds = new Set(afterDeals.map((d) => d.id));
  report.dealsAdded = afterDeals.filter((d) => !beforeDealIds.has(d.id));
  report.dealsRemoved = beforeDeals.filter((d) => !afterDealIds.has(d.id));

  return report;
}

function fmtGbp(value: number | null): string {
  if (value === null) return "—";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  return `${sign}£${abs.toFixed(2)}`;
}

function renderChangelogMarkdown(report: ChangeReport): string {
  const lines: string[] = [];
  lines.push(`# ${report.store} changelog — ${report.generatedAt}`);
  lines.push("");
  lines.push("## Summary");
  lines.push(`- ${report.itemsAdded.length} new item(s)`);
  lines.push(`- ${report.itemsRemoved.length} removed item(s)`);
  lines.push(`- ${report.optionsAdded.length} new option(s) on existing items`);
  lines.push(`- ${report.optionsRemoved.length} removed option(s)`);
  lines.push(`- ${report.priceChanges.length} significant price change(s)`);
  lines.push(`- ${report.dealsAdded.length} new deal(s)`);
  lines.push(`- ${report.dealsRemoved.length} removed deal(s)`);
  lines.push("");

  if (report.itemsAdded.length) {
    lines.push("## New items");
    for (const x of report.itemsAdded) {
      lines.push(`- \`${x.id}\` — ${x.canonicalName}`);
    }
    lines.push("");
  }

  if (report.itemsRemoved.length) {
    lines.push("## Removed items");
    for (const x of report.itemsRemoved) {
      lines.push(`- \`${x.id}\` — ${x.canonicalName}`);
    }
    lines.push("");
  }

  if (report.priceChanges.length) {
    // Sort by absolute % change descending so the most striking moves
    // are at the top of the report.
    const sorted = [...report.priceChanges].sort(
      (a, b) => Math.abs(b.deltaPct) - Math.abs(a.deltaPct)
    );
    lines.push("## Price changes (≥ 5% or ≥ £0.10)");
    for (const c of sorted) {
      const arrow = c.deltaAbs > 0 ? "↑" : "↓";
      lines.push(
        `- ${arrow} ${c.canonicalName} · ${c.optionName}: ` +
          `${fmtGbp(c.previousPrice)} → ${fmtGbp(c.newPrice)} ` +
          `(${c.deltaAbs > 0 ? "+" : ""}${fmtGbp(c.deltaAbs)} / ` +
          `${c.deltaPct > 0 ? "+" : ""}${c.deltaPct.toFixed(1)}%) ` +
          `[${c.itemId}]`
      );
    }
    lines.push("");
  }

  if (report.optionsAdded.length) {
    lines.push("## New options on existing items");
    for (const o of report.optionsAdded) {
      lines.push(
        `- ${o.canonicalName} · ${o.optionName} (${fmtGbp(o.price)}) [${o.itemId}]`
      );
    }
    lines.push("");
  }

  if (report.optionsRemoved.length) {
    lines.push("## Removed options");
    for (const o of report.optionsRemoved) {
      lines.push(
        `- ${o.canonicalName} · ${o.optionName} (was ${fmtGbp(o.previousPrice)}) [${o.itemId}]`
      );
    }
    lines.push("");
  }

  if (report.dealsAdded.length) {
    lines.push("## New deals");
    for (const d of report.dealsAdded) {
      lines.push(
        `- "${d.label}" (${d.rule.kind}, ${d.memberOptionIds.length} member${
          d.memberOptionIds.length === 1 ? "" : "s"
        }) [${d.id}]`
      );
    }
    lines.push("");
  }

  if (report.dealsRemoved.length) {
    lines.push("## Removed deals");
    for (const d of report.dealsRemoved) {
      lines.push(`- "${d.label}" [${d.id}]`);
    }
    lines.push("");
  }

  if (
    !report.itemsAdded.length &&
    !report.itemsRemoved.length &&
    !report.optionsAdded.length &&
    !report.optionsRemoved.length &&
    !report.priceChanges.length &&
    !report.dealsAdded.length &&
    !report.dealsRemoved.length
  ) {
    lines.push("_No notable changes this run._");
  }

  return lines.join("\n");
}

async function writeChangelogFiles(report: ChangeReport): Promise<string> {
  const changesDir = path.join(OUTPUT_DIR, "changes");
  await fs.mkdir(changesDir, { recursive: true });
  // Filename-safe ISO timestamp — colons get replaced with dashes so
  // the same file path works on Windows / S3 / wherever later.
  const stamp = report.generatedAt.replace(/[:.]/g, "-");
  const jsonPath = path.join(changesDir, `${stamp}-${report.store}.json`);
  const mdPath = path.join(changesDir, `${stamp}-${report.store}.md`);
  await fs.writeFile(jsonPath, JSON.stringify(report, null, 2));
  await fs.writeFile(mdPath, renderChangelogMarkdown(report));
  return mdPath;
}

async function loadExistingCatalog(store: StoreKey): Promise<CatalogItem[]> {
  const filePath = path.join(OUTPUT_DIR, `${store}-catalog.json`);
  if (!(await pathExists(filePath))) return [];
  try {
    const parsed = JSON.parse(await fs.readFile(filePath, "utf8"));
    return Array.isArray(parsed) ? (parsed as CatalogItem[]) : [];
  } catch {
    return [];
  }
}

async function loadExistingDeals(store: StoreKey): Promise<CatalogDeal[]> {
  const filePath = path.join(OUTPUT_DIR, `${store}-deals.json`);
  if (!(await pathExists(filePath))) return [];
  try {
    const parsed = JSON.parse(await fs.readFile(filePath, "utf8"));
    return Array.isArray(parsed) ? (parsed as CatalogDeal[]) : [];
  } catch {
    // A corrupted previous-run file shouldn't block this run — just
    // start fresh. The new write will overwrite the bad file.
    return [];
  }
}

async function flushDeals(store: StoreKey, deals: CatalogDeal[]) {
  const filePath = path.join(OUTPUT_DIR, `${store}-deals.json`);
  // Sort for stable diffs run-to-run: by id, then memberOptionIds
  // sorted inside each entry.
  const stable = deals
    .map((d) => ({
      ...d,
      memberOptionIds: [...d.memberOptionIds].sort(),
    }))
    .sort((a, b) => a.id.localeCompare(b.id));
  await fs.writeFile(filePath, JSON.stringify(stable, null, 2));
  return stable.length;
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
  const requestedCategory = getRequestedCategory();

  let itemsToProcess = allItems;

  // --category filters by category field first
  if (requestedCategory) {
    const validCategories = [...new Set(allItems.map((i) => i.category))];
    if (!validCategories.includes(requestedCategory)) {
      throw new Error(
        `Unknown category "${requestedCategory}" in --category.\n` +
          `Valid categories: ${validCategories.sort().join(", ")}.`
      );
    }
    itemsToProcess = allItems.filter((i) => i.category === requestedCategory);
    console.log(
      `Filtering to category "${requestedCategory}": ${itemsToProcess.length} item(s).`
    );
  }

  // --items further narrows (or selects from the full list if no --category)
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
    itemsToProcess = itemsToProcess.filter((i) =>
      wanted.has(i.slug || slugify(i.canonicalName))
    );
  }

  const discovered = await discoverManualItems(itemsToProcess, requestedStores);

  const processedByStore: ProcessedByStore = {};
  for (const store of requestedStores) {
    processedByStore[store] = await loadProcessedForStore(store);
  }
  const explicitItemFilter = requestedItemSlugs !== null || requestedCategory !== null;

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
    lidl: [],
    asda: [],
    morrisons: [],
  };

  // Per-store accumulator for cross-SKU deals found across the items
  // processed this run. Merged into the persisted output/<store>-
  // deals.json after each item via flushDeals(), same crash-safety
  // pattern as flushCatalog().
  const dealsByStore: Record<StoreKey, CatalogDeal[]> = {
    tesco: [],
    sainsburys: [],
    aldi: [],
    lidl: [],
    asda: [],
    morrisons: [],
  };

  // Seed dealsByStore with whatever's already on disk from prior runs,
  // so resuming an interrupted run preserves earlier deal extractions.
  for (const store of requestedStores) {
    dealsByStore[store] = await loadExistingDeals(store);
  }

  // Snapshot pre-run state for the changelog diff at the end of the
  // run. Must read BEFORE the per-item loop, because flushCatalog and
  // flushDeals overwrite output/<store>-{catalog,deals}.json as items
  // are processed.
  const beforeCatalogByStore: Record<StoreKey, CatalogItem[]> = {
    tesco: [],
    sainsburys: [],
    aldi: [],
    lidl: [],
    asda: [],
    morrisons: [],
  };
  const beforeDealsByStore: Record<StoreKey, CatalogDeal[]> = {
    tesco: [],
    sainsburys: [],
    aldi: [],
    lidl: [],
    asda: [],
    morrisons: [],
  };
  for (const store of requestedStores) {
    beforeCatalogByStore[store] = await loadExistingCatalog(store);
    beforeDealsByStore[store] = await loadExistingDeals(store);
  }

  console.log(`Using model: ${model}`);
  console.log(`Minimum confidence: ${minConfidence}`);
  console.log(
    qaPassEnabled
      ? `QA pass: ENABLED (model: ${qaModel}). One extra LLM call per image — verdicts in output/debug/<item>/<store>.qa.json.`
      : `QA pass: disabled. Set QA_PASS=on to enable a second LLM call that verifies extractions against the source images.`
  );
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

    const allOptions: LlmExtractedOption[] = [];
    const allDealsRaw: LlmExtractedDeal[] = [];
    // One QA entry per image, persisted to debug for human review even
    // when verdicts are all "ok" — turns the rejection rate into a
    // monitorable signal over time.
    const qaPerImage: Array<{
      image: string;
      verdicts: QaResult;
      // The claims as fed to the QA call, so the log is self-contained.
      submittedOptions: LlmExtractedOption[];
      submittedDeals: LlmExtractedDeal[];
    }> = [];

    for (const imagePath of imagePaths) {
      console.log(`  Image: ${imagePath}`);

      const extracted = await extractProducts(
        imagePath,
        item,
        store,
        config.brand
      );

      let keptOptions = extracted.options;
      let keptDeals = extracted.deals;

      if (qaPassEnabled && (extracted.options.length || extracted.deals.length)) {
        const qa = await validateExtraction(
          imagePath,
          item,
          store,
          extracted.options,
          extracted.deals
        );

        // Caller-side filtering: drop "wrong"; keep "ok" and "uncertain".
        // Defaults to keeping anything the QA model didn't return a
        // verdict for (e.g. index out of range) — being conservative
        // about removing extractions the QA layer didn't actively
        // contradict.
        const dropOption = new Set(
          qa.options.filter((v) => v.verdict === "wrong").map((v) => v.index)
        );
        const dropDeal = new Set(
          qa.deals.filter((v) => v.verdict === "wrong").map((v) => v.index)
        );

        if (dropOption.size || dropDeal.size) {
          console.log(
            `    QA: dropped ${dropOption.size} option(s) and ${dropDeal.size} deal(s)`
          );
        }

        keptOptions = extracted.options.filter((_, i) => !dropOption.has(i));
        keptDeals = extracted.deals.filter((_, i) => !dropDeal.has(i));

        qaPerImage.push({
          image: path.basename(imagePath),
          verdicts: qa,
          submittedOptions: extracted.options,
          submittedDeals: extracted.deals,
        });
      }

      allOptions.push(...keptOptions);
      allDealsRaw.push(...keptDeals);
    }

    // Persist the QA log for this item / store. Mirrors the .raw.json
    // pattern: one file per (item, store) capturing the per-image
    // verdicts so a developer can spot-check the QA model's decisions.
    if (qaPassEnabled && qaPerImage.length > 0) {
      await fs.writeFile(
        path.join(OUTPUT_DIR, "debug", itemSlug, `${store}.qa.json`),
        JSON.stringify(qaPerImage, null, 2)
      );
    }

    const cleanedOptions = allOptions
      .map((option) => cleanOption(option, store))
      .filter((option) => option.price !== null)
      .sort((a, b) => {
        const aPrice = a.loyaltyPrice ?? a.price ?? Number.MAX_SAFE_INTEGER;
        const bPrice = b.loyaltyPrice ?? b.price ?? Number.MAX_SAFE_INTEGER;
        return aPrice - bPrice;
      });

    const dedupedLlmOptions = Array.from(
      new Map(
        cleanedOptions.map((option) => [
          `${option.name}-${option.size}-${option.price}-${option.loyaltyPrice}`,
          option,
        ])
      ).values()
    );

    // Assign stable optionIds AFTER dedup so we don't waste id-generation
    // work on entries that will be discarded. If two surviving options
    // happen to derive the same id (same brand + name + size at different
    // prices), keep the first one — which, given the ascending price
    // sort above, is the cheaper variant.
    const dedupedOptions: CatalogOption[] = Array.from(
      new Map(
        dedupedLlmOptions.map((option) => {
          const optionId = makeOptionId(catalogId, option);
          return [optionId, { ...option, optionId }] as const;
        })
      ).values()
    );

    if (!dedupedOptions.length) {
      throw new Error(
        `No reliable options found for ${item.canonicalName} / ${store}. ` +
          `Check screenshots in manual-screenshots/${item.category}/${itemSlug}/${store}/ ` +
          `or review output/debug/${itemSlug}/${store}.raw.json`
      );
    }

    // Resolve LLM-reported memberOptionNames against the optionIds we
    // actually emitted. Drops member names that didn't survive cleaning/
    // dedup; drops the entire deal if fewer than 2 members remain. This
    // is the strict-acceptance gate — hallucinated deals can't slip
    // through because they'd reference names that aren't in our options
    // list. Within this loop the same deal may be observed under
    // multiple items (same banner photographed on multiple category
    // pages), so we merge memberOptionIds at the store level rather
    // than appending duplicates.
    const resolvedDeals = resolveDealMembers(store, allDealsRaw, dedupedOptions);
    for (const deal of resolvedDeals) {
      const existing = dealsByStore[store].find((d) => d.id === deal.id);
      if (existing) {
        existing.memberOptionIds = Array.from(
          new Set([...existing.memberOptionIds, ...deal.memberOptionIds])
        );
      } else {
        dealsByStore[store].push(deal);
      }
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

    // Flush catalog + deals + checkpoint after every successful item so
    // a crash mid-run never loses completed work.
    const total = await flushCatalog(store, [newItem]);
    const dealsCount = await flushDeals(store, dealsByStore[store]);
    await markProcessed(processedByStore, store, itemSlug);

    console.log(
      `  OK: ${best.name} | £${best.price}` +
        (best.loyaltyPrice ? ` | loyalty £${best.loyaltyPrice}` : "") +
        ` | options ${dedupedOptions.length} | catalog now ${total} items` +
        (dealsCount ? ` | ${dealsCount} deal(s) for ${store}` : "")
    );
  }

  for (const store of requestedStores) {
    console.log(
      `Done ${store}: ${newItemsByStore[store].length} new/updated this run`
    );
  }

  // Changelog diff. Read the AFTER state from disk (everything has been
  // flushed by per-item flushCatalog/flushDeals) and diff against the
  // BEFORE snapshot captured at the top of main(). One report file per
  // store touched this run; stores that weren't requested don't get a
  // changelog entry.
  for (const store of requestedStores) {
    const afterCatalog = await loadExistingCatalog(store);
    const afterDeals = await loadExistingDeals(store);
    const report = computeChangelog(
      store,
      beforeCatalogByStore[store],
      afterCatalog,
      beforeDealsByStore[store],
      afterDeals
    );
    const mdPath = await writeChangelogFiles(report);
    const counts = [
      report.itemsAdded.length && `${report.itemsAdded.length} new`,
      report.itemsRemoved.length && `${report.itemsRemoved.length} removed`,
      report.priceChanges.length && `${report.priceChanges.length} price change(s)`,
      report.dealsAdded.length && `${report.dealsAdded.length} new deal(s)`,
      report.dealsRemoved.length && `${report.dealsRemoved.length} removed deal(s)`,
    ]
      .filter(Boolean)
      .join(", ");
    console.log(
      counts
        ? `Changelog ${store}: ${counts} → ${mdPath}`
        : `Changelog ${store}: no notable changes → ${mdPath}`
    );
  }

  await mirrorCatalogsToDownstream(requestedStores);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});