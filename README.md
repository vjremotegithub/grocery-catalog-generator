# ALL STORE

npx tsx scripts/generate-from-manual-screenshots.ts --stores tesco,sainsburys,aldi

# SPECIFIC STORES

npx tsx scripts/generate-from-manual-screenshots.ts --stores aldi

npx tsx scripts/generate-from-manual-screenshots.ts --stores tesco

npx tsx scripts/generate-from-manual-screenshots.ts --stores sainsburys

# SPECIFIC STORE ITEM

npx tsx scripts/generate-from-manual-screenshots.ts --stores tesco,sainsburys,aldi


# Run everything (initial full run):

npx tsx scripts/generate-from-manual-screenshots.ts --stores aldi

# Re-run just one item after a failure:
npx tsx scripts/generate-from-manual-screenshots.ts --stores aldi --items shampoo

# Re-run several at once:

npx tsx scripts/generate-from-manual-screenshots.ts --stores aldi --items shampoo,coffee,toilet-paper

# Re-run one item across multiple stores:

npx tsx scripts/generate-from-manual-screenshots.ts --stores aldi,tesco --items shampoo

---

## Project layout

```
grocery-catalog-generator/
├── .env / .env.example / .gitignore / README.md / tsconfig.json
├── package.json
├── package-lock.json
├── data/
│   └── store-search-urls.json           (store brand/search URL config — used by the script)
├── manual-screenshots/
│   ├── items.json                       (canonical item list — slug, canonicalName, category, keywords)
│   └── <56 item folders>/<store>/<png/jpg>
├── scripts/
│   └── generate-from-manual-screenshots.ts
└── output/
    ├── aldi-catalog.json                (merged catalog, written incrementally per item)
    ├── sainsburys-catalog.json
    ├── tesco-catalog.json
    ├── aldi.processed.json              (checkpoint of items already done — delete to redo)
    ├── sainsburys.processed.json
    ├── tesco.processed.json
    ├── crops/                           (regenerated per run)
    └── debug/                           (raw model output per item, regenerated per run)
```

## npm commands

```
npm run generate              # all stores
npm run generate:aldi
npm run generate:sainsburys
npm run generate:tesco
npm run check                 # tsc --noEmit (type-check only)
```

Pass extra flags through npm with `--`:

```
npm run generate:aldi -- --items shampoo,coffee
npm run generate:aldi -- --limit 5
```

## How items are discovered

The script reads `manual-screenshots/items.json` and, for each entry, looks for screenshots at:

```
manual-screenshots/<slug>/<store>/<image>.{png,jpg,jpeg,webp}
```

If any item/store combination is missing screenshots (folder absent, store subfolder absent, or store folder empty), the script **stops with an error before making any OpenAI calls** and prints exactly which paths are missing. Add the screenshots (or remove the item from `items.json`) and re-run.

## Checkpoints and resumability

After every successful item, the script:

1. Writes the new entry into `output/<store>-catalog.json` (merge-by-id with anything already there).
2. Records the slug in `output/<store>.processed.json`.

If the script crashes mid-run, both files reflect everything completed so far. Re-run the same command and it picks up from the next un-processed item.

Reset behaviour:

```
# Force-reprocess specific items (ignores the checkpoint):
npm run generate:aldi -- --items shampoo,coffee

# Reset one store entirely:
rm output/aldi.processed.json
npm run generate:aldi

# Reset everything:
rm output/*.processed.json
npm run generate
```

## Flags reference

| Flag | Description |
|------|-------------|
| `--stores aldi,tesco,sainsburys` | Which store(s) to process. Defaults to all three. |
| `--items shampoo,coffee` | Only process these item slugs (slugs come from `manual-screenshots/items.json`). Bypasses the checkpoint. |
| `--limit 5` | Cap the number of item/store combinations processed this run. |

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required. Real key, not the placeholder. |
| `OPENAI_VISION_MODEL` | `gpt-4o` | Vision model used for product extraction. |
| `MIN_CONFIDENCE` | `0.65` | Drop extracted products below this confidence score. |
