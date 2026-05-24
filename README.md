# Grocery Catalog Generator

Generates structured grocery catalog JSON files from manual product screenshots using OpenAI vision. Supports Tesco, Sainsbury's, Aldi, Asda, Lidl, and Morrisons.

---

## Quick start

```bash
npm run generate                  # all stores, all items
npm run generate:tesco            # one store
npm run generate:dairy            # one category across all stores
npm run generate -- --items whole-milk --stores sainsburys   # one item, one store
```

---

## Project layout

```
grocery-catalog-generator/
├── .env / .env.example / .gitignore / tsconfig.json
├── package.json
├── data/
│   └── store-search-urls.json           store brand + search URL config
├── manual-screenshots/
│   ├── items.json                       canonical item list (slug, canonicalName, category, keywords)
│   └── <category>/<slug>/<store>/*.png  screenshots organised by category → item → store
├── scripts/
│   └── generate-from-manual-screenshots.ts
└── output/
    ├── tesco-catalog.json               merged catalog, written incrementally per item
    ├── sainsburys-catalog.json
    ├── aldi-catalog.json
    ├── asda-catalog.json
    ├── lidl-catalog.json
    ├── morrisons-catalog.json
    ├── tesco.processed.json             checkpoint — delete to force a full re-run
    ├── sainsburys.processed.json
    ├── aldi.processed.json
    ├── asda.processed.json
    ├── lidl.processed.json
    ├── morrisons.processed.json
    ├── crops/                           regenerated each run
    └── debug/                           raw model output per item, regenerated each run
```

---

## Screenshot folder structure

Screenshots must live at:

```
manual-screenshots/<category>/<slug>/<store>/<image>.{png,jpg,jpeg,webp}
```

For example:
```
manual-screenshots/dairy/cheese/tesco/cheese1.png
manual-screenshots/dairy/cheese/sainsburys/cheese1.png
```

Categories match the `category` field in `items.json`: `milk`, `eggs`, `dairy`, `bakery`, `breakfast`, `cupboard`, `vegetables`, `fruit`, `meat`, `seafood`, `frozen`, `drinks`, `alcohol`, `snacks`, `household`, `toiletries`, `beauty`, `pet`.

If any item/store combination is missing (folder absent, store subfolder absent, or store folder empty), the script **stops before making any OpenAI calls** and prints exactly which paths are missing.

---

## npm scripts

### By store

```bash
npm run generate              # all stores
npm run generate:tesco
npm run generate:sainsburys
npm run generate:aldi
npm run generate:asda
npm run generate:lidl
npm run generate:morrisons
```

### By category

```bash
npm run generate:milk
npm run generate:eggs
npm run generate:dairy
npm run generate:bakery
npm run generate:breakfast
npm run generate:cupboard
npm run generate:vegetables
npm run generate:fruit
npm run generate:meat
npm run generate:seafood
npm run generate:frozen
npm run generate:drinks
npm run generate:alcohol
npm run generate:snacks
npm run generate:household
npm run generate:toiletries
npm run generate:beauty
npm run generate:pet
```

### Other

```bash
npm run check                 # tsc --noEmit (type-check only)
npm run enrich-tags           # add dietary/allergen tags to existing catalog entries
```

---

## Flags reference

Pass extra flags via `--` after the npm script:

```bash
npm run generate:tesco -- --items whole-milk,semi-skimmed-milk
npm run generate:dairy -- --stores tesco,sainsburys
npm run generate -- --limit 5
```

| Flag | Example | Description |
|------|---------|-------------|
| `--stores` | `--stores tesco,aldi` | Which store(s) to process. Defaults to all six. |
| `--items` | `--items whole-milk,cheese` | Process only these slugs (from `items.json`). Always reprocesses, bypasses checkpoint. |
| `--category` | `--category dairy` | Process only items in this category. Always reprocesses, bypasses checkpoint. |
| `--limit` | `--limit 5` | Cap the number of item/store combinations processed this run. |

`--category` and `--items` can be combined — category sets the pool, `--items` narrows within it:

```bash
npm run generate -- --category milk --items whole-milk --stores tesco
```

---

## Checkpoints and resumability

After each successful item the script:

1. Writes the new entry into `output/<store>-catalog.json` (merged by ID with anything already there).
2. Records the slug in `output/<store>.processed.json`.

If the script crashes mid-run, re-run the same command — it skips already-processed items and continues from where it stopped.

```bash
# Force-reprocess specific items (ignores the checkpoint):
npm run generate:aldi -- --items shampoo,coffee

# Reset one store entirely:
rm output/aldi.processed.json && npm run generate:aldi

# Reset everything:
rm output/*.processed.json && npm run generate
```

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required. |
| `OPENAI_VISION_MODEL` | `gpt-4o` | Vision model used for product extraction. |
| `MIN_CONFIDENCE` | `0.65` | Drop extracted products below this confidence score. |
