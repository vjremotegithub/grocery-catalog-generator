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