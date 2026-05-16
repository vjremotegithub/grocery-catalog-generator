import type { CatalogOption, CommonItem } from "./types";

const BLOCKED_WORDS = [
  "recipe",
  "ready meal",
  "pet",
  "dog",
  "cat",
  "baby",
  "chocolate",
  "dessert",
  "cake",
  "ice cream",
  "sandwich",
  "meal deal",
  "magazine"
];

const IMPORTANT_CANONICAL_WORDS = new Set([
  "milk",
  "eggs",
  "bread",
  "chicken",
  "beef",
  "lamb",
  "rice",
  "pasta",
  "coffee",
  "tea",
  "beans",
  "tomatoes",
  "butter",
  "cheese",
  "yogurt",
  "halloumi",
  "apples",
  "bananas",
  "blueberries",
  "broccoli",
  "potatoes",
  "carrots",
  "onions",
  "shampoo"
]);

export function isComparableProduct(option: CatalogOption, item: CommonItem) {
  const name = option.name.toLowerCase();
  const canonical = item.canonicalName.toLowerCase();

  if (!option.price && !option.loyaltyPrice && !option.buy2Price) return false;
  if (BLOCKED_WORDS.some((word) => name.includes(word))) return false;

  const canonicalWords = canonical.split(/\s+/).filter((word) => word.length > 2);
  const importantWords = canonicalWords.filter((word) => IMPORTANT_CANONICAL_WORDS.has(word));
  const wordsToMatch = importantWords.length ? importantWords : canonicalWords;

  return wordsToMatch.some((word) => name.includes(word));
}

export function rankOptions(options: CatalogOption[], item: CommonItem) {
  const preferredSizes = (item.preferredSizes ?? []).map((size) => size.toLowerCase());

  return [...options].sort((a, b) => {
    const aName = `${a.name} ${a.size ?? ""}`.toLowerCase();
    const bName = `${b.name} ${b.size ?? ""}`.toLowerCase();

    const aPreferred = preferredSizes.some((size) => aName.includes(size)) ? 1 : 0;
    const bPreferred = preferredSizes.some((size) => bName.includes(size)) ? 1 : 0;

    if (aPreferred !== bPreferred) return bPreferred - aPreferred;

    const aPrice = a.loyaltyPrice ?? a.price ?? Number.POSITIVE_INFINITY;
    const bPrice = b.loyaltyPrice ?? b.price ?? Number.POSITIVE_INFINITY;

    return aPrice - bPrice;
  });
}
