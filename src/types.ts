export type StoreKey = "tesco" | "sainsburys" | "aldi";

export type CommonItem = {
  canonicalName: string;
  category: string;
  keywords: string[];
  preferredSizes?: string[];
};

export type StoreConfig = {
  brand: string;
  searchUrl: string;
  notes?: string;
};

export type CatalogOption = {
  name: string;
  brand: string;
  size: string | null;
  unit: string | null;
  price: number | null;
  loyaltyPrice: number | null;
  buy2Price: number | null;
};

export type CatalogItem = {
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
  sourceUrl: string;
  extractionMethod: "screenshot-openai" | "text-openai";
  needsReview: boolean;
};
