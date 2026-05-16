import fs from "fs/promises";
import OpenAI from "openai";
import { z } from "zod";
import type { CatalogOption, CommonItem, StoreKey } from "./types";

const OptionSchema = z.object({
  name: z.string(),
  brand: z.string(),
  size: z.string().nullable(),
  unit: z.string().nullable(),
  price: z.number().nullable(),
  loyaltyPrice: z.number().nullable(),
  buy2Price: z.number().nullable()
});

const ExtractSchema = z.object({
  options: z.array(OptionSchema)
});

function getOutputText(response: any): string {
  if (typeof response.output_text === "string") return response.output_text;
  const parts = response.output?.flatMap((outputItem: any) => outputItem.content ?? []) ?? [];
  const textPart = parts.find((part: any) => typeof part.text === "string");
  return textPart?.text ?? "";
}

export async function extractComparableProducts(params: {
  client: OpenAI;
  model: string;
  screenshotPath: string;
  visibleText: string;
  store: StoreKey;
  brand: string;
  item: CommonItem;
  maxOptions: number;
}): Promise<CatalogOption[]> {
  const image = await fs.readFile(params.screenshotPath);
  const imageUrl = `data:image/jpeg;base64,${image.toString("base64")}`;

  const response = await params.client.responses.create({
    model: params.model,
    input: [
      {
        role: "system",
        content:
          "You extract UK supermarket product search results from screenshots and visible page text. Return JSON only. Include only products comparable to the requested grocery item. Exclude ads, recipes, unrelated products, sponsored irrelevant items, and bundles that are not directly comparable."
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: `Store: ${params.store}\nBrand fallback: ${params.brand}\nCanonical item: ${params.item.canonicalName}\nCategory: ${params.item.category}\nKeywords: ${params.item.keywords.join(", ")}\nPreferred sizes: ${(params.item.preferredSizes ?? []).join(", ") || "none"}\n\nVisible page text:\n${params.visibleText}\n\nExtract up to ${params.maxOptions} comparable options.\nUse numeric prices only, for example 1.25 not £1.25.\nSet loyaltyPrice if a Clubcard/Nectar/member price exists.\nSet buy2Price for offers like 2 for £3.\nUse null where unknown.`
          },
          {
            type: "input_image",
            image_url: imageUrl,
            detail: "low"
          }
        ]
      }
    ],
    text: {
      format: {
        type: "json_schema",
        name: "catalog_extract",
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
                  buy2Price: { type: ["number", "null"] }
                },
                required: ["name", "brand", "size", "unit", "price", "loyaltyPrice", "buy2Price"]
              }
            }
          },
          required: ["options"]
        }
      }
    }
  });

  const parsed = ExtractSchema.parse(JSON.parse(getOutputText(response)));
  return parsed.options;
}
