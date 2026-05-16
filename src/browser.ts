import fs from "fs/promises";
import path from "path";
import { chromium } from "playwright";
import { ensureDir, slugify } from "./utils";
import type { CommonItem, StoreKey } from "./types";

export type PageCapture = {
  screenshotPath: string;
  title: string;
  visibleText: string;
  blocked: boolean;
};

export async function captureSearchPage(params: {
  url: string;
  store: StoreKey;
  item: CommonItem;
  headless: boolean;
  pageWaitMs: number;
}): Promise<PageCapture> {
  const browser = await chromium.launch({ headless: params.headless });
  const context = await browser.newContext({
    viewport: { width: 1365, height: 1700 },
    userAgent:
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    locale: "en-GB"
  });

  const page = await context.newPage();
  const screenshotDir = path.join(process.cwd(), "screenshots", params.store);
  await ensureDir(screenshotDir);

  const screenshotPath = path.join(
    screenshotDir,
    `${slugify(params.item.canonicalName)}.jpg`
  );

  try {
    await page.goto(params.url, { waitUntil: "domcontentloaded", timeout: 60000 });
    await page.waitForTimeout(params.pageWaitMs);

    // Best-effort cookie accept. Safe if selectors do not exist.
    const cookieButtons = [
      "button:has-text('Accept all')",
      "button:has-text('Accept All')",
      "button:has-text('Accept')",
      "button:has-text('I accept')",
      "button:has-text('Allow all')"
    ];

    for (const selector of cookieButtons) {
      const button = page.locator(selector).first();
      if (await button.isVisible().catch(() => false)) {
        await button.click().catch(() => undefined);
        await page.waitForTimeout(1000);
        break;
      }
    }

    await page.screenshot({ path: screenshotPath, type: "jpeg", quality: 72, fullPage: false });
    const title = await page.title().catch(() => "");
    const visibleText = await page.locator("body").innerText().catch(() => "");
    const blocked = /access denied|forbidden|robot|captcha|unusual traffic/i.test(`${title}\n${visibleText}`);

    return {
      screenshotPath,
      title,
      visibleText: visibleText.slice(0, 15000),
      blocked
    };
  } finally {
    await context.close().catch(() => undefined);
    await browser.close().catch(() => undefined);
  }
}
