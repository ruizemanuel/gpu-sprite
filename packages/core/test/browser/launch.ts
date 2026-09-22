import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { chromium, type Browser, type Page } from "playwright";

const bundlePath = fileURLToPath(new URL("../../dist/gpu-sprite.test.iife.js", import.meta.url));

/** True when the script was run with `--swiftshader`: Chromium then uses its software WebGPU adapter. */
export const swiftshader = process.argv.includes("--swiftshader");
const args = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", ...(swiftshader ? ["--use-webgpu-adapter=swiftshader"] : [])];
// navigator.gpu only exists in secure contexts and Chromium does not treat about:blank as one,
// so an empty page is served from http://localhost (a secure origin) by request interception.
const pageUrl = "http://localhost/";

export interface Session {
  browser: Browser;
  page: Page;
}

async function launch(headless: boolean, bundle: string, title: string): Promise<Session | undefined> {
  const browser = await chromium.launch({ headless, channel: "chromium", args });
  const page = await browser.newPage();
  await page.route(pageUrl, (route) => route.fulfill({ contentType: "text/html", body: `<!doctype html><title>${title}</title>` }));
  await page.goto(pageUrl);
  await page.addScriptTag({ content: bundle });
  const ready = await page.evaluate(async () => {
    const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    return Boolean(window.isSecureContext && gpu && (await gpu.requestAdapter()));
  });
  if (ready) return { browser, page };
  await browser.close();
  return undefined;
}

/**
 * Open Chromium with WebGPU enabled on a secure page with the test bundle loaded
 * (`globalThis.GpuSpriteTest`). Tries headless first, then headed. Exits the process
 * with an error when neither exposes a WebGPU adapter.
 */
export async function openWebGpuPage(title: string): Promise<Session> {
  const bundle = readFileSync(bundlePath, "utf8");
  const session = (await launch(true, bundle, title)) ?? (await launch(false, bundle, title));
  if (!session) {
    console.error("No WebGPU adapter in headless or headed Chromium. Re-run with --swiftshader to use the software adapter (reported as such).");
    process.exit(1);
  }
  session.page.on("console", (m) => console.log(`[browser] ${m.text()}`));
  session.page.on("pageerror", (e) => console.error(`[browser error] ${e.message}`));
  return session;
}
