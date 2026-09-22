import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { chromium, type Browser, type Page } from "playwright";
import type { ParityReport } from "./entry.ts";

const bundlePath = fileURLToPath(new URL("../../dist/gpu-sprite.test.iife.js", import.meta.url));
const parityPath = fileURLToPath(new URL("../../../training/active/parity.json", import.meta.url));
const bundle = readFileSync(bundlePath, "utf8");
const parity = JSON.parse(readFileSync(parityPath, "utf8")) as { entries: { seed: number }[] };
const seeds = parity.entries.map((e) => e.seed);
const swiftshader = process.argv.includes("--swiftshader");
const args = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", ...(swiftshader ? ["--use-webgpu-adapter=swiftshader"] : [])];
// navigator.gpu only exists in secure contexts and Chromium does not treat about:blank as one,
// so an empty page is served from http://localhost (a secure origin) by request interception.
const pageUrl = "http://localhost/";

async function launch(headless: boolean): Promise<{ browser: Browser; page: Page } | undefined> {
  const browser = await chromium.launch({ headless, channel: "chromium", args });
  const page = await browser.newPage();
  await page.route(pageUrl, (route) => route.fulfill({ contentType: "text/html", body: "<!doctype html><title>gpu-sprite parity</title>" }));
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

const session = (await launch(true)) ?? (await launch(false));
if (!session) {
  console.error("No WebGPU adapter in headless or headed Chromium. Re-run with --swiftshader to use the software adapter (reported as such).");
  process.exit(1);
}
const { browser, page } = session;
page.on("console", (m) => console.log(`[browser] ${m.text()}`));
page.on("pageerror", (e) => console.error(`[browser error] ${e.message}`));

const report = await page.evaluate((s) => (globalThis as unknown as { GpuSpriteTest: { runParity(seeds: number[]): Promise<ParityReport> } }).GpuSpriteTest.runParity(s), seeds);
const api = await page.evaluate((s) => (globalThis as unknown as { GpuSpriteTest: { runApi(seeds: number[]): Promise<{ backend: string; adapter?: string; identical: boolean; count: number }> } }).GpuSpriteTest.runApi(s), seeds.slice(0, 16));
await browser.close();

const failures: string[] = [];
// Threshold checks are written as !(x < limit) so a NaN diff fails instead of passing silently.
if (report.nonFinite > 0) failures.push(`${report.nonFinite} non-finite GPU logits`);
if (!(report.maxDiff < 1e-4)) failures.push(`max logit diff ${report.maxDiff} >= 1e-4`);
if (report.mismatches.length > 0) failures.push(`${report.mismatches.length} bit mismatches above margin, first at ${report.mismatches.slice(0, 5).join(",")}`);
if (report.lowMargin > report.total * 0.001) failures.push(`low-margin pixels ${report.lowMargin} > 0.1% of ${report.total}`);
if (!(report.batchVsSingleMaxDiff < 1e-4)) failures.push(`batched vs single diff ${report.batchVsSingleMaxDiff}`);
if (api.backend !== "webgpu") failures.push(`generateMany used ${api.backend}, expected webgpu`);
if (!api.identical) failures.push("generateMany sprites differ between webgpu and cpu");

console.log(`adapter: ${report.adapter}${swiftshader ? " (swiftshader software adapter)" : ""}`);
console.log(`seeds: ${seeds.length}, pixels: ${report.total}, max logit diff: ${report.maxDiff.toExponential(2)}, low-margin pixels: ${report.lowMargin}, batch-vs-single diff: ${report.batchVsSingleMaxDiff.toExponential(2)}, non-finite: ${report.nonFinite}`);
for (const p of report.lowMarginPixels) {
  console.log(`seed ${p.seed} pixel ${p.pixel}: cpu ${p.cpu.toExponential(6)} gpu ${p.gpu.toExponential(6)}`);
}
console.log(`api: backend=${api.backend} adapter=${api.adapter ?? "-"} identical=${api.identical} count=${api.count}`);
if (failures.length) {
  for (const f of failures) console.error(`FAIL ${f}`);
  process.exit(1);
}
console.log("PASS webgpu parity");
