import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { ParityReport, RecoveryReport } from "./entry.ts";
import { openWebGpuPage, swiftshader } from "./launch.ts";

const parityPath = fileURLToPath(new URL("../../../training/active/parity.json", import.meta.url));
const parity = JSON.parse(readFileSync(parityPath, "utf8")) as { entries: { seed: number }[] };
const seeds = parity.entries.map((e) => e.seed);

const { browser, page } = await openWebGpuPage("gpu-sprite parity");

const report = await page.evaluate((s) => (globalThis as unknown as { GpuSpriteTest: { runParity(seeds: number[]): Promise<ParityReport> } }).GpuSpriteTest.runParity(s), seeds);
const api = await page.evaluate((s) => (globalThis as unknown as { GpuSpriteTest: { runApi(seeds: number[]): Promise<{ backend: string; adapter?: string; identical: boolean; count: number }> } }).GpuSpriteTest.runApi(s), seeds.slice(0, 16));
// 64 seeds: at AUTO_WEBGPU_MIN_BATCH, so `auto` picks the GPU.
const recovery = await page.evaluate((s) => (globalThis as unknown as { GpuSpriteTest: { runRecovery(seeds: number[]): Promise<RecoveryReport> } }).GpuSpriteTest.runRecovery(s), seeds);
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
// The call right after the loss (index 1) may run on either backend; every other call must be on the GPU.
const [autoBefore, , autoAfter, explicitBefore, explicitAfter] = recovery.backends;
if ([autoBefore, autoAfter, explicitBefore, explicitAfter].some((b) => b !== "webgpu")) failures.push(`device-loss recovery backends ${recovery.backends.join(",")}`);
if (!recovery.identical) failures.push("device-loss recovery sprites differ from the cpu");

console.log(`adapter: ${report.adapter}${swiftshader ? " (swiftshader software adapter)" : ""}`);
console.log(`seeds: ${seeds.length}, pixels: ${report.total}, max logit diff: ${report.maxDiff.toExponential(2)}, low-margin pixels: ${report.lowMargin}, batch-vs-single diff: ${report.batchVsSingleMaxDiff.toExponential(2)}, non-finite: ${report.nonFinite}`);
for (const p of report.lowMarginPixels) {
  console.log(`seed ${p.seed} pixel ${p.pixel}: cpu ${p.cpu.toExponential(6)} gpu ${p.gpu.toExponential(6)}`);
}
console.log(`api: backend=${api.backend} adapter=${api.adapter ?? "-"} identical=${api.identical} count=${api.count}`);
console.log(`device-loss recovery: backends=${recovery.backends.join(",")} identical=${recovery.identical} devices=${recovery.devices}`);
if (failures.length) {
  for (const f of failures) console.error(`FAIL ${f}`);
  process.exit(1);
}
console.log("PASS webgpu parity");
