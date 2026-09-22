import { AUTO_WEBGPU_MIN_BATCH } from "../../src/index.ts";
import type { BenchReport } from "./entry.ts";
import { openWebGpuPage, swiftshader } from "./launch.ts";

// Measures where WebGPU starts beating the CPU for `generateMany`, to check the `auto` threshold.
// Not part of `pnpm check`: timings depend on the machine. Run after `pnpm build:core`.
const sizes = [1, 4, 16, 64, 256, 1024, 4096];
const reps = 15;

const { browser, page } = await openWebGpuPage("gpu-sprite bench");
const report = await page.evaluate(
  ([s, r]) => (globalThis as unknown as { GpuSpriteTest: { runBench(sizes: number[], reps: number): Promise<BenchReport> } }).GpuSpriteTest.runBench(s, r),
  [sizes, reps] as [number[], number],
);
await browser.close();

console.log(`adapter: ${report.adapter}${swiftshader ? " (swiftshader software adapter)" : ""}`);
console.log(`generateMany, median of ${reps} timed calls after one warm-up call per backend and batch`);
console.log("batch     cpu ms  webgpu ms  cpu/webgpu");
for (const row of report.rows) {
  console.log(`${String(row.batch).padStart(5)} ${row.cpuMs.toFixed(2).padStart(10)} ${row.gpuMs.toFixed(2).padStart(10)} ${(row.cpuMs / row.gpuMs).toFixed(2).padStart(11)}`);
}
const crossover = report.rows.find((row) => row.gpuMs < row.cpuMs);
console.log(`crossover: ${crossover ? crossover.batch : "none"} (smallest measured batch where webgpu is faster; AUTO_WEBGPU_MIN_BATCH = ${AUTO_WEBGPU_MIN_BATCH})`);
