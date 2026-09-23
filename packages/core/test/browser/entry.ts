import { defineGenerator, LATENT_DIM, type GenerateResult, type Sprite } from "../../src/index.ts";
import { CpuModel } from "../../src/model/cpu.ts";
import { decodeModel } from "../../src/model/decode.ts";
import { hasWebGpu, WebGpuModel } from "../../src/model/webgpu.ts";
import { model } from "../../src/model/weights.ts";
import { seedToLatent } from "../../src/seed.ts";
import { SPRITE_PIXELS } from "../../src/sprite.ts";

/** A pixel whose CPU logit is within the low-margin band (|logit| <= 1e-3). */
export interface LowMarginPixel {
  seed: number;
  pixel: number;
  cpu: number;
  gpu: number;
}

export interface ParityReport {
  adapter: string;
  total: number;
  maxDiff: number;
  mismatches: number[];
  lowMargin: number;
  lowMarginPixels: LowMarginPixel[];
  batchVsSingleMaxDiff: number;
  /** Non-finite GPU logits across the batched and the single-seed outputs. */
  nonFinite: number;
}

function latents(seeds: number[]): Float32Array {
  const z = new Float32Array(seeds.length * LATENT_DIM);
  seeds.forEach((s, i) => z.set(seedToLatent(s), i * LATENT_DIM));
  return z;
}

export async function runParity(seeds: number[]): Promise<ParityReport> {
  const layers = decodeModel(model);
  const cpu = new CpuModel(layers);
  const gpu = await WebGpuModel.create(layers);
  const z = latents(seeds);
  const a = cpu.forward(z, seeds.length);
  const b = await gpu.forward(z, seeds.length);
  const single = await gpu.forward(z.subarray(0, LATENT_DIM), 1);
  let maxDiff = 0;
  let batchVsSingleMaxDiff = 0;
  let nonFinite = 0;
  const lowMarginPixels: LowMarginPixel[] = [];
  const mismatches: number[] = [];
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(b[i])) nonFinite++;
    maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
    if (Math.abs(a[i]) <= 1e-3) {
      lowMarginPixels.push({ seed: seeds[Math.floor(i / SPRITE_PIXELS)], pixel: i % SPRITE_PIXELS, cpu: a[i], gpu: b[i] });
    } else if (a[i] > 0 !== b[i] > 0) mismatches.push(i);
  }
  for (let i = 0; i < SPRITE_PIXELS; i++) {
    if (!Number.isFinite(single[i])) nonFinite++;
    batchVsSingleMaxDiff = Math.max(batchVsSingleMaxDiff, Math.abs(single[i] - b[i]));
  }
  gpu.dispose();
  return {
    adapter: gpu.adapterName,
    total: a.length,
    maxDiff,
    mismatches,
    lowMargin: lowMarginPixels.length,
    lowMarginPixels,
    batchVsSingleMaxDiff,
    nonFinite,
  };
}

function sameSprites(a: Sprite[], b: Sprite[]): boolean {
  return a.length === b.length && a.every((s, i) => s.pixels.every((p, j) => p === b[i].pixels[j]));
}

export async function runApi(seeds: number[]): Promise<{ backend: string; adapter?: string; identical: boolean; count: number }> {
  const gpuGen = defineGenerator({ backend: "webgpu" });
  const cpuGen = defineGenerator({ backend: "cpu" });
  const g = await gpuGen.generateMany(seeds);
  const c = await cpuGen.generateMany(seeds);
  const identical = sameSprites(g.sprites, c.sprites);
  gpuGen.dispose();
  cpuGen.dispose();
  return { backend: g.backend, adapter: g.adapter, identical, count: g.sprites.length };
}

export interface RecoveryReport {
  /** Backend of each call, in order: auto, auto right after a device loss, auto, webgpu, webgpu after a device loss. */
  backends: string[];
  /** Every call's sprites equal the CPU's. */
  identical: boolean;
  devices: number;
}

/**
 * Device loss on real hardware, simulated with `destroy()` on the device a generator holds:
 * `auto` must not reject and must return to the GPU on a new device, and explicit `webgpu`
 * must work again after the loss. Devices are observed by wrapping `GPUAdapter.requestDevice`.
 */
export async function runRecovery(seeds: number[]): Promise<RecoveryReport> {
  const devices: GPUDevice[] = [];
  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (this: GPUAdapter, descriptor?: GPUDeviceDescriptor) {
    const device = await requestDevice.call(this, descriptor);
    devices.push(device);
    return device;
  };
  const cpuGen = defineGenerator({ backend: "cpu" });
  const autoGen = defineGenerator({ backend: "auto" });
  const gpuGen = defineGenerator({ backend: "webgpu" });
  const loseNewestDevice = async () => {
    const device = devices[devices.length - 1];
    device.destroy();
    await device.lost;
    await new Promise((resolve) => setTimeout(resolve, 0));
  };
  try {
    const expected = await cpuGen.generateMany(seeds);
    const results: GenerateResult[] = [];
    results.push(await autoGen.generateMany(seeds));
    devices[devices.length - 1].destroy();
    results.push(await autoGen.generateMany(seeds));
    await new Promise((resolve) => setTimeout(resolve, 0));
    results.push(await autoGen.generateMany(seeds));
    results.push(await gpuGen.generateMany(seeds));
    await loseNewestDevice();
    results.push(await gpuGen.generateMany(seeds));
    return {
      backends: results.map((r) => r.backend),
      identical: results.every((r) => sameSprites(r.sprites, expected.sprites)),
      devices: devices.length,
    };
  } finally {
    GPUAdapter.prototype.requestDevice = requestDevice;
    cpuGen.dispose();
    autoGen.dispose();
    gpuGen.dispose();
  }
}

/** Median `generateMany` wall time per backend for one batch size, in milliseconds. */
export interface BenchRow {
  batch: number;
  cpuMs: number;
  gpuMs: number;
}

export interface BenchReport {
  adapter: string;
  rows: BenchRow[];
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** One untimed warm-up call, then the median of `reps` timed calls. */
async function medianMs(call: () => Promise<unknown>, reps: number): Promise<number> {
  await call();
  const times: number[] = [];
  for (let r = 0; r < reps; r++) {
    const t0 = performance.now();
    await call();
    times.push(performance.now() - t0);
  }
  return median(times);
}

/**
 * Time the public `generateMany` on a CPU generator and on a WebGPU generator for each batch
 * size. Both include seed → latent and the CPU decode of logits into sprites, as `auto` would.
 */
export async function runBench(sizes: number[], reps: number): Promise<BenchReport> {
  const cpuGen = defineGenerator({ backend: "cpu" });
  const gpuGen = defineGenerator({ backend: "webgpu" });
  let adapter = "unknown adapter";
  const rows: BenchRow[] = [];
  for (const batch of sizes) {
    const seeds = Array.from({ length: batch }, (_, i) => i);
    const cpuMs = await medianMs(() => cpuGen.generateMany(seeds), reps);
    const gpuMs = await medianMs(async () => {
      const result = await gpuGen.generateMany(seeds);
      adapter = result.adapter ?? adapter;
    }, reps);
    rows.push({ batch, cpuMs, gpuMs });
  }
  gpuGen.dispose();
  cpuGen.dispose();
  return { adapter, rows };
}

export { hasWebGpu };
