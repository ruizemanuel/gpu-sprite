import { defineGenerator, LATENT_DIM } from "../../src/index.ts";
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

export async function runApi(seeds: number[]): Promise<{ backend: string; adapter?: string; identical: boolean; count: number }> {
  const gpuGen = defineGenerator({ backend: "webgpu" });
  const cpuGen = defineGenerator({ backend: "cpu" });
  const g = await gpuGen.generateMany(seeds);
  const c = await cpuGen.generateMany(seeds);
  let identical = g.sprites.length === c.sprites.length;
  for (let i = 0; identical && i < g.sprites.length; i++) {
    const p = g.sprites[i].pixels;
    const q = c.sprites[i].pixels;
    for (let j = 0; j < p.length; j++) {
      if (p[j] !== q[j]) {
        identical = false;
        break;
      }
    }
  }
  gpuGen.dispose();
  cpuGen.dispose();
  return { backend: g.backend, adapter: g.adapter, identical, count: g.sprites.length };
}

export { hasWebGpu };
