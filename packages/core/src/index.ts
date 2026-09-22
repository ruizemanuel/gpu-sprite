import { CpuModel } from "./model/cpu.ts";
import { decodeModel } from "./model/decode.ts";
import type { DecodedLayer, ModelSpec } from "./model/types.ts";
import { hasWebGpu, WebGpuModel } from "./model/webgpu.ts";
import { model } from "./model/weights.ts";
import { LATENT_DIM, seedToLatent, toUint32Seed } from "./seed.ts";
import { logitsToSprite, SPRITE_PIXELS, type Sprite } from "./sprite.ts";

export const VERSION = "0.0.1";
export const MAX_BATCH = 4096;
/** `auto` uses WebGPU only for batches at least this large; smaller ones are faster on the CPU. */
export const AUTO_WEBGPU_MIN_BATCH = 64;
export { LATENT_DIM, seedToLatent };
export type { Sprite, ModelSpec };

export type Backend = "cpu" | "webgpu" | "auto";

export interface GenerateOptions {
  backend?: Backend;
}

export interface GenerateResult {
  sprites: Sprite[];
  backend: "cpu" | "webgpu";
  /** WebGPU adapter description when the GPU ran the batch. */
  adapter?: string;
}

let sharedLayers: DecodedLayer[] | undefined;
function layers(): DecodedLayer[] {
  return (sharedLayers ??= decodeModel(model));
}

function validateSeeds(seeds: number[]): void {
  if (seeds.length > MAX_BATCH) throw new RangeError(`at most ${MAX_BATCH} seeds per call, got ${seeds.length}`);
  for (const s of seeds) toUint32Seed(s);
}

export class Generator {
  private readonly backend: Backend;
  private cpu: CpuModel | undefined;
  private gpu: Promise<WebGpuModel> | undefined;
  private disposed = false;

  constructor(options: GenerateOptions = {}) {
    this.backend = options.backend ?? "auto";
  }

  async generate(seed: number): Promise<Sprite> {
    const result = await this.generateMany([seed]);
    return result.sprites[0];
  }

  async generateMany(seeds: number[]): Promise<GenerateResult> {
    this.assertLive();
    validateSeeds(seeds);
    if (this.backend === "webgpu" && !hasWebGpu()) throw new Error("WebGPU is not available in this environment");
    if (seeds.length === 0) return { sprites: [], backend: this.backend === "webgpu" ? "webgpu" : "cpu" };
    const z = new Float32Array(seeds.length * LATENT_DIM);
    seeds.forEach((seed, i) => z.set(seedToLatent(seed), i * LATENT_DIM));
    return this.run(z, seeds.length);
  }

  async fromLatent(z: Float32Array): Promise<Sprite> {
    this.assertLive();
    if (!(z instanceof Float32Array) || z.length !== LATENT_DIM) {
      throw new RangeError(`latent must be a Float32Array of length ${LATENT_DIM}`);
    }
    const logits = this.cpuModel().forward(z, 1);
    return logitsToSprite(logits, 0);
  }

  dispose(): void {
    this.disposed = true;
    this.cpu = undefined;
    void this.gpu?.then((g) => g.dispose(), () => undefined);
    this.gpu = undefined;
  }

  private assertLive(): void {
    if (this.disposed) throw new Error("generator is disposed");
  }

  private cpuModel(): CpuModel {
    return (this.cpu ??= new CpuModel(layers()));
  }

  private gpuModel(): Promise<WebGpuModel> {
    return (this.gpu ??= WebGpuModel.create(layers()));
  }

  private async run(z: Float32Array, batch: number): Promise<GenerateResult> {
    // generateMany() already rejected an explicit "webgpu" backend without support.
    let useGpu = this.backend === "webgpu";
    if (!useGpu && this.backend === "auto" && batch >= AUTO_WEBGPU_MIN_BATCH && hasWebGpu()) {
      try {
        await this.gpuModel();
        useGpu = true;
      } catch {
        useGpu = false;
      }
    }
    if (useGpu) {
      const gpu = await this.gpuModel();
      const logits = await gpu.forward(z, batch);
      return { sprites: split(logits, batch), backend: "webgpu", adapter: gpu.adapterName };
    }
    return { sprites: split(this.cpuModel().forward(z, batch), batch), backend: "cpu" };
  }
}

function split(logits: Float32Array, batch: number): Sprite[] {
  const out: Sprite[] = new Array(batch);
  for (let b = 0; b < batch; b++) out[b] = logitsToSprite(logits, b * SPRITE_PIXELS);
  return out;
}

export function defineGenerator(options?: GenerateOptions): Generator {
  return new Generator(options);
}

// The bare `generate()` function shares one internal Generator per backend across calls
// (never disposed), rather than constructing and tearing one down on every invocation.
const sharedGenerators = new Map<Backend, Generator>();
function sharedGenerator(options?: GenerateOptions): Generator {
  const backend = options?.backend ?? "auto";
  let g = sharedGenerators.get(backend);
  if (!g) {
    g = new Generator(options);
    sharedGenerators.set(backend, g);
  }
  return g;
}

export function generate(seed: number, options?: GenerateOptions): Promise<Sprite>;
export function generate(seeds: number[], options?: GenerateOptions): Promise<Sprite[]>;
export async function generate(seedOrSeeds: number | number[], options?: GenerateOptions): Promise<Sprite | Sprite[]> {
  const g = sharedGenerator(options);
  if (Array.isArray(seedOrSeeds)) {
    const result = await g.generateMany(seedOrSeeds);
    return result.sprites;
  }
  return g.generate(seedOrSeeds);
}
