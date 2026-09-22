import type { DecodedLayer } from "./types.ts";

export class WebGpuModel {
  readonly adapterName = "stub";
  static async create(_layers: DecodedLayer[]): Promise<WebGpuModel> {
    throw new Error("WebGPU runtime not implemented yet");
  }
  async forward(_input: Float32Array, _batch: number): Promise<Float32Array> {
    throw new Error("WebGPU runtime not implemented yet");
  }
  dispose(): void {}
}

export function hasWebGpu(): boolean {
  return typeof navigator !== "undefined" && typeof (navigator as { gpu?: unknown }).gpu !== "undefined";
}
