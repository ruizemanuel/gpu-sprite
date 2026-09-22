import { DENSE_SHADER, WORKGROUP_SIZE } from "./shader.ts";
import type { DecodedLayer } from "./types.ts";

export function hasWebGpu(): boolean {
  return typeof navigator !== "undefined" && typeof (navigator as { gpu?: unknown }).gpu !== "undefined";
}

interface GpuLayer {
  in: number;
  out: number;
  relu: boolean;
  weights: GPUBuffer;
  bias: GPUBuffer;
  params: GPUBuffer;
}

function uploadF32(device: GPUDevice, data: Float32Array): GPUBuffer {
  const buffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/**
 * WGSL runtime. Weights are uploaded once; activation, input, logits and readback
 * buffers grow to the largest batch seen and are reused. Calls are serialized so
 * buffer reuse is safe. Returns logits; thresholding stays on the CPU.
 */
export class WebGpuModel {
  readonly adapterName: string;
  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly layers: GpuLayer[];
  private readonly inputDim: number;
  private readonly outputDim: number;
  private readonly maxDim: number;
  private capacity = 0;
  private zBuf!: GPUBuffer;
  private actA!: GPUBuffer;
  private actB!: GPUBuffer;
  private logitsBuf!: GPUBuffer;
  private readBuf!: GPUBuffer;
  private chain: Promise<unknown> = Promise.resolve();
  private disposed = false;

  private constructor(device: GPUDevice, pipeline: GPUComputePipeline, layers: GpuLayer[], adapterName: string) {
    this.device = device;
    this.pipeline = pipeline;
    this.layers = layers;
    this.adapterName = adapterName;
    this.inputDim = layers[0].in;
    this.outputDim = layers[layers.length - 1].out;
    this.maxDim = Math.max(this.inputDim, ...layers.map((l) => l.out));
    for (const layer of layers) {
      if (layer.out > WORKGROUP_SIZE) throw new Error(`layer width ${layer.out} exceeds workgroup size ${WORKGROUP_SIZE}`);
    }
  }

  static async create(layers: DecodedLayer[]): Promise<WebGpuModel> {
    if (!hasWebGpu()) throw new Error("WebGPU is not available in this environment");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU: no adapter available");
    const device = await adapter.requestDevice();
    const info = (adapter as unknown as { info?: GPUAdapterInfo }).info;
    const adapterName = [info?.vendor, info?.architecture, info?.device, info?.description].filter(Boolean).join(" ") || "unknown adapter";
    const module = device.createShaderModule({ code: DENSE_SHADER });
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const gpuLayers: GpuLayer[] = layers.map((l) => ({
      in: l.in,
      out: l.out,
      relu: l.relu,
      weights: uploadF32(device, l.weights),
      bias: uploadF32(device, l.bias),
      params: device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    }));
    return new WebGpuModel(device, pipeline, gpuLayers, adapterName);
  }

  forward(input: Float32Array, batch: number): Promise<Float32Array> {
    if (this.disposed) return Promise.reject(new Error("WebGPU model is disposed"));
    if (input.length < batch * this.inputDim) {
      return Promise.reject(new RangeError(`input has ${input.length} values, need ${batch * this.inputDim}`));
    }
    const run = () => this.run(input, batch);
    const result = this.chain.then(run, run);
    this.chain = result.catch(() => undefined);
    return result;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const l of this.layers) {
      l.weights.destroy();
      l.bias.destroy();
      l.params.destroy();
    }
    if (this.capacity > 0) {
      this.zBuf.destroy();
      this.actA.destroy();
      this.actB.destroy();
      this.logitsBuf.destroy();
      this.readBuf.destroy();
    }
    this.device.destroy();
  }

  private ensureCapacity(batch: number): void {
    if (batch <= this.capacity) return;
    if (this.capacity > 0) {
      this.zBuf.destroy();
      this.actA.destroy();
      this.actB.destroy();
      this.logitsBuf.destroy();
      this.readBuf.destroy();
    }
    const d = this.device;
    this.capacity = batch;
    this.zBuf = d.createBuffer({ size: batch * this.inputDim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.actA = d.createBuffer({ size: batch * this.maxDim * 4, usage: GPUBufferUsage.STORAGE });
    this.actB = d.createBuffer({ size: batch * this.maxDim * 4, usage: GPUBufferUsage.STORAGE });
    this.logitsBuf = d.createBuffer({ size: batch * this.outputDim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.readBuf = d.createBuffer({ size: batch * this.outputDim * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  private async run(input: Float32Array, batch: number): Promise<Float32Array> {
    this.ensureCapacity(batch);
    const d = this.device;
    // writeBuffer accepts any typed array view at runtime; @webgpu/types only lists ArrayBuffer-backed ones.
    d.queue.writeBuffer(this.zBuf, 0, input as Float32Array<ArrayBuffer>, 0, batch * this.inputDim);
    const encoder = d.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    let xs = this.zBuf;
    const last = this.layers.length - 1;
    this.layers.forEach((layer, li) => {
      d.queue.writeBuffer(layer.params, 0, new Uint32Array([batch, layer.in, layer.out, layer.relu ? 1 : 0]));
      const ys = li === last ? this.logitsBuf : li % 2 === 0 ? this.actA : this.actB;
      const bindGroup = d.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: layer.params } },
          { binding: 1, resource: { buffer: layer.weights } },
          { binding: 2, resource: { buffer: layer.bias } },
          { binding: 3, resource: { buffer: xs } },
          { binding: 4, resource: { buffer: ys } },
        ],
      });
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(batch);
      xs = ys;
    });
    pass.end();
    const bytes = batch * this.outputDim * 4;
    encoder.copyBufferToBuffer(this.logitsBuf, 0, this.readBuf, 0, bytes);
    d.queue.submit([encoder.finish()]);
    await this.readBuf.mapAsync(GPUMapMode.READ, 0, bytes);
    const out = new Float32Array(this.readBuf.getMappedRange(0, bytes).slice(0));
    this.readBuf.unmap();
    return out;
  }
}
