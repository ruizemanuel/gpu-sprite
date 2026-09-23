import test, { afterEach } from "node:test";
import assert from "node:assert/strict";
import { AUTO_WEBGPU_MIN_BATCH, defineGenerator, type Sprite } from "../src/index.ts";
import { decodeModel } from "../src/model/decode.ts";
import { WebGpuModel } from "../src/model/webgpu.ts";
import { model } from "../src/model/weights.ts";

// A fake WebGPU on `navigator.gpu`: enough surface for WebGpuModel to create, run and lose a
// device, with switches for the failures under test. It computes nothing (logits read back as
// zeros); real GPU results are covered by the browser parity test.

interface FakeDevice {
  destroyed: boolean;
  isLost: boolean;
  lost: Promise<GPUDeviceLostInfo>;
  loseDevice(): void;
}

interface FakeGpu {
  noAdapter: boolean;
  pipelineFails: boolean;
  readFails: boolean;
  adapterRequests: number;
  devices: FakeDevice[];
}

function fakeDevice(gpu: FakeGpu) {
  let resolveLost: (info: GPUDeviceLostInfo) => void = () => undefined;
  const lose = (reason: string) => {
    device.isLost = true;
    resolveLost({ reason, message: `fake device ${reason}` } as GPUDeviceLostInfo);
  };
  const buffer = (size: number) => ({
    getMappedRange: (offset = 0, length = size - offset) => new ArrayBuffer(length),
    unmap() {},
    destroy() {},
    mapAsync: () => (gpu.readFails || device.isLost ? Promise.reject(new Error("fake GPU read failed")) : Promise.resolve()),
  });
  const device = {
    destroyed: false,
    isLost: false,
    lost: new Promise<GPUDeviceLostInfo>((resolve) => {
      resolveLost = resolve;
    }),
    loseDevice: () => lose("unknown"),
    destroy() {
      device.destroyed = true;
      lose("destroyed");
    },
    queue: { writeBuffer() {}, submit() {} },
    createShaderModule: () => ({}),
    createComputePipelineAsync: () =>
      gpu.pipelineFails ? Promise.reject(new Error("fake pipeline compile error")) : Promise.resolve({ getBindGroupLayout: () => ({}) }),
    createBuffer: (descriptor: { size: number }) => buffer(descriptor.size),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }),
      copyBufferToBuffer() {},
      finish: () => ({}),
    }),
  };
  gpu.devices.push(device);
  return device;
}

function installFakeGpu(): FakeGpu {
  const gpu: FakeGpu = { noAdapter: false, pipelineFails: false, readFails: false, adapterRequests: 0, devices: [] };
  const adapter = { info: { vendor: "fake" }, requestDevice: async () => fakeDevice(gpu) };
  const requestAdapter = async () => {
    gpu.adapterRequests++;
    return gpu.noAdapter ? null : adapter;
  };
  Object.defineProperty(navigator, "gpu", { configurable: true, value: { requestAdapter } });
  Object.assign(globalThis, {
    GPUBufferUsage: { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 },
    GPUMapMode: { READ: 1 },
  });
  return gpu;
}

afterEach(() => {
  delete (navigator as { gpu?: unknown }).gpu;
  delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
  delete (globalThis as { GPUMapMode?: unknown }).GPUMapMode;
});

const seeds = Array.from({ length: AUTO_WEBGPU_MIN_BATCH }, (_, i) => i);
const pixels = (sprites: Sprite[]) => sprites.map((s) => Array.from(s.pixels));

async function cpuPixels(): Promise<number[][]> {
  const cpu = defineGenerator({ backend: "cpu" });
  const result = await cpu.generateMany(seeds);
  cpu.dispose();
  return pixels(result.sprites);
}

/** Lets the generator's handlers on a device's `lost` promise run. */
const settle = () => new Promise((resolve) => setImmediate(resolve));

test("create checks layer widths before requesting a device", async () => {
  const gpu = installFakeGpu();
  const [first, second] = decodeModel(model);
  await assert.rejects(() => WebGpuModel.create([{ ...first, out: 300 }, second]), /exceeds workgroup size/);
  assert.equal(gpu.adapterRequests, 0);
});

test("a pipeline that fails to build rejects create and destroys the device", async () => {
  const gpu = installFakeGpu();
  gpu.pipelineFails = true;
  await assert.rejects(() => WebGpuModel.create(decodeModel(model)), /fake pipeline compile error/);
  assert.equal(gpu.devices.length, 1);
  assert.equal(gpu.devices[0].destroyed, true);
});

test("auto stays on the CPU when the GPU cannot be created, without probing again", async () => {
  const gpu = installFakeGpu();
  gpu.noAdapter = true;
  const g = defineGenerator({ backend: "auto" });
  const first = await g.generateMany(seeds);
  const second = await g.generateMany(seeds);
  assert.equal(first.backend, "cpu");
  assert.equal(second.backend, "cpu");
  assert.deepEqual(pixels(first.sprites), await cpuPixels());
  assert.equal(gpu.adapterRequests, 1);
  g.dispose();
});

test("explicit webgpu tries again after a failed creation", async () => {
  const gpu = installFakeGpu();
  gpu.noAdapter = true;
  const g = defineGenerator({ backend: "webgpu" });
  await assert.rejects(() => g.generateMany(seeds), /no adapter/);
  gpu.noAdapter = false;
  const result = await g.generateMany(seeds);
  assert.equal(result.backend, "webgpu");
  assert.equal(gpu.adapterRequests, 2);
  g.dispose();
});

test("auto re-runs a batch on the CPU when the GPU call fails, and uses a new device next time", async () => {
  const gpu = installFakeGpu();
  gpu.readFails = true;
  const g = defineGenerator({ backend: "auto" });
  const failed = await g.generateMany(seeds);
  assert.equal(failed.backend, "cpu");
  assert.deepEqual(pixels(failed.sprites), await cpuPixels());
  assert.equal(gpu.devices[0].destroyed, true);
  gpu.readFails = false;
  const next = await g.generateMany(seeds);
  assert.equal(next.backend, "webgpu");
  assert.equal(gpu.devices.length, 2);
  g.dispose();
});

test("explicit webgpu rejects a failed GPU call, never falls back, and uses a new device next time", async () => {
  const gpu = installFakeGpu();
  gpu.readFails = true;
  const g = defineGenerator({ backend: "webgpu" });
  await assert.rejects(() => g.generateMany(seeds), /fake GPU read failed/);
  assert.equal(gpu.devices[0].destroyed, true);
  gpu.readFails = false;
  const next = await g.generateMany(seeds);
  assert.equal(next.backend, "webgpu");
  assert.equal(gpu.devices.length, 2);
  g.dispose();
});

test("a lost device is released and replaced on the next call", async () => {
  const gpu = installFakeGpu();
  const g = defineGenerator({ backend: "webgpu" });
  assert.equal((await g.generateMany(seeds)).backend, "webgpu");
  gpu.devices[0].loseDevice();
  await settle();
  assert.equal(gpu.devices[0].destroyed, true);
  const next = await g.generateMany(seeds);
  assert.equal(next.backend, "webgpu");
  assert.equal(gpu.devices.length, 2);
  g.dispose();
});

test("auto survives a device lost right before a call", async () => {
  const gpu = installFakeGpu();
  const g = defineGenerator({ backend: "auto" });
  assert.equal((await g.generateMany(seeds)).backend, "webgpu");
  gpu.devices[0].loseDevice();
  // Must not reject, whichever backend ends up running it.
  const during = await g.generateMany(seeds);
  assert.equal(during.sprites.length, seeds.length);
  if (during.backend === "cpu") assert.deepEqual(pixels(during.sprites), await cpuPixels());
  await settle();
  const after = await g.generateMany(seeds);
  assert.equal(after.backend, "webgpu");
  g.dispose();
});
