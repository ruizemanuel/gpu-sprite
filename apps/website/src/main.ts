import { defineGenerator, seedToLatent, type GenerateResult, type Generator } from "gpu-sprite";
import { DetailPanel } from "./detail.ts";
import { cellAt, COUNT, drawGrid } from "./grid.ts";

type BackendName = "cpu" | "webgpu";

const gridCanvas = document.querySelector("#grid") as HTMLCanvasElement;
const gridCtx = gridCanvas.getContext("2d")!;
const tooltip = document.querySelector("#tooltip") as HTMLElement;
const status = document.querySelector("#status") as HTMLElement;
const webgpuRadio = document.querySelector("#webgpu-radio") as HTMLInputElement;
const webgpuStatus = document.querySelector("#webgpu-status") as HTMLElement;
const generators: Record<BackendName, Generator> = {
  cpu: defineGenerator({ backend: "cpu" }),
  webgpu: defineGenerator({ backend: "webgpu" }),
};
const detail = new DetailPanel(document.querySelector("#detail") as HTMLElement, generators.cpu);

const state = { base: 0, backend: "cpu" as BackendName, lastMs: 0, count: 0, seeds: [] as number[], rendering: false, renderedBase: -1 };
(window as unknown as { __gpuSprite: typeof state }).__gpuSprite = state;

function seedsFrom(base: number): number[] {
  return Array.from({ length: COUNT }, (_, i) => base + i);
}

// Both the backend radio and Regenerate fire `render()` without awaiting it, so two calls can
// be in flight at once. A monotonic token makes the *last issued* call the only one allowed to
// touch `state`/the canvas/`#status`, regardless of which `generateMany()` promise resolves last.
let renderToken = 0;

async function render(): Promise<void> {
  const token = ++renderToken;
  const backend = state.backend;
  const base = state.base;
  const seeds = seedsFrom(base);
  state.rendering = true;
  const t0 = performance.now();
  let result: GenerateResult;
  try {
    result = await generators[backend].generateMany(seeds);
  } catch (err) {
    if (token !== renderToken) return; // superseded by a newer render; do not touch state or #status
    state.rendering = false;
    status.textContent = `${backend} failed: ${(err as Error).message}`;
    return;
  }
  if (token !== renderToken) return; // superseded; discard this stale result
  state.seeds = seeds;
  state.lastMs = performance.now() - t0;
  state.count = result.sprites.length;
  state.renderedBase = base;
  state.rendering = false;
  drawGrid(gridCtx, result.sprites);
  const adapter = result.adapter ? ` on ${result.adapter}` : "";
  status.textContent = `${result.backend}: ${state.lastMs.toFixed(1)} ms for ${state.count} sprites${adapter} (seeds ${base}–${base + COUNT - 1})`;
}

/** Runs the 64-seed CPU/WebGPU parity check once on load and shows the visible result next to the toggle. */
async function checkWebGpu(): Promise<void> {
  if (!("gpu" in navigator)) {
    const reason = "WebGPU is not available in this browser";
    webgpuRadio.title = reason;
    webgpuStatus.textContent = reason;
    return;
  }
  const seeds = seedsFrom(1_000_000).slice(0, 64);
  try {
    const [cpu, gpu] = await Promise.all([generators.cpu.generateMany(seeds), generators.webgpu.generateMany(seeds)]);
    const identical = cpu.sprites.every((s, i) => s.pixels.every((p, j) => p === gpu.sprites[i].pixels[j]));
    if (!identical) {
      const reason = "WebGPU output differs from the CPU reference on this device; disabled";
      webgpuRadio.title = reason;
      webgpuStatus.textContent = reason;
      return;
    }
    webgpuRadio.disabled = false;
    const ok = `parity OK on ${gpu.adapter ?? "unknown adapter"}`;
    webgpuRadio.title = ok;
    webgpuStatus.textContent = ok;
  } catch (err) {
    const reason = `WebGPU unavailable: ${(err as Error).message}`;
    webgpuRadio.title = reason;
    webgpuStatus.textContent = reason;
  }
}

document.querySelectorAll<HTMLInputElement>('input[name="backend"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    state.backend = radio.value as BackendName;
    void render();
  });
});
document.querySelector("#regenerate")!.addEventListener("click", () => {
  state.base += COUNT;
  void render();
});
gridCanvas.addEventListener("mousemove", (e) => {
  const i = cellAt(gridCanvas, e.clientX, e.clientY);
  tooltip.hidden = i < 0;
  if (i >= 0) {
    tooltip.textContent = `seed ${state.seeds[i]}`;
    tooltip.style.left = `${e.clientX - gridCanvas.getBoundingClientRect().left + 12}px`;
    tooltip.style.top = `${e.clientY - gridCanvas.getBoundingClientRect().top + 12}px`;
  }
});
gridCanvas.addEventListener("mouseleave", () => (tooltip.hidden = true));
gridCanvas.addEventListener("click", (e) => {
  const i = cellAt(gridCanvas, e.clientX, e.clientY);
  if (i >= 0) void detail.show(state.seeds[i], seedToLatent(state.seeds[i]));
});

fetch("./size.json")
  .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
  .then((s: { brotli: number; minified: number }) => {
    (document.querySelector("#size") as HTMLElement).textContent = `${(s.brotli / 1024).toFixed(1)} KiB Brotli (${(s.minified / 1024).toFixed(1)} KiB minified)`;
  })
  .catch(() => undefined);

await render();
await checkWebGpu();
