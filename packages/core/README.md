# gpu-sprite

`gpu-sprite` is an experimental, learned 16×16 1-bit sprite generator that runs inside a web page. One small model turns any integer seed into a character or creature sprite, on the CPU or on WebGPU, with no server, no model download and no runtime dependencies.

```js
import { generate } from "gpu-sprite";

const sprite = await generate(42);
// { width: 16, height: 16, pixels: Uint8Array(256) } — row-major, 1 = foreground

const sprites = await generate([1, 2, 3], { backend: "webgpu" });
```

For repeated calls, `defineGenerator({ backend })` returns a generator that keeps its model (and, for WebGPU, its device and buffers) between calls: `generateMany(seeds)` returns the sprites plus the backend and adapter that ran them, `fromLatent(z)` decodes a 32-float latent directly, and `dispose()` releases everything.

It exists to be embedded in games: a roguelike can generate a new bestiary per run from seeds, at runtime, and ship the generator inside its bundle. It is not a replacement for an artist or for cloud pixel-art tools; it produces small variations in the style of its training set.

## How it works

A seed drives a 32-bit xorshift generator; for each of the 32 latent dimensions, twelve uniforms are summed and 6 is subtracted, which approximates a standard normal. A two-layer MLP decoder (32 → 128 → 256) maps that latent to 256 logits, one per pixel. A pixel is foreground when its logit is positive; then every foreground pixel with no foreground neighbour among its 8 neighbours is cleared, in one pass. The decoder was trained as the decoder of a variational autoencoder on the characters and creatures of Kenney's CC0 1-Bit Pack; the encoder is not shipped.

The browser package embeds the decoder as int6 weights (one character each) and evaluates it either in plain TypeScript or in a WGSL compute shader with one workgroup per sprite and one thread per output unit. The GPU returns logits; thresholding and the isolated-pixel cleanup stay on the CPU, so both backends can be compared bit for bit. `backend: "auto"` (the default) uses WebGPU for batches of 64 or more sprites when a device is available, and the CPU otherwise. Explicit `"webgpu"` never falls back. If the GPU fails during an `auto` call, for example because the device was lost, that batch runs on the CPU and a later batch gets a new device. An explicit `"webgpu"` generator rejects that call and starts over on the next one. If a device cannot be created at all, that `auto` generator stays on the CPU for good; create a new generator to try again.

## License

MIT © Emanuel Ruiz. Training data is Kenney's 1-Bit Pack, CC0.
