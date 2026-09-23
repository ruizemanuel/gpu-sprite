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

The browser package embeds the decoder as int6 weights (one character each) and evaluates it either in plain TypeScript or in a WGSL compute shader with one workgroup per sprite and one thread per output unit. The GPU returns logits; thresholding and the isolated-pixel cleanup stay on the CPU, so both backends can be compared bit for bit. `backend: "auto"` (the default) uses WebGPU for batches of 64 or more sprites when a device is available, and the CPU otherwise. Explicit `"webgpu"` never falls back. If the GPU fails during an `auto` call, for example because the device was lost, that batch runs on the CPU and a later batch gets a new device. An explicit `"webgpu"` generator rejects that call and starts over on the next one.

## Model

The shipped checkpoint is `characters@d6b814721862` with 37,248 decoder parameters, trained with β = 4 only on the characters-and-creatures family of the pack: 106 unique tiles. The minified bundle is 48,437 bytes, 24,877 bytes Brotli-compressed, against a budget of 40,000. On 1,000 fixed seeds, 99.20% of sprites are unique, the median Hamming distance to the nearest training sprite is 28 pixels, and 64.30% have at most two connected components, against the 32.92% raw share among all the real sprites of the dataset that sets the gate's threshold: the samples are not more fragmented than the real sprites, which is what that check guards against. See [MODEL_CARD.md](MODEL_CARD.md) for the evaluation contract, the model history, the measured CPU/WebGPU crossover and the limitations.

## Development

Node 24+, pnpm 11, Python 3.12 and uv.

```sh
pnpm install
uv sync --project packages/training
pnpm -C packages/core exec playwright install chromium   # browser for the WebGPU tests
pnpm test          # Node unit tests and CPU parity
pnpm test:py       # Python tests
pnpm build:core    # dist/ and the bundle
pnpm size          # Brotli size gate (40,000 bytes)
pnpm test:browser  # real Chromium WebGPU parity and demo smoke
pnpm check         # all of the above, in that order
pnpm dev           # demo at http://localhost:4174
pnpm --filter gpu-sprite bench:browser   # CPU vs WebGPU timings (after build:core; not part of check)
```

To reproduce the model:

```sh
pnpm data                                                              # downloads Kenney 1-Bit Pack (CC0), builds train/val from the characters and creatures
uv run --project packages/training python packages/training/train.py --run characters --beta 4 --epochs 600
uv run --project packages/training python packages/training/evaluate.py --checkpoint packages/training/runs/characters/best.pt
uv run --project packages/training python packages/training/export.py --checkpoint packages/training/runs/characters/best.pt
```

`pnpm data` keeps only the tiles inside the `include` index ranges of `packages/training/data/manifest.json`, which select the characters-and-creatures family. `export.py` runs the quality gate on the quantized decoder and refuses to write `weights.ts` if it fails. `packages/training/active/` holds the promoted report and the parity fixtures the tests check against.

## Repository

- `packages/core`: the `gpu-sprite` package, WGSL kernel and CPU reference
- `packages/training`: dataset, PyTorch training, evaluation, export and provenance
- `apps/website`: static demo

Architecture details are in [architecture.md](architecture.md). Attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT © Emanuel Ruiz. Training data is Kenney's 1-Bit Pack, CC0.
