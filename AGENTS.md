# gpu-sprite

- pnpm for every JavaScript operation, uv for every Python operation. Never npm, npx or pip. Python scripts run as `uv run --project packages/training python packages/training/<script>.py`.
- `packages/core` has zero runtime dependencies. Keep it that way.
- Node runs the TypeScript sources directly (tests, scripts, browser harness), and `tsconfig.json` sets `erasableSyntaxOnly`. Do not use TypeScript parameter properties (`constructor(private readonly x: T)`), enums or namespaces; declare fields and assign them in the constructor.
- Never hand-edit `packages/core/src/model/weights.ts`. `export.py` generates it and records its hash in `packages/training/active/manifest.json`.
- Read every number for docs from `packages/training/active/report.json`, or from the output of `pnpm size`, `pnpm test:browser` or `pnpm --filter gpu-sprite bench:browser`. Never quote a metric from memory or from this file.
- Do not lower a gate. The Brotli budget is 40,000 bytes (`packages/core/scripts/size.ts`). The fixed quality thresholds (uniqueness, median Hamming distance) live in `packages/training/metrics.py`. The data-derived thresholds (coherence, density) are computed by `data.py` into `packages/training/data/stats.json`; change them only by changing the data, never by editing the file. The reconstruction floor in `active/report.json` never goes down for a given dataset; it is keyed to the sha256 of the training and validation arrays, and a new dataset starts its own floor.
- The shipped model is trained only on the characters-and-creatures family, selected by the `include` index ranges in `packages/training/data/manifest.json`. Changing `include` or `exclude` changes the dataset, its thresholds and its reconstruction floor; say so in `MODEL_CARD.md`.
- The decode rule (`logit > 0`, then clear foreground pixels with no foreground 8-neighbour, one pass) is shared by `packages/core/src/sprite.ts` and `packages/training/sprite.py`. Change both or neither, then re-export.
- The seed → latent algorithm is a contract shared by `seed.ts` and `seed.py`; changing it requires bumping `SEED_VERSION` in both, regenerating `seed-vectors.json` (`node packages/core/scripts/gen-seed-vectors.ts`), and re-exporting.
- The CPU runtime and `quantize.py:forward_sequential` must stay operation-for-operation identical; parity tests depend on it.
- WebGPU parity runs in real Chromium (`pnpm test:browser`). Report which adapter ran. If only SwiftShader is available (`--swiftshader`), say so; never count a CPU result as a GPU pass.
- `AUTO_WEBGPU_MIN_BATCH` in `packages/core/src/index.ts` sets where `auto` switches to WebGPU. If you change it, re-run `pnpm --filter gpu-sprite bench:browser` and update the crossover in `MODEL_CARD.md`.
- Before promoting a model: `pnpm data` unchanged, `evaluate.py` passes, `export.py` run, then `pnpm check`. Update `README.md` and `MODEL_CARD.md` from the new `active/report.json`.

## Commands

```sh
pnpm test          # Node tests
pnpm test:py       # Python tests
pnpm build:core && pnpm size
pnpm test:browser  # Chromium WebGPU parity + demo smoke
pnpm check         # all of the above
pnpm --filter gpu-sprite bench:browser   # CPU vs WebGPU timings, after build:core; not part of check
```
