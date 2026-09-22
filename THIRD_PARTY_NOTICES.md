# Third-party notices

The `gpu-sprite` package has no runtime dependencies. Everything below is training data, a development dependency, or prior work that shaped this project.

## Training data

- **Kenney 1-Bit Pack** (version 1.2) by Kenney (https://kenney.nl/assets/1-bit-pack), released under CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/), as stated in the `License.txt` inside the pack. Pinned zip: `https://kenney.nl/media/pages/assets/1-bit-pack/aa867a1f37-1677578516/kenney_1-bit-pack.zip`, sha256 `129a0e74e1dc9091a769d5118be5c88780484c7a3771ee33ac9ef0df02583a9a`. The pack is downloaded by `packages/training/data.py` and is not redistributed in this repository. The trained weights in `packages/core/src/model/weights.ts` are derived from it. CC0 requires no attribution; it is given anyway.

## Prior work

- **gpu-lexer** by Shu Ding / Vercel Labs (https://github.com/vercel-labs/gpu-lexer, MIT) and **gpu-time** by Arik Chakma (https://github.com/arikchakma/gpu-time, MIT): the repository layout, model card conventions, parity testing and size-gate approach follow theirs. No code or weights are reused.
- **pixel-sprite-generator** by Zelimir Fedoran (https://github.com/zfedoran/pixel-sprite-generator, MIT): the idea of runtime procedural sprite generation. No code is reused.

## Development dependencies

TypeScript, esbuild, Playwright, Vite and `@webgpu/types` (JavaScript); PyTorch, NumPy, Pillow and pytest (Python). They are used to build, train and test, and none of them is bundled into the package. See `pnpm-lock.yaml` and `packages/training/uv.lock` for exact versions, and each project for its license.
