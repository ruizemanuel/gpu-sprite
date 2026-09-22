# gpu-sprite model card

Every measured number below comes from `packages/training/active/report.json` or from the output of `pnpm --filter gpu-sprite test:browser` and `pnpm --filter gpu-sprite bench:browser`; test tolerances come from the tests themselves. Percentages are the recorded fractions × 100, rounded to two decimals.

## Model

Checkpoint `baseline@2650bc19f36a`, exported `2026-09-22T19:31:50.471293+00:00`. Decoder: 32 → 128 (ReLU) → 256, 37,248 parameters, int8 symmetric per-tensor quantization of the weights (scales 0.011377585120499134 and 0.010982365347445011), float32 biases. Latent: 32 dimensions from the seed via xorshift32 + Irwin-Hall (`seedVersion` 1). Output: 256 logits. Decoding: a pixel is foreground when its logit is > 0; then every foreground pixel with no foreground 8-neighbour is cleared, in one pass over the thresholded image (pixels outside the grid count as background).

Training objective: `vae` (β 1.0 with a linear warm-up over the first 30% of epochs, 300 epochs, learning rate 0.001, batch 64, seed 1337; encoder 256 → 128 → 64, i.e. mean and log-variance of the 32-dimensional latent). The checkpoint kept is the epoch with the lowest validation reconstruction loss: epoch 246 (0-based), validation reconstruction BCE 52.13 per sprite (summed over the 256 pixels), pixel accuracy 91.56% (float decoder, before quantization and cleanup). Training runs single-threaded (`torch.set_num_threads(1)`), which is much faster on the CPU for a model this small; the recorded run took 23.3 seconds.

A GAN objective (`train.py --objective gan`: the same decoder trained as a generator against a small discriminator) exists as a fallback and was tried. It did not pass the quality gate and was not promoted.

## Intended use

Runtime generation of small 1-bit sprites in games and demos, and experiments on tiny generative models in the browser. Not intended for producing final art assets; outputs are variations in the style of the training set and may be malformed.

## Training data

Kenney 1-Bit Pack (CC0), 1,078 tiles of 16×16 from `Tilesheet/monochrome_packed.png`. 217 were removed: empty and full tiles, and the 37 curated index ranges in `packages/training/data/manifest.json` (noise and dither textures, diagonal and dither terrain, UI buttons, frames, sliders and progress bars, mouse cursors, playing cards, letters, digits and punctuation, math symbols, arrows and chevrons). 13 of the remaining tiles were duplicates under horizontal flip, leaving 848 unique tiles. Split by tile hash and augmented with horizontal flips: 1,205 training and 117 validation examples. Mean foreground density of the training examples: 30.35% of pixels (p1 7.42%, p5 10.94%, p95 51.56%, p99 73.44%).

## Evaluation

All numbers are for the quantized int8 decoder, seeds 0–999 (1,000 sprites), after the full decode rule above.

| Check | Value | Threshold | Result |
| --- | --- | --- | --- |
| Unique sprites | 100.00% | ≥ 95.00% (fixed) | PASS |
| Median Hamming distance to nearest training sprite | 43 pixels | ≥ 8 (fixed) | PASS |
| Sprites with ≤ 2 connected components (8-connectivity) | 41.20% | ≥ 38.46%, the validation set's own share | PASS |
| Sample density p5–p95 | 9.36%–45.72% | within the training p1–p99, 7.42%–73.44% | PASS |
| Validation reconstruction pixel accuracy | 91.46% | ≥ 90.46% (`first-promotion`: measured − 0.01) | PASS |

Fixed thresholds live in `packages/training/metrics.py`. The coherence and density thresholds are computed from the dataset by `data.py` into `data/stats.json` and recorded under `gate.checks.*.threshold` in `active/report.json`. The reconstruction check encodes the validation set with the encoder mean, decodes it with the int8 decoder and compares the final sprites pixel by pixel; its floor was set on the first promotion as the measured accuracy minus 0.01 and is never lowered.

These checks catch memorization, collapse and fragmentation. They do not measure how good a sprite looks; the contact sheet `samples.png` of the promoted run was reviewed by eye.

### Gate history

The original plan required at least 80% of samples with ≤ 2 components and sample density p5/p95 within the training p5/p95. Neither threshold had been measured against the data, and the real validation sprites fail both: only 38.46% of them have ≤ 2 components, and their density p95 lies above the training p95. On 2026-09-22 the owner replaced them with the data-calibrated thresholds above (the validation set's own share for coherence, the training p1/p99 for density) and added the isolated-pixel cleanup to the decode rule. Uniqueness, median Hamming distance and the reconstruction floor are unchanged.

## Parity

64 seeds are recorded in `active/parity.json` with the latent, float32 logits computed in numpy in the same operation order as the CPU runtime, the raw thresholded bits and the final sprite. The Node test requires identical latents, logits within 1e-5, identical bits where the logit margin exceeds 1e-5, and identical final sprites. The browser test runs the same 64 seeds in real Chromium and requires finite WebGPU logits within 1e-4 of the CPU, identical bits where the margin exceeds 1e-3, low-margin pixels at most 0.1% of all fixture pixels, a single-seed call within 1e-4 of the batched one, and, for 16 of the seeds, identical final sprites from `generateMany` on both backends.

Last measured (`pnpm --filter gpu-sprite test:browser`): adapter `amd gcn-4`, 64 seeds, 16,384 pixels, max logit difference 0.00e+0, 5 low-margin pixels, batched vs single-seed difference 0.00e+0, `generateMany` sprites identical between WebGPU and CPU.

## Backend crossover

`backend: "auto"` uses WebGPU from `AUTO_WEBGPU_MIN_BATCH = 64` seeds. `pnpm --filter gpu-sprite bench:browser` times `generateMany` on a CPU generator and on a WebGPU generator in Chromium, one warm-up call and then the median of 15 timed calls per backend and batch size. Both timings include seed → latent and the CPU decode of the logits. Measured on a Windows 10 desktop with an Intel Core i5-7500 and an AMD Radeon RX 480 (Chromium adapter `amd gcn-4`):

| Batch | CPU ms | WebGPU ms | CPU / WebGPU |
| ---: | ---: | ---: | ---: |
| 1 | 0.20 | 3.00 | 0.07 |
| 4 | 0.70 | 3.00 | 0.23 |
| 16 | 2.50 | 3.00 | 0.83 |
| 64 | 9.60 | 4.00 | 2.40 |
| 256 | 38.20 | 5.10 | 7.49 |
| 1024 | 152.30 | 15.10 | 10.09 |
| 4096 | 609.60 | 45.60 | 13.37 |

The smallest measured batch where WebGPU is faster is 64, which matches `AUTO_WEBGPU_MIN_BATCH`. On this machine the true crossover lies above 16 and at most 64. A WebGPU call costs about 3 ms even for one sprite (the fixed cost of a submit and a readback), and the browser reports `performance.now()` in 0.1 ms steps, so the smallest CPU timings are coarse. Other machines will differ; the threshold was not tuned to this measurement.

## Limitations

- 1-bit only, 16×16 only, no conditioning on category or text.
- Pixels are independent given the latent; ambiguous regions can look blurry or fragmented. The isolated-pixel cleanup removes single stray pixels but not larger fragments.
- Coherence passes by a small margin: 41.20% of samples have at most two connected components against a threshold of 38.46%, so 58.80% of samples have three or more. The threshold is the real validation set's own share (61.54% of the validation sprites also have three or more components, since many sprites in the pack are made of several parts), but a retrained model could easily fall below it.
- The densest samples are lighter than the densest training sprites (sample p95 45.72% against training p95 51.56%).
- The training set is one artist's pack of 848 unique tiles, so diversity is bounded by it.
- Seeds map to latents deterministically but not meaningfully; neighbouring seeds are unrelated sprites. Use `fromLatent` to interpolate.
- WebGPU requires a secure context and a supported browser. Small batches are faster on the CPU.

## Reproducibility

`pnpm data`, then `train.py --run <name>`, `evaluate.py`, `export.py` as in the README. Training is seeded (1337) but CPU PyTorch kernels can differ across versions; a re-run should land near the recorded metrics, not on them exactly. `active/manifest.json` records the sha256 of the checkpoint, of the training and validation arrays and of the generated `weights.ts`. Promotion never lowers the reconstruction floor.
