# gpu-sprite model card

Every measured number below comes from `packages/training/active/report.json` or from the output of `pnpm size`, `pnpm --filter gpu-sprite test:browser` and `pnpm --filter gpu-sprite bench:browser`; test tolerances come from the tests themselves. Percentages are the recorded fractions × 100, rounded to two decimals.

## Model

Checkpoint `characters@d6b814721862`, exported `2026-09-23T01:47:30.313107+00:00`. Decoder: 32 → 128 (ReLU) → 256, 37,248 parameters, int6 symmetric per-tensor quantization of the weights (scales 0.03697691857814789 and 0.01948658749461174), biases rounded to 3 decimals and stored as float32. Latent: 32 dimensions from the seed via xorshift32 + Irwin-Hall (`seedVersion` 1). Output: 256 logits. Decoding: a pixel is foreground when its logit is > 0; then every foreground pixel with no foreground 8-neighbour is cleared, in one pass over the thresholded image (pixels outside the grid count as background).

Training objective: `vae` (β 4.0 with a linear warm-up over the first 30% of epochs, 600 epochs, learning rate 0.001, batch 64, seed 1337; encoder 256 → 128 → 64, i.e. mean and log-variance of the 32-dimensional latent). The checkpoint kept is the epoch with the lowest validation reconstruction loss: epoch 587 (0-based), validation reconstruction BCE 70.48 per sprite (summed over the 256 pixels), pixel accuracy 88.01% (float decoder, before quantization and cleanup). β = 4 was chosen from a scratch experiment on this dataset that gave legible samples without copying. Training runs single-threaded (`torch.set_num_threads(1)`), which is much faster on the CPU for a model this small; the recorded run took 6.3 seconds.

A GAN objective (`train.py --objective gan`: the same decoder trained as a generator against a small discriminator) exists as a fallback. It was tried on the whole-pack dataset of the first v1 model, did not pass the quality gate and was not promoted; it has not been tried on the characters-and-creatures dataset.

## Intended use

Runtime generation of small 1-bit character and creature sprites in games and demos (a bestiary, enemies, NPCs), and experiments on tiny generative models in the browser. Not intended for producing final art assets; outputs are variations in the style of the training set and may be malformed. The model does not generate items, terrain, furniture or UI.

## Training data

Kenney 1-Bit Pack (CC0), 1,078 tiles of 16×16 from `Tilesheet/monochrome_packed.png`. Only the characters-and-creatures family is used: the `include` index ranges in `packages/training/data/manifest.json` select it, and the other 972 tiles are dropped (`data.not_included`). None of the included tiles is empty, full or inside a curated `exclude` range (`data.excluded` is 0; those ranges still document the curation of the rest of the sheet), and none duplicates another under horizontal flip, leaving 106 unique tiles. Split by tile hash and augmented with horizontal flips: 144 training and 17 validation examples. Mean foreground density of the training examples: 30.73% of pixels (p1 21.26%, p5 23.44%, p95 41.66%, p99 46.09%).

## Evaluation

All numbers are for the quantized int6 decoder, seeds 0–999 (1,000 sprites), after the full decode rule above.

| Check | Value | Threshold | Result |
| --- | --- | --- | --- |
| Unique sprites | 99.20% | ≥ 95.00% (fixed) | PASS |
| Median Hamming distance to nearest training sprite | 28 pixels | ≥ 8 (fixed) | PASS |
| Sprites with ≤ 2 connected components (8-connectivity) | 64.30% | ≥ 32.92%, the raw share among all the real sprites of the dataset (no cleanup) | PASS |
| Sample density p5–p95 | 21.48%–35.94% | within the training p1–p99, 21.26%–46.09% | PASS |
| Validation reconstruction pixel accuracy | 87.91% | ≥ 86.96% (`active`: this dataset's floor, set at its first promotion as measured − 0.01) | PASS |

Fixed thresholds live in `packages/training/metrics.py`. The coherence threshold is the share of sprites with ≤ 2 components among all the real sprites of the dataset, the 144 training and 17 validation examples together (`data.components_fraction_real`), measured on *raw* pixels: the isolated-pixel cleanup is not applied to that reference, because cleanup is part of generation and the dataset tiles are not generated. The samples reach 64.30% against 32.92%, so they are not more fragmented than the real sprites, which is all this check claims. Most real sprites have three or more components because characters are drawn with detached parts (held weapons and shields, eyes); a higher share in the samples does not mean they look better. The density threshold is computed from the dataset by `data.py` into `data/stats.json` and recorded under `gate.checks.*.threshold` in `active/report.json`. The reconstruction check encodes the validation examples with the encoder mean, decodes them with the quantized decoder and compares the final sprites pixel by pixel. Its floor belongs to the dataset, identified by the sha256 of the training and validation arrays. `export.py` records the floors in `active/report.json` under `floors_by_dataset`, carrying forward every floor the previous active report knows, so a dataset's floor survives a switch to another dataset and back. `evaluate.py` uses the floor recorded for the current dataset (`active`). A dataset without one starts at the measured accuracy minus 0.01, labelled `new-dataset` when the active report records floors for other datasets and `first-promotion` when it records none. This dataset's floor was set when this checkpoint was first promoted with int8 weights, and the int6 re-export was checked against it (`active`). It is now recorded in `floors_by_dataset` and is never lowered for this dataset.

These checks catch memorization, collapse and fragmentation. They do not measure how good a sprite looks; the contact sheet `samples.png` of the promoted run was reviewed by eye. It shows recognisable humanoids (head, body, arms and legs) and crab-, bug- and skull-like creatures, with a few fragmented or nearly empty sprites; animals such as ducks, dogs or snakes appear rarely and less clearly than in the training set.

### Gate history

The original plan required at least 80% of samples with ≤ 2 components and sample density p5/p95 within the training p5/p95. Neither threshold had been measured against the data, and the real validation sprites of the whole pack failed both. On 2026-09-22 the owner replaced them with data-calibrated thresholds (the real sprites' own raw share for coherence, the training p1/p99 for density) and added the isolated-pixel cleanup to the decode rule. The coherence reference was first the share among the validation sprites only. When the dataset changed to the characters-and-creatures family, whose validation split is far too small to estimate that share, it became the share among all the real sprites of the dataset, and reconstruction floors became per dataset (see Model history). Uniqueness and median Hamming distance are unchanged.

## Model history

- v1 first shipped `baseline@2650bc19f36a`: the same decoder architecture trained on the whole pack (walls, furniture, weapons, characters and every other category that survived the curation). It passed the gate, but its samples mixed categories and were hard to read.
- It was replaced by `characters@d6b814721862`, a decoder trained only on the characters and creatures, with β = 4.
- The reconstruction floor restarted with the new dataset: this model's floor was set at its own first promotion. Floors are remembered per dataset in `active/report.json` from the next export on. The whole-pack dataset's floor (90.46%, recorded in the `active/report.json` of `baseline@2650bc19f36a` before that map existed) is not carried in the map, so re-promoting a whole-pack model would start a new floor.
- The coherence reference is now the share over all the real sprites of the dataset, because the new validation split (17 examples) is too small to estimate it.
- On 2026-09-22 the same checkpoint was re-exported with int6 weights (one payload character per weight) and biases rounded to 3 decimals, to recover size headroom: the Brotli bundle went from 39,846 bytes to 24,777 bytes (`pnpm size`). It passed the gate against this dataset's recorded floor. A given seed's sprite differs slightly from the int8 export.

## Parity

64 seeds are recorded in `active/parity.json` with the latent, float32 logits computed in numpy in the same operation order as the CPU runtime, the raw thresholded bits and the final sprite. The Node test requires identical latents, logits within 1e-5, identical bits where the logit margin exceeds 1e-5, and identical final sprites. The browser test runs the same 64 seeds in real Chromium and requires finite WebGPU logits within 1e-4 of the CPU, identical bits where the margin exceeds 1e-3, low-margin pixels at most 0.1% of all fixture pixels, a single-seed call within 1e-4 of the batched one, and, for 16 of the seeds, identical final sprites from `generateMany` on both backends. It also destroys the GPU device under an `auto` and an explicit `webgpu` generator: every call must return the CPU's sprites, and every call except the `auto` one right after the loss, which may run on the CPU, must run on the GPU.

Last measured (`pnpm --filter gpu-sprite test:browser`): adapter `amd gcn-4`, 64 seeds, 16,384 pixels, max logit difference 0.00e+0, 6 low-margin pixels, batched vs single-seed difference 0.00e+0, `generateMany` sprites identical between WebGPU and CPU; device-loss recovery with backends webgpu, cpu, webgpu, webgpu, webgpu and sprites identical to the CPU.

## Backend crossover

`backend: "auto"` uses WebGPU from `AUTO_WEBGPU_MIN_BATCH = 64` seeds. `pnpm --filter gpu-sprite bench:browser` times `generateMany` on a CPU generator and on a WebGPU generator in Chromium, one warm-up call and then the median of 15 timed calls per backend and batch size. Both timings include seed → latent and the CPU decode of the logits. The table was measured before the switch to characters and creatures, with the same architecture (32 → 128 → 256) and runtime; only the weights changed, so it was not re-measured. The int6 re-export also changed only the parameter values; decoding the payload happens once per page, outside the timed calls. Measured on a Windows 10 desktop with an Intel Core i5-7500 and an AMD Radeon RX 480 (Chromium adapter `amd gcn-4`):

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

- 1-bit only, 16×16 only, no conditioning on category or text. Characters and creatures only.
- Small dataset: 106 unique tiles from one artist's pack (144 training examples after flips), so diversity is bounded by it. Humanoids dominate the samples; the rarer shapes of the family (animals, snakes) come out less often and less clearly.
- Pixels are independent given the latent; ambiguous regions can look blurry or fragmented. The isolated-pixel cleanup removes single stray pixels but not larger fragments.
- The density check passes close to its lower bound: the sample p5 is 21.48% against a training p1 of 21.26%, so a retrained model that drifts slightly toward lighter sprites can fail it.
- The densest samples are lighter than the densest training sprites (sample p95 35.94% against training p95 41.66%).
- The validation split has only 17 examples, so the reconstruction accuracy and its floor are coarse.
- The coherence check only guards against fragmentation. It passes by a wide margin (64.30% against 32.92%), but it says nothing about whether a sample reads as a character or a creature; that was judged by eye.
- Seeds map to latents deterministically but not meaningfully; neighbouring seeds are unrelated sprites. Use `fromLatent` to interpolate.
- WebGPU requires a secure context and a supported browser. Small batches are faster on the CPU.

## Reproducibility

`pnpm data`, then `train.py --run characters --beta 4 --epochs 600`, and `evaluate.py` and `export.py` with `--checkpoint packages/training/runs/characters/best.pt`, as in the README. Training is seeded (1337) but CPU PyTorch kernels can differ across versions; a re-run should land near the recorded metrics, not on them exactly. `active/manifest.json` records the sha256 of the checkpoint, of the training and validation arrays and of the generated `weights.ts`. Promotion never lowers a dataset's recorded reconstruction floor. Floors are remembered per dataset in `active/report.json` (`floors_by_dataset`); a dataset without a recorded floor starts its own, and the gate output says so (`new-dataset` or `first-promotion`).
