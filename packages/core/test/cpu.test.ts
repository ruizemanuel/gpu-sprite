import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { decodeModel } from "../src/model/decode.ts";
import { CpuModel } from "../src/model/cpu.ts";
import { model } from "../src/model/weights.ts";
import type { ModelSpec } from "../src/model/types.ts";
import { despeckle, hexBitsToPixels, logitsToSprite, SPRITE_PIXELS } from "../src/sprite.ts";
import { LATENT_DIM, seedToLatent } from "../src/seed.ts";

function toBase64(bytes: Int8Array): string {
  return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString("base64");
}

// First layer has LATENT_DIM inputs (decodeModel requires layers[0].in === spec.latent === LATENT_DIM),
// but only the first two input columns are non-zero, so the hand-computed math below only ever depends
// on inputs 0 and 1, exactly like a genuine 2-input toy model would.
const layer0Weights = new Int8Array(LATENT_DIM * 2); // 2 outputs x LATENT_DIM inputs, row-major
layer0Weights[0] = 2; // row 0 (output 0), col 0
layer0Weights[LATENT_DIM + 1] = 2; // row 1 (output 1), col 1
const layer1Weights = new Int8Array([1, 0, 0, 1, 1, 1]); // rows: [1, 0], [0, 1], [1, 1]
const tinyWeights = new Int8Array(layer0Weights.length + layer1Weights.length);
tinyWeights.set(layer0Weights, 0);
tinyWeights.set(layer1Weights, layer0Weights.length);

/** LATENT_DIM inputs (only the first two matter) → 2 hidden (relu) → 3 outputs, weights chosen so the math is easy to check by hand. */
const tiny: ModelSpec = {
  format: 1,
  seedVersion: 1,
  latent: LATENT_DIM,
  checkpoint: "tiny",
  layers: [
    { in: LATENT_DIM, out: 2, activation: "relu", scale: 0.5, bias: [0, -1] },
    { in: 2, out: 3, activation: "none", scale: 1, bias: [0.25, 0, -0.25] },
  ],
  weights: toBase64(tinyWeights),
};

/** Builds a LATENT_DIM-wide input row with `a` at index 0, `b` at index 1 and zeros elsewhere. */
function tinyInput(a: number, b: number): number[] {
  const row = new Array(LATENT_DIM).fill(0);
  row[0] = a;
  row[1] = b;
  return row;
}

test("decodeModel dequantizes int8 weights with the per-layer scale", () => {
  const layers = decodeModel(tiny);
  assert.equal(layers.length, 2);
  assert.equal(layers[0].weights.length, LATENT_DIM * 2);
  assert.equal(layers[0].weights[0], 1);
  assert.equal(layers[0].weights[LATENT_DIM + 1], 1);
  assert.ok(Array.from(layers[0].weights).every((w, i) => i === 0 || i === LATENT_DIM + 1 || w === 0));
  assert.deepEqual(Array.from(layers[1].weights), [1, 0, 0, 1, 1, 1]);
  assert.deepEqual(Array.from(layers[0].bias), [0, -1]);
  assert.equal(layers[0].relu, true);
  assert.equal(layers[1].relu, false);
});

test("decodeModel rejects a payload whose length does not match the layers", () => {
  assert.throws(() => decodeModel({ ...tiny, weights: toBase64(new Int8Array([1, 2, 3])) }), /weights payload/);
});

test("decodeModel rejects a spec with a mismatched latent", () => {
  assert.throws(() => decodeModel({ ...tiny, latent: 16 }), /latent/);
});

test("CpuModel.forward computes the MLP with relu and bias for a batch", () => {
  const cpu = new CpuModel(decodeModel(tiny));
  // sample 0: x=[3, 0.5, 0...] → h=[3, max(0.5-1,0)=0] → out=[3.25, 0, 2.75]
  // sample 1: x=[-1, 4, 0...]  → h=[0, 3]              → out=[0.25, 3, 2.75]
  const out = cpu.forward(new Float32Array([...tinyInput(3, 0.5), ...tinyInput(-1, 4)]), 2);
  assert.deepEqual(Array.from(out), [3.25, 0, 2.75, 0.25, 3, 2.75]);
});

test("logitsToSprite thresholds at zero, despeckles, and reads from an offset", () => {
  const logits = new Float32Array(SPRITE_PIXELS * 2);
  logits.fill(-1);
  // First sprite: pin the `> 0` boundary. Pixel 100 has an exact-zero logit and a
  // foreground neighbour at 101 (100 and 101 are horizontally adjacent). With the
  // correct `> 0` threshold, pixel 100 is background from the start regardless of
  // despeckle; with an (incorrect) `>= 0` threshold it would be foreground and kept,
  // since 101 is foreground.
  logits[100] = 0;
  logits[101] = 3;
  logits[102] = 3;
  // Second sprite: a horizontal pair at pixels 5,6 (kept) and an isolated pixel at 255 (cleared).
  logits[SPRITE_PIXELS + 5] = 0.001;
  logits[SPRITE_PIXELS + 6] = 3;
  logits[SPRITE_PIXELS + 255] = 3;

  const first = logitsToSprite(logits, 0);
  assert.equal(first.pixels[100], 0, "a logit of exactly 0 is background");
  assert.equal(first.pixels[101], 1);
  assert.equal(first.pixels[102], 1);
  assert.equal(first.pixels.reduce((a, b) => a + b, 0), 2);

  const sprite = logitsToSprite(logits, SPRITE_PIXELS);
  assert.equal(sprite.width, 16);
  assert.equal(sprite.pixels.length, 256);
  assert.equal(sprite.pixels[5], 1);
  assert.equal(sprite.pixels[6], 1);
  assert.equal(sprite.pixels[255], 0);
  assert.equal(sprite.pixels.reduce((a, b) => a + b, 0), 2);
});

test("despeckle clears an isolated foreground pixel", () => {
  const px = new Uint8Array(SPRITE_PIXELS);
  px[7 * 16 + 7] = 1;
  const out = despeckle(px);
  assert.equal(out.reduce((a, b) => a + b, 0), 0);
});

test("despeckle keeps horizontal, vertical and diagonal pairs", () => {
  const pairs: Array<[[number, number], [number, number]]> = [
    [[5, 5], [5, 6]],
    [[5, 5], [6, 5]],
    [[5, 5], [6, 6]],
    [[5, 6], [6, 5]],
  ];
  for (const pair of pairs) {
    const px = new Uint8Array(SPRITE_PIXELS);
    for (const [y, x] of pair) px[y * 16 + x] = 1;
    assert.deepEqual(Array.from(despeckle(px)), Array.from(px), JSON.stringify(pair));
  }
});

test("despeckle clears isolated corner and edge pixels", () => {
  const points: Array<[number, number]> = [
    [0, 0],
    [0, 15],
    [15, 0],
    [15, 15],
    [0, 7],
    [7, 0],
    [15, 7],
    [7, 15],
  ];
  for (const [y, x] of points) {
    const px = new Uint8Array(SPRITE_PIXELS);
    px[y * 16 + x] = 1;
    assert.equal(despeckle(px).reduce((a, b) => a + b, 0), 0, `(${y}, ${x})`);
  }
});

test("despeckle does not wrap around row edges", () => {
  // Pixel 15 (row 0, col 15) and pixel 16 (row 1, col 0) are adjacent in the flat array
  // but not neighbours on the grid; both must be cleared when alone together.
  const px = new Uint8Array(SPRITE_PIXELS);
  px[15] = 1;
  px[16] = 1;
  assert.equal(despeckle(px).reduce((a, b) => a + b, 0), 0);
});

test("despeckle is a single pass: clearing an isolated pixel does not cascade", () => {
  const px = new Uint8Array(SPRITE_PIXELS);
  px[3 * 16 + 3] = 1;
  px[3 * 16 + 4] = 1;
  px[10 * 16 + 10] = 1;
  const expected = new Uint8Array(SPRITE_PIXELS);
  expected[3 * 16 + 3] = 1;
  expected[3 * 16 + 4] = 1;
  assert.deepEqual(Array.from(despeckle(px)), Array.from(expected));
});

test("hexBitsToPixels unpacks MSB-first like numpy packbits", () => {
  const px = hexBitsToPixels("80" + "00".repeat(30) + "01");
  assert.equal(px.length, 256);
  assert.equal(px[0], 1);
  assert.equal(px[255], 1);
  assert.equal(px.reduce((a, b) => a + b, 0), 2);
});

test("shipped model matches the Python parity fixture", () => {
  const parity = JSON.parse(readFileSync(new URL("../../training/active/parity.json", import.meta.url), "utf8"));
  assert.equal(parity.seedVersion, model.seedVersion);
  assert.equal(parity.checkpoint, model.checkpoint);
  const cpu = new CpuModel(decodeModel(model));
  let lowMargin = 0;
  for (const entry of parity.entries) {
    const z = seedToLatent(entry.seed);
    assert.deepEqual(Array.from(z), entry.z, `latent for seed ${entry.seed}`);
    const logits = cpu.forward(z, 1);
    const expected = hexBitsToPixels(entry.bits);
    let maxDiff = 0;
    for (let i = 0; i < SPRITE_PIXELS; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(logits[i] - entry.logits[i]));
      if (Math.abs(entry.logits[i]) > 1e-5) {
        assert.equal(logits[i] > 0 ? 1 : 0, expected[i], `seed ${entry.seed} pixel ${i}`);
      } else {
        lowMargin++;
      }
    }
    assert.ok(maxDiff < 1e-5, `seed ${entry.seed} max logit diff ${maxDiff}`);
    assert.deepEqual(
      Array.from(logitsToSprite(logits).pixels),
      Array.from(hexBitsToPixels(entry.sprite)),
      `seed ${entry.seed} final sprite`,
    );
  }
  assert.ok(lowMargin <= parity.entries.length * SPRITE_PIXELS * 0.001, `low-margin pixels: ${lowMargin}`);
});
