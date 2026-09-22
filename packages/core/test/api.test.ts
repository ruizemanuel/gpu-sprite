import test from "node:test";
import assert from "node:assert/strict";
import { AUTO_WEBGPU_MIN_BATCH, defineGenerator, generate, LATENT_DIM, MAX_BATCH } from "../src/index.ts";

test("generate(seed) returns a 16x16 sprite on CPU in Node", async () => {
  const sprite = await generate(42);
  assert.equal(sprite.width, 16);
  assert.equal(sprite.height, 16);
  assert.equal(sprite.pixels.length, 256);
  assert.ok(sprite.pixels.every((p) => p === 0 || p === 1));
});

test("generate is deterministic per seed", async () => {
  const a = await generate(7);
  const b = await generate(7);
  const c = await generate(8);
  assert.deepEqual(Array.from(a.pixels), Array.from(b.pixels));
  assert.notDeepEqual(Array.from(a.pixels), Array.from(c.pixels));
});

test("generate(seeds[]) returns one sprite per seed in order", async () => {
  const sprites = await generate([1, 2, 3]);
  assert.equal(sprites.length, 3);
  const single = await generate(2);
  assert.deepEqual(Array.from(sprites[1].pixels), Array.from(single.pixels));
});

test("generateMany reports the backend actually used", async () => {
  const g = defineGenerator({ backend: "auto" });
  const result = await g.generateMany(Array.from({ length: AUTO_WEBGPU_MIN_BATCH }, (_, i) => i));
  assert.equal(result.backend, "cpu"); // no navigator.gpu in Node, auto falls back to CPU
  assert.equal(result.sprites.length, AUTO_WEBGPU_MIN_BATCH);
  g.dispose();
});

test("explicit webgpu without support rejects, never falls back", async () => {
  const g = defineGenerator({ backend: "webgpu" });
  await assert.rejects(() => g.generate(1), /WebGPU/);
  g.dispose();
});

test("fromLatent accepts a 32-float latent and rejects other lengths", async () => {
  const g = defineGenerator({ backend: "cpu" });
  const sprite = await g.fromLatent(new Float32Array(LATENT_DIM));
  assert.equal(sprite.pixels.length, 256);
  await assert.rejects(() => g.fromLatent(new Float32Array(3)), RangeError);
  g.dispose();
});

test("argument validation", async () => {
  await assert.rejects(() => generate(1.5), TypeError);
  await assert.rejects(() => generate([1, Number.NaN]), TypeError);
  await assert.rejects(() => generate(Array.from({ length: MAX_BATCH + 1 }, (_, i) => i)), RangeError);
  assert.deepEqual(await generate([]), []);
});

test("dispose makes the generator unusable", async () => {
  const g = defineGenerator();
  g.dispose();
  await assert.rejects(() => g.generate(1), /disposed/);
});
