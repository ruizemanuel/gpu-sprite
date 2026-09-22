import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { LATENT_DIM, seedToLatent, toUint32Seed } from "../src/seed.ts";

test("latent has 32 float32 entries and is deterministic", () => {
  const a = seedToLatent(42);
  const b = seedToLatent(42);
  assert.equal(a.length, LATENT_DIM);
  assert.ok(a instanceof Float32Array);
  assert.deepEqual(Array.from(a), Array.from(b));
});

test("different seeds give different latents", () => {
  assert.notDeepEqual(Array.from(seedToLatent(1)), Array.from(seedToLatent(2)));
});

test("seed 0 is remapped and equals the golden constant seed", () => {
  assert.deepEqual(Array.from(seedToLatent(0)), Array.from(seedToLatent(0x9e3779b9)));
});

test("seeds are reduced modulo 2^32", () => {
  assert.equal(toUint32Seed(-1), 4294967295);
  assert.equal(toUint32Seed(2 ** 32 + 5), 5);
  assert.deepEqual(Array.from(seedToLatent(-1)), Array.from(seedToLatent(4294967295)));
});

test("non-integer seeds throw TypeError", () => {
  assert.throws(() => seedToLatent(1.5), TypeError);
  assert.throws(() => seedToLatent(Number.NaN), TypeError);
  assert.throws(() => seedToLatent(2 ** 53), TypeError);
});

test("values are roughly standard normal over many seeds", () => {
  let sum = 0;
  let sumSq = 0;
  const n = 2000;
  for (let s = 1; s <= n; s++) {
    const z = seedToLatent(s);
    for (const v of z) {
      sum += v;
      sumSq += v * v;
    }
  }
  const count = n * LATENT_DIM;
  const mean = sum / count;
  const variance = sumSq / count - mean * mean;
  assert.ok(Math.abs(mean) < 0.02, `mean ${mean}`);
  assert.ok(Math.abs(variance - 1) < 0.05, `variance ${variance}`);
});

test("matches the committed seed-vectors fixture", () => {
  const fixture = JSON.parse(readFileSync(new URL("./fixtures/seed-vectors.json", import.meta.url), "utf8"));
  assert.equal(fixture.seedVersion, 1);
  for (const { seed, z } of fixture.vectors) {
    assert.deepEqual(Array.from(seedToLatent(seed)), z, `seed ${seed}`);
  }
});
