import test from "node:test";
import assert from "node:assert/strict";
import { defineGenerator } from "../../../packages/core/src/index.ts";
import { CANDIDATES_PER_BATCH, candidateSeeds, MAX_BATCHES, passesFilter } from "../src/game/bestiary.ts";
import { generateBestiary, type GeneratedBatch } from "../src/game/run.ts";
import { spriteMetrics } from "../src/game/sprite-metrics.ts";
import { spriteWithFg } from "./helpers.ts";

const emptyBatch = (seeds: number[]): GeneratedBatch => ({ sprites: seeds.map(() => ({ pixels: new Uint8Array(256) })), backend: "fake" });
/** Every sprite distinct and passing the filter: 52 + (i mod 60) foreground pixels. */
const passingBatch = (seeds: number[]): GeneratedBatch => ({ sprites: seeds.map((_, i) => ({ pixels: spriteWithFg(52 + (i % 60)) })), backend: "fake", adapter: "fake adapter" });

function clock(): () => number {
  let t = 0;
  return () => (t += 5);
}

test("end to end: the real CPU generator builds a bestiary from the first batch", async () => {
  const generator = defineGenerator({ backend: "cpu" });
  const calls: number[][] = [];
  const run = await generateBestiary(42, (seeds) => (calls.push(seeds), generator.generateMany(seeds)), () => performance.now(), () => true);
  generator.dispose();
  assert.ok(run?.bestiary);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0], candidateSeeds(42, CANDIDATES_PER_BATCH));
  assert.deepEqual([run.candidates, run.backend], [CANDIDATES_PER_BATCH, "cpu"]);
  assert.equal(run.bestiary.species.length, 8);
  for (const s of [run.bestiary.player, ...run.bestiary.species]) assert.ok(passesFilter(spriteMetrics(s.pixels)));
});

test("batches continue the seed stream until the bestiary is complete", async () => {
  const calls: number[][] = [];
  const run = await generateBestiary(7, async (seeds) => (calls.push(seeds), calls.length === 1 ? emptyBatch(seeds) : passingBatch(seeds)), clock(), () => true);
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[1], candidateSeeds(7, 2 * CANDIDATES_PER_BATCH).slice(CANDIDATES_PER_BATCH));
  assert.ok(run?.bestiary);
  assert.deepEqual([run.candidates, run.ms, run.backend, run.adapter], [2 * CANDIDATES_PER_BATCH, 10, "fake", "fake adapter"]);
});

test("after MAX_BATCHES batches without a bestiary the run reports none", async () => {
  let calls = 0;
  const run = await generateBestiary(7, async (seeds) => (calls++, emptyBatch(seeds)), clock(), () => true);
  assert.equal(calls, MAX_BATCHES);
  assert.ok(run);
  assert.equal(run.bestiary, undefined);
  assert.equal(run.candidates, MAX_BATCHES * CANDIDATES_PER_BATCH);
});

test("a run replaced during generation stops and reports nothing", async () => {
  let calls = 0;
  let current = true;
  const run = await generateBestiary(7, async (seeds) => {
    calls++;
    current = false; // a newer run started while this batch was generating
    return emptyBatch(seeds);
  }, clock(), () => current);
  assert.equal(run, undefined);
  assert.equal(calls, 1);
});
