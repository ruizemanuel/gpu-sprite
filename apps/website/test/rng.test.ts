import test from "node:test";
import assert from "node:assert/strict";
import { deriveSeed, mulberry32, parseRunSeed } from "../src/game/rng.ts";

const draws = (seed: number, n: number) => {
  const rng = mulberry32(seed);
  return Array.from({ length: n }, () => rng.next());
};

test("mulberry32 is deterministic and pinned", () => {
  assert.deepEqual(draws(42, 5), draws(42, 5));
  assert.notDeepEqual(draws(42, 5), draws(43, 5));
  // Pinned so a change to the generator, which would change every run, fails loudly.
  assert.deepEqual(draws(42, 3), [2581720956, 1925393290, 3661312704]);
});

test("mulberry32 yields uint32 and int() stays within its inclusive bounds", () => {
  const rng = mulberry32(7);
  const seen = new Set<number>();
  for (let i = 0; i < 2000; i++) {
    const u = rng.next();
    assert.ok(Number.isInteger(u) && u >= 0 && u <= 0xffffffff);
    const v = rng.int(3, 6);
    assert.ok(v >= 3 && v <= 6);
    seen.add(v);
  }
  assert.deepEqual([...seen].sort(), [3, 4, 5, 6]);
});

test("deriveSeed separates streams by label and by run seed", () => {
  assert.equal(deriveSeed(42, "bestiary"), deriveSeed(42, "bestiary"));
  assert.notEqual(deriveSeed(42, "bestiary"), deriveSeed(42, "floor-1"));
  assert.notEqual(deriveSeed(42, "floor-1"), deriveSeed(42, "floor-2"));
  assert.notEqual(deriveSeed(42, "floor-1"), deriveSeed(43, "floor-1"));
  for (const seed of [0, 1, 42, 0xffffffff]) {
    const d = deriveSeed(seed, "bestiary");
    assert.ok(Number.isInteger(d) && d >= 0 && d <= 0xffffffff);
  }
});

test("parseRunSeed accepts decimal uint32 and rejects everything else", () => {
  assert.equal(parseRunSeed("0"), 0);
  assert.equal(parseRunSeed("42"), 42);
  assert.equal(parseRunSeed("007"), 7);
  assert.equal(parseRunSeed("4294967295"), 4294967295);
  for (const bad of ["", "4294967296", "99999999999", "-1", "1.5", "1e3", " 42", "42 ", "42x", "abc", "0x10"]) {
    assert.equal(parseRunSeed(bad), undefined, JSON.stringify(bad));
  }
});
