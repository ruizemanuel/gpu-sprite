import test from "node:test";
import assert from "node:assert/strict";
import {
  atkFor,
  buildBestiary,
  CANDIDATES_PER_BATCH,
  candidateSeeds,
  colorFor,
  hpFor,
  nameFor,
  passesFilter,
  type Candidate,
} from "../src/game/bestiary.ts";
import { spriteMetrics } from "../src/game/sprite-metrics.ts";
import { blockSprite, spriteWithFg } from "./helpers.ts";

test("hp and atk formulas at their edges", () => {
  assert.deepEqual([52, 60, 61, 115].map(hpFor), [5, 5, 6, 10]);
  assert.deepEqual([9, 10, 13, 14].map(atkFor), [1, 2, 2, 3]);
});

test("the filter keeps 1-2 components with 52 to 115 foreground pixels", () => {
  assert.equal(passesFilter({ fg: 52, components: 1, width: 10 }), true);
  assert.equal(passesFilter({ fg: 115, components: 2, width: 10 }), true);
  assert.equal(passesFilter({ fg: 51, components: 1, width: 10 }), false);
  assert.equal(passesFilter({ fg: 116, components: 1, width: 10 }), false);
  assert.equal(passesFilter({ fg: 80, components: 3, width: 10 }), false);
});

test("candidate seeds are one stream: later batches extend earlier ones", () => {
  const first = candidateSeeds(42, CANDIDATES_PER_BATCH);
  const two = candidateSeeds(42, 2 * CANDIDATES_PER_BATCH);
  assert.deepEqual(two.slice(0, CANDIDATES_PER_BATCH), first);
  assert.notDeepEqual(candidateSeeds(43, 8), candidateSeeds(42, 8));
});

test("buildBestiary skips failing and duplicate sprites, sorts species by foreground and doubles the boss", () => {
  // Three separate 4x6 blocks: 72 pixels, inside the density range, but fragmented.
  const threeComponents = blockSprite(0, 0, 4, 6);
  for (const x of [6, 12]) blockSprite(x, 0, 4, 6).forEach((on, i) => (threeComponents[i] |= on));
  const candidates: Candidate[] = [
    { seed: 1, pixels: spriteWithFg(51) }, // too sparse
    { seed: 2, pixels: spriteWithFg(116) }, // too dense
    { seed: 3, pixels: threeComponents }, // fragmented
    { seed: 10, pixels: spriteWithFg(60) }, // player
    { seed: 11, pixels: spriteWithFg(100) },
    { seed: 12, pixels: spriteWithFg(100) }, // duplicate of seed 11
    { seed: 13, pixels: spriteWithFg(70) },
    { seed: 14, pixels: spriteWithFg(70, 12) }, // same fg as 13, later: sorts after it
    { seed: 15, pixels: spriteWithFg(52) },
    { seed: 16, pixels: spriteWithFg(90) },
    { seed: 17, pixels: spriteWithFg(115, 14) },
    { seed: 18, pixels: spriteWithFg(80) },
    { seed: 19, pixels: spriteWithFg(60, 8) },
    { seed: 20, pixels: spriteWithFg(75) }, // never reached: 9 already accepted
  ];
  assert.equal(spriteMetrics(threeComponents).components, 3);
  const bestiary = buildBestiary(candidates);
  assert.ok(bestiary);
  assert.equal(bestiary.player.seed, 10);
  assert.deepEqual(
    bestiary.species.map((s) => s.seed),
    [15, 19, 13, 14, 18, 16, 11, 17],
  );
  assert.deepEqual(
    bestiary.species.map((s) => s.fg),
    [52, 60, 70, 70, 80, 90, 100, 115],
  );
  const boss = bestiary.species[7];
  assert.equal(boss.boss, true);
  assert.equal(boss.hp, 2 * hpFor(115));
  assert.equal(boss.atk, atkFor(14));
  assert.ok(bestiary.species.slice(0, 7).every((s) => !s.boss && s.hp === hpFor(s.fg)));
  assert.equal(new Set(bestiary.species.map((s) => s.name)).size, 8);
  assert.equal(bestiary.species[0].color, colorFor(15));
});

test("buildBestiary returns undefined when fewer than 9 candidates pass", () => {
  const candidates = Array.from({ length: 8 }, (_, i) => ({ seed: i, pixels: spriteWithFg(52 + i) }));
  assert.equal(buildBestiary(candidates), undefined);
});

test("nameFor is deterministic and avoids taken names", () => {
  const name = nameFor(123, new Set());
  assert.equal(nameFor(123, new Set()), name);
  assert.match(name, /^[A-Z][a-z]+$/);
  assert.notEqual(nameFor(123, new Set([name])), name);
});
