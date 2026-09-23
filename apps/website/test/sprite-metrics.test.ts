import test from "node:test";
import assert from "node:assert/strict";
import { spriteMetrics } from "../src/game/sprite-metrics.ts";
import { blockSprite, spriteFrom } from "./helpers.ts";

const EMPTY_ROW = "................";

test("an empty sprite has no foreground, no components and no width", () => {
  assert.deepEqual(spriteMetrics(new Uint8Array(256)), { fg: 0, components: 0, width: 0 });
});

test("a solid block is one component; width spans its columns", () => {
  assert.deepEqual(spriteMetrics(blockSprite(3, 2, 10, 6)), { fg: 60, components: 1, width: 10 });
});

test("diagonal neighbours join one component (8-connectivity)", () => {
  const rows = Array(16).fill(EMPTY_ROW);
  rows[0] = "#...............";
  rows[1] = ".#..............";
  rows[2] = "..#.............";
  assert.deepEqual(spriteMetrics(spriteFrom(rows)), { fg: 3, components: 1, width: 3 });
});

test("pixels two apart are separate components, and width counts the gap", () => {
  const rows = Array(16).fill(EMPTY_ROW);
  rows[5] = "##.##.........##";
  assert.deepEqual(spriteMetrics(spriteFrom(rows)), { fg: 6, components: 3, width: 16 });
});

test("the wrong pixel count is rejected", () => {
  assert.throws(() => spriteMetrics(new Uint8Array(255)), RangeError);
});
