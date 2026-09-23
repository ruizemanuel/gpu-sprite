import test from "node:test";
import assert from "node:assert/strict";
import { TILES } from "../src/game/tiles.ts";

test("every hand-drawn tile is a non-empty 16x16 bitmap with a color", () => {
  for (const [name, tile] of Object.entries(TILES)) {
    assert.equal(tile.bits.length, 256, name);
    assert.ok(tile.bits.some((b) => b === 1), name);
    assert.ok(tile.bits.every((b) => b === 0 || b === 1), name);
    assert.match(tile.color, /^#[0-9a-f]{6}$/, name);
  }
});
