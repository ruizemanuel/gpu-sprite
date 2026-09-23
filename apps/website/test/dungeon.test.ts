import test from "node:test";
import assert from "node:assert/strict";
import { BOSS, FLOOR_GROUPS } from "../src/game/bestiary.ts";
import { center, FLOORS, generateFloor, isFloor, MAP_H, MAP_W, type Floor, type Point } from "../src/game/dungeon.ts";

const SEEDS = [0, 1, 42, 1234, 99999, 0xffffffff];

/** Tiles reachable from `from` through floor tiles, 4-connectivity. */
function reachable(floor: Floor, from: Point): Set<number> {
  const seen = new Set([from.y * MAP_W + from.x]);
  const queue = [from];
  while (queue.length > 0) {
    const p = queue.shift() as Point;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const x = p.x + dx;
      const y = p.y + dy;
      const key = y * MAP_W + x;
      if (isFloor(floor, x, y) && !seen.has(key)) {
        seen.add(key);
        queue.push({ x, y });
      }
    }
  }
  return seen;
}

const inRoom = (p: Point, r: { x: number; y: number; w: number; h: number }) =>
  p.x >= r.x && p.x < r.x + r.w && p.y >= r.y && p.y < r.y + r.h;

test("floors are deterministic per run seed and depth, and differ across both", () => {
  assert.deepEqual(generateFloor(42, 1), generateFloor(42, 1));
  assert.notDeepEqual(generateFloor(42, 1).tiles, generateFloor(42, 2).tiles);
  assert.notDeepEqual(generateFloor(42, 1).tiles, generateFloor(43, 1).tiles);
});

test("every floor has 5 to 7 rooms inside the border and the map's outer ring is wall", () => {
  for (const seed of SEEDS) {
    for (let depth = 1; depth <= FLOORS; depth++) {
      const floor = generateFloor(seed, depth);
      assert.ok(floor.rooms.length >= 5 && floor.rooms.length <= 7, `seed ${seed} depth ${depth}`);
      for (let x = 0; x < MAP_W; x++) assert.ok(!isFloor(floor, x, 0) && !isFloor(floor, x, MAP_H - 1));
      for (let y = 0; y < MAP_H; y++) assert.ok(!isFloor(floor, 0, y) && !isFloor(floor, MAP_W - 1, y));
    }
  }
});

test("every room, the stairs and every spawn are reachable from the start", () => {
  for (const seed of SEEDS) {
    for (let depth = 1; depth <= FLOORS; depth++) {
      const floor = generateFloor(seed, depth);
      const reach = reachable(floor, floor.start);
      const key = (p: Point) => p.y * MAP_W + p.x;
      for (const room of floor.rooms) assert.ok(reach.has(key(center(room))));
      assert.ok(reach.has(key(floor.stairs)));
      for (const e of floor.enemies) assert.ok(reach.has(key(e.at)));
      for (const p of floor.potions) assert.ok(reach.has(key(p)));
    }
  }
});

test("start and stairs sit at the centers of the first and last rooms", () => {
  const floor = generateFloor(42, 1);
  assert.deepEqual(floor.start, center(floor.rooms[0]));
  assert.deepEqual(floor.stairs, center(floor.rooms[floor.rooms.length - 1]));
});

test("enemies and potions sit on distinct floor tiles outside the first room, never on start or stairs", () => {
  for (const seed of SEEDS) {
    for (let depth = 1; depth <= FLOORS; depth++) {
      const floor = generateFloor(seed, depth);
      const spots = [...floor.enemies.map((e) => e.at), ...floor.potions];
      const keys = spots.map((p) => p.y * MAP_W + p.x);
      assert.equal(new Set(keys).size, keys.length);
      for (const p of spots) {
        assert.ok(isFloor(floor, p.x, p.y));
        assert.ok(!inRoom(p, floor.rooms[0]));
        assert.notDeepEqual(p, floor.start);
        assert.notDeepEqual(p, floor.stairs);
      }
      assert.ok(floor.potions.length >= 1 && floor.potions.length <= 2);
    }
  }
});

test("enemy species and counts follow the floor groups; the boss guards the last room of the last floor", () => {
  for (const seed of SEEDS) {
    for (let depth = 1; depth < FLOORS; depth++) {
      const floor = generateFloor(seed, depth);
      assert.ok(floor.enemies.length >= 3 && floor.enemies.length <= 5);
      for (const e of floor.enemies) assert.ok(FLOOR_GROUPS[depth - 1].includes(e.species));
    }
    const last = generateFloor(seed, FLOORS);
    const bosses = last.enemies.filter((e) => e.species === BOSS);
    assert.equal(bosses.length, 1);
    assert.ok(inRoom(bosses[0].at, last.rooms[last.rooms.length - 1]));
    const others = last.enemies.filter((e) => e.species !== BOSS);
    assert.ok(others.length >= 2 && others.length <= 3);
    for (const e of others) assert.ok(FLOOR_GROUPS[FLOORS - 1].includes(e.species));
  }
});
