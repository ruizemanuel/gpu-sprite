import test from "node:test";
import assert from "node:assert/strict";
import type { Bestiary, Species } from "../src/game/bestiary.ts";
import { generateFloor, MAP_H, MAP_W, type Floor, type Point } from "../src/game/dungeon.ts";
import { newGame, PLAYER_MAX_HP, step, type Action, type Enemy, type GameState } from "../src/game/rules.ts";

/** Every tile inside the border is floor; start top-left, stairs bottom-right. */
function openFloor(depth = 1): Floor {
  const tiles = new Uint8Array(MAP_W * MAP_H);
  for (let y = 1; y < MAP_H - 1; y++) for (let x = 1; x < MAP_W - 1; x++) tiles[y * MAP_W + x] = 1;
  return { depth, tiles, rooms: [{ x: 1, y: 1, w: MAP_W - 2, h: MAP_H - 2 }], start: { x: 1, y: 1 }, stairs: { x: MAP_W - 2, y: MAP_H - 2 }, enemies: [], potions: [] };
}

/** Species 0-6 have 6 hp and 2 atk; species 7 is the boss with 12 hp and 3 atk. */
function testBestiary(): Bestiary {
  const species = Array.from({ length: 8 }, (_, id): Species => ({
    seed: id,
    pixels: new Uint8Array(256),
    name: `Beast${id}`,
    color: "#fff",
    fg: 60,
    width: 10,
    hp: id === 7 ? 12 : 6,
    atk: id === 7 ? 3 : 2,
    boss: id === 7,
  }));
  return { player: { seed: 99, pixels: new Uint8Array(256) }, species };
}

function stateAt(player: Point, parts: Partial<GameState> = {}): GameState {
  return {
    runSeed: 42,
    bestiary: testBestiary(),
    depth: 1,
    floor: openFloor(),
    player: { ...player, hp: PLAYER_MAX_HP },
    enemies: [],
    potions: [],
    turn: 0,
    explored: new Uint8Array(MAP_W * MAP_H),
    discovered: Array(8).fill(false),
    log: [],
    result: "playing",
    ...parts,
  };
}

const enemy = (x: number, y: number, species = 0, hp = 6): Enemy => ({ species, hp, x, y });
const WAIT: Action = { type: "wait" };
const move = (dir: "up" | "down" | "left" | "right"): Action => ({ type: "move", dir });

test("walking into a wall costs no turn and returns the same state", () => {
  const s = stateAt({ x: 1, y: 1 });
  assert.equal(step(s, move("left")), s);
  assert.equal(step(s, move("up")), s);
});

test("moving to a free tile takes a turn", () => {
  const next = step(stateAt({ x: 1, y: 1 }), move("right"));
  assert.deepEqual([next.player.x, next.player.y, next.turn], [2, 1, 1]);
});

test("step never changes the state it is given", () => {
  const s = stateAt({ x: 5, y: 5 }, { enemies: [enemy(6, 5)], potions: [{ x: 4, y: 5 }] });
  const before = structuredClone(s);
  step(s, move("right"));
  step(s, move("left"));
  step(s, WAIT);
  assert.deepEqual(s, before);
});

test("bumping an enemy attacks it; it hits back while adjacent and dies at 0 hp", () => {
  let s = stateAt({ x: 1, y: 1 }, { enemies: [enemy(2, 1)] });
  s = step(s, move("right"));
  assert.deepEqual([s.player.x, s.enemies[0].hp, s.player.hp], [1, 3, 38]);
  assert.deepEqual(s.log, ["You hit Beast0 for 3.", "Beast0 hits you for 2."]);
  s = step(s, move("right"));
  assert.equal(s.enemies.length, 0);
  assert.equal(s.log.at(-1), "You kill Beast0.");
  assert.equal(s.player.hp, 38);
});

test("enemies within sight (4 tiles) step toward the player; farther ones wait", () => {
  const s = step(stateAt({ x: 1, y: 1 }, { enemies: [enemy(5, 1), enemy(1, 5), enemy(6, 1, 1), enemy(5, 5, 2), enemy(6, 5, 3)] }), WAIT);
  assert.deepEqual(s.enemies.map((e) => [e.x, e.y]), [[4, 1], [1, 4], [6, 1], [4, 5], [6, 5]]);
});

test("an enemy moves along the longer axis first, horizontally on a tie", () => {
  const s = step(stateAt({ x: 1, y: 1 }, { enemies: [enemy(3, 3), enemy(2, 5)] }), WAIT);
  assert.deepEqual(s.enemies.map((e) => [e.x, e.y]), [[2, 3], [2, 4]]);
});

test("an enemy whose moves are all blocked stays put", () => {
  const floor = openFloor();
  floor.tiles[2 * MAP_W + 1] = 0; // wall right below the player
  const s = step(stateAt({ x: 1, y: 1 }, { floor, enemies: [enemy(1, 3)] }), WAIT);
  assert.deepEqual([s.enemies[0].x, s.enemies[0].y], [1, 3]);
});

test("enemies do not walk into each other", () => {
  const s = step(stateAt({ x: 1, y: 1 }, { enemies: [enemy(3, 1), enemy(4, 1)] }), WAIT);
  assert.deepEqual(s.enemies.map((e) => [e.x, e.y]), [[2, 1], [3, 1]]);
});

test("a potion heals 15, up to the maximum, and is used up", () => {
  const low = step(stateAt({ x: 1, y: 1 }, { player: { x: 1, y: 1, hp: 10 }, potions: [{ x: 2, y: 1 }] }), move("right"));
  assert.equal(low.player.hp, 25);
  assert.equal(low.log.at(-1), "You drink a potion (+15).");
  const high = step(stateAt({ x: 1, y: 1 }, { player: { x: 1, y: 1, hp: 30 }, potions: [{ x: 2, y: 1 }] }), move("right"));
  assert.equal(high.player.hp, PLAYER_MAX_HP);
  assert.equal(high.potions.length, 0);
  assert.equal(high.log.at(-1), "You drink a potion (+10).");
});

test("stairs lead to the next floor, keeping the player's hp", () => {
  const floor = { ...openFloor(), stairs: { x: 2, y: 1 } };
  const s = step(stateAt({ x: 1, y: 1 }, { floor, player: { x: 1, y: 1, hp: 13 } }), move("right"));
  const next = generateFloor(42, 2);
  assert.equal(s.depth, 2);
  assert.equal(s.player.hp, 13);
  assert.deepEqual([s.player.x, s.player.y], [next.start.x, next.start.y]);
  assert.equal(s.enemies.length, next.enemies.length);
  assert.equal(s.turn, 1);
  assert.equal(s.log.at(-1), "You descend to floor 2.");
});

test("an enemy on the stairs blocks them: moving there attacks it instead of descending", () => {
  const floor = { ...openFloor(), stairs: { x: 2, y: 1 } };
  const s = step(stateAt({ x: 1, y: 1 }, { floor, enemies: [enemy(2, 1)] }), move("right"));
  assert.equal(s.depth, 1);
  assert.deepEqual([s.player.x, s.player.y, s.enemies[0].hp], [1, 1, 3]);
});

test("the exit of the last floor wins; nothing changes afterwards", () => {
  const floor = { ...openFloor(3), stairs: { x: 2, y: 1 } };
  const s = step(stateAt({ x: 1, y: 1 }, { depth: 3, floor }), move("right"));
  assert.equal(s.result, "won");
  assert.equal(s.log.at(-1), "You escape the dungeon!");
  assert.equal(step(s, WAIT), s);
  assert.equal(step(s, move("left")), s);
});

test("hp at 0 loses the run", () => {
  const s = step(stateAt({ x: 1, y: 1 }, { player: { x: 1, y: 1, hp: 2 }, enemies: [enemy(2, 1)] }), WAIT);
  assert.equal(s.result, "died");
  assert.equal(s.log.at(-1), "You die on floor 1.");
  assert.equal(step(s, WAIT), s);
});

test("tiles within 4 are explored and enemies within 4 are discovered", () => {
  const s = step(stateAt({ x: 10, y: 10 }, { enemies: [enemy(14, 10, 2), enemy(15, 10, 3)] }), WAIT);
  assert.equal(s.explored[14 * MAP_W + 14], 1);
  assert.equal(s.explored[10 * MAP_W + 15], 0);
  assert.equal(s.discovered[2], true); // 4 away: in sight, steps to 13
  assert.equal(s.discovered[3], false); // 5 away: out of sight, does not move
});

test("newGame starts on floor 1 with full hp, the floor's enemies and the start area explored", () => {
  const bestiary = testBestiary();
  const s = newGame(42, bestiary);
  const floor = generateFloor(42, 1);
  assert.deepEqual([s.depth, s.turn, s.player.hp, s.result], [1, 0, PLAYER_MAX_HP, "playing"]);
  assert.deepEqual([s.player.x, s.player.y], [floor.start.x, floor.start.y]);
  assert.deepEqual(s.enemies.map((e) => e.species), floor.enemies.map((e) => e.species));
  assert.ok(s.enemies.every((e) => e.hp === bestiary.species[e.species].hp));
  assert.equal(s.explored[floor.start.y * MAP_W + floor.start.x], 1);
});

test("a run replays exactly: same seed and actions give the same final state", () => {
  const actions: Action[] = Array.from({ length: 60 }, (_, i) => (i % 5 === 4 ? WAIT : move((["right", "down", "left", "up"] as const)[i % 4])));
  const play = () => actions.reduce(step, newGame(42, testBestiary()));
  assert.deepEqual(play(), play());
});
