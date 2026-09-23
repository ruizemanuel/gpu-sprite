import type { Bestiary } from "./bestiary.ts";
import { FLOORS, generateFloor, isFloor, MAP_H, MAP_W, type Floor, type Point } from "./dungeon.ts";

// Balanced by simulating 200 run seeds with two bots (one fights everything, one rushes the
// stairs): they win about 77% and 84% of runs, in roughly 230 and 125 turns.
export const PLAYER_MAX_HP = 40;
export const PLAYER_ATK = 3;
export const POTION_HEAL = 15;
/** Chebyshev radius the player sees. Enemies within it chase the player: if it sees you, it hunts you. */
export const SIGHT = 4;
export const LOG_LINES = 4;

export type Direction = "up" | "down" | "left" | "right";
export type Action = { type: "move"; dir: Direction } | { type: "wait" };
export type Result = "playing" | "won" | "died";

export interface Enemy {
  species: number;
  hp: number;
  x: number;
  y: number;
}

export interface GameState {
  runSeed: number;
  bestiary: Bestiary;
  depth: number;
  floor: Floor;
  player: { x: number; y: number; hp: number };
  enemies: Enemy[];
  potions: Point[];
  /** Player actions that took a turn (every action except walking into a wall). */
  turn: number;
  /** Per tile of the current floor: 1 once seen. */
  explored: Uint8Array;
  /** Per species: seen at least once this run. */
  discovered: boolean[];
  /** The last LOG_LINES messages, oldest first. */
  log: string[];
  result: Result;
}

const DELTAS: Record<Direction, Point> = { up: { x: 0, y: -1 }, down: { x: 0, y: 1 }, left: { x: -1, y: 0 }, right: { x: 1, y: 0 } };

function floorFields(runSeed: number, bestiary: Bestiary, depth: number) {
  const floor = generateFloor(runSeed, depth);
  return {
    depth,
    floor,
    enemies: floor.enemies.map((e) => ({ species: e.species, hp: bestiary.species[e.species].hp, x: e.at.x, y: e.at.y })),
    potions: floor.potions.map((p) => ({ ...p })),
    explored: new Uint8Array(MAP_W * MAP_H),
  };
}

export function newGame(runSeed: number, bestiary: Bestiary): GameState {
  const fields = floorFields(runSeed, bestiary, 1);
  const state: GameState = {
    runSeed,
    bestiary,
    ...fields,
    player: { x: fields.floor.start.x, y: fields.floor.start.y, hp: PLAYER_MAX_HP },
    turn: 0,
    discovered: bestiary.species.map(() => false),
    log: [],
    result: "playing",
  };
  look(state);
  return state;
}

export function isVisible(state: GameState, x: number, y: number): boolean {
  return Math.max(Math.abs(x - state.player.x), Math.abs(y - state.player.y)) <= SIGHT;
}

/**
 * One player action and the enemies' reply. Pure: returns a new state and never changes `state`.
 * Walking into a wall and any action after the game ended return `state` itself.
 */
export function step(state: GameState, action: Action): GameState {
  if (state.result !== "playing") return state;
  const s = clone(state);
  if (action.type === "move") {
    const d = DELTAS[action.dir];
    const x = s.player.x + d.x;
    const y = s.player.y + d.y;
    const target = s.enemies.find((e) => e.x === x && e.y === y);
    if (target) {
      attack(s, target);
    } else if (!isFloor(s.floor, x, y)) {
      return state;
    } else {
      s.player.x = x;
      s.player.y = y;
      drinkPotionAt(s, x, y);
      if (x === s.floor.stairs.x && y === s.floor.stairs.y) {
        s.turn += 1;
        if (s.depth === FLOORS) {
          s.result = "won";
          say(s, "You escape the dungeon!");
        } else {
          descend(s);
        }
        return s;
      }
    }
  }
  s.turn += 1;
  enemiesAct(s);
  if (s.player.hp <= 0) {
    s.result = "died";
    say(s, `You die on floor ${s.depth}.`);
    return s;
  }
  look(s);
  return s;
}

function clone(s: GameState): GameState {
  return {
    ...s,
    player: { ...s.player },
    enemies: s.enemies.map((e) => ({ ...e })),
    potions: s.potions.map((p) => ({ ...p })),
    explored: s.explored.slice(),
    discovered: [...s.discovered],
    log: [...s.log],
  };
}

function say(s: GameState, message: string): void {
  s.log = [...s.log, message].slice(-LOG_LINES);
}

function attack(s: GameState, target: Enemy): void {
  const name = s.bestiary.species[target.species].name;
  target.hp -= PLAYER_ATK;
  if (target.hp > 0) {
    say(s, `You hit ${name} for ${PLAYER_ATK}.`);
    return;
  }
  s.enemies = s.enemies.filter((e) => e !== target);
  say(s, `You kill ${name}.`);
}

function drinkPotionAt(s: GameState, x: number, y: number): void {
  const index = s.potions.findIndex((p) => p.x === x && p.y === y);
  if (index < 0) return;
  s.potions.splice(index, 1);
  const healed = Math.min(PLAYER_MAX_HP, s.player.hp + POTION_HEAL) - s.player.hp;
  s.player.hp += healed;
  say(s, `You drink a potion (+${healed}).`);
}

function descend(s: GameState): void {
  Object.assign(s, floorFields(s.runSeed, s.bestiary, s.depth + 1));
  s.player.x = s.floor.start.x;
  s.player.y = s.floor.start.y;
  say(s, `You descend to floor ${s.depth}.`);
  look(s);
}

/** Axis moves toward (dx, dy), the axis with the larger distance first (horizontal on a tie). */
function stepsToward(dx: number, dy: number): Point[] {
  const h = { x: Math.sign(dx), y: 0 };
  const v = { x: 0, y: Math.sign(dy) };
  return (Math.abs(dx) >= Math.abs(dy) ? [h, v] : [v, h]).filter((p) => p.x !== 0 || p.y !== 0);
}

function enemiesAct(s: GameState): void {
  for (const e of s.enemies) {
    const dx = s.player.x - e.x;
    const dy = s.player.y - e.y;
    if (Math.abs(dx) + Math.abs(dy) === 1) {
      const species = s.bestiary.species[e.species];
      s.player.hp -= species.atk;
      say(s, `${species.name} hits you for ${species.atk}.`);
      if (s.player.hp <= 0) return;
      continue;
    }
    if (Math.max(Math.abs(dx), Math.abs(dy)) > SIGHT) continue;
    // One step from the player it attacked above, so a step never lands on the player.
    for (const m of stepsToward(dx, dy)) {
      const x = e.x + m.x;
      const y = e.y + m.y;
      const blocked = !isFloor(s.floor, x, y) || s.enemies.some((o) => o.x === x && o.y === y);
      if (!blocked) {
        e.x = x;
        e.y = y;
        break;
      }
    }
  }
}

/** Marks the tiles in sight as explored and discovers the species of every enemy in sight. */
function look(s: GameState): void {
  const { x: px, y: py } = s.player;
  for (let y = Math.max(0, py - SIGHT); y <= Math.min(MAP_H - 1, py + SIGHT); y++) {
    for (let x = Math.max(0, px - SIGHT); x <= Math.min(MAP_W - 1, px + SIGHT); x++) s.explored[y * MAP_W + x] = 1;
  }
  for (const e of s.enemies) {
    if (isVisible(s, e.x, e.y)) s.discovered[e.species] = true;
  }
}
