import { BOSS, FLOOR_GROUPS } from "./bestiary.ts";
import { deriveSeed, mulberry32, type Rng } from "./rng.ts";

export const MAP_W = 32;
export const MAP_H = 20;
export const FLOORS = 3;
const MAX_ROOMS = 7;
const MIN_ROOMS = 5;
const ROOM_ATTEMPTS = 200;

export interface Point {
  x: number;
  y: number;
}

/** A room's interior: columns [x, x + w) and rows [y, y + h). */
export interface Room {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface EnemySpawn {
  species: number;
  at: Point;
}

export interface Floor {
  depth: number;
  /** MAP_W × MAP_H, row-major: 1 = floor, 0 = wall. */
  tiles: Uint8Array;
  rooms: Room[];
  start: Point;
  /** Stairs down on floors 1 and 2, the exit on the last floor. */
  stairs: Point;
  enemies: EnemySpawn[];
  potions: Point[];
}

export function isFloor(floor: Floor, x: number, y: number): boolean {
  return x >= 0 && y >= 0 && x < MAP_W && y < MAP_H && floor.tiles[y * MAP_W + x] === 1;
}

export function center(room: Room): Point {
  return { x: room.x + Math.floor(room.w / 2), y: room.y + Math.floor(room.h / 2) };
}

/** True when two rooms overlap or leave no wall tile between them. */
function touches(a: Room, b: Room): boolean {
  return a.x <= b.x + b.w && b.x <= a.x + a.w && a.y <= b.y + b.h && b.y <= a.y + a.h;
}

function placeRooms(rng: Rng): Room[] {
  const rooms: Room[] = [];
  for (let attempt = 0; attempt < ROOM_ATTEMPTS && rooms.length < MAX_ROOMS; attempt++) {
    const w = rng.int(4, 8);
    const h = rng.int(3, 6);
    const room = { x: rng.int(1, MAP_W - 1 - w), y: rng.int(1, MAP_H - 1 - h), w, h };
    if (rooms.every((r) => !touches(r, room))) rooms.push(room);
  }
  return rooms;
}

function carve(tiles: Uint8Array, x0: number, y0: number, x1: number, y1: number): void {
  for (let y = Math.min(y0, y1); y <= Math.max(y0, y1); y++) {
    for (let x = Math.min(x0, x1); x <= Math.max(x0, x1); x++) tiles[y * MAP_W + x] = 1;
  }
}

/** A free tile of `room`, drawn by `rng` until one is not in `taken`; the tile is then added to `taken`. */
function freeTile(rng: Rng, room: Room, taken: Set<number>): Point {
  for (;;) {
    const p = { x: room.x + rng.int(0, room.w - 1), y: room.y + rng.int(0, room.h - 1) };
    const key = p.y * MAP_W + p.x;
    if (taken.has(key)) continue;
    taken.add(key);
    return p;
  }
}

/** A free tile of a room other than the first one. */
function freeTileAwayFromStart(rng: Rng, rooms: Room[], taken: Set<number>): Point {
  return freeTile(rng, rooms[rng.int(1, rooms.length - 1)], taken);
}

/** Floor `depth` (1-based) of a run, from its own "floor-N" stream. */
export function generateFloor(runSeed: number, depth: number): Floor {
  const rng = mulberry32(deriveSeed(runSeed, `floor-${depth}`));
  let rooms = placeRooms(rng);
  while (rooms.length < MIN_ROOMS) rooms = placeRooms(rng);

  const tiles = new Uint8Array(MAP_W * MAP_H);
  for (const r of rooms) carve(tiles, r.x, r.y, r.x + r.w - 1, r.y + r.h - 1);
  for (let i = 0; i + 1 < rooms.length; i++) {
    const a = center(rooms[i]);
    const b = center(rooms[i + 1]);
    const corner = rng.int(0, 1) === 1 ? { x: b.x, y: a.y } : { x: a.x, y: b.y };
    carve(tiles, a.x, a.y, corner.x, corner.y);
    carve(tiles, corner.x, corner.y, b.x, b.y);
  }

  const start = center(rooms[0]);
  const last = rooms[rooms.length - 1];
  const stairs = center(last);
  const taken = new Set([start.y * MAP_W + start.x, stairs.y * MAP_W + stairs.x]);
  const group = FLOOR_GROUPS[depth - 1];
  const enemies: EnemySpawn[] = [];
  const count = depth < FLOORS ? 3 + rng.int(0, 2) : 2 + rng.int(0, 1);
  for (let i = 0; i < count; i++) {
    enemies.push({ species: group[rng.int(0, group.length - 1)], at: freeTileAwayFromStart(rng, rooms, taken) });
  }
  if (depth === FLOORS) enemies.push({ species: BOSS, at: freeTile(rng, last, taken) });
  const potions = Array.from({ length: 1 + rng.int(0, 1) }, () => freeTileAwayFromStart(rng, rooms, taken));
  return { depth, tiles, rooms, start, stairs, enemies, potions };
}
