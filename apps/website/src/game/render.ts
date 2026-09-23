import { PLAYER_COLOR } from "./bestiary.ts";
import { FLOORS, isFloor, MAP_H, MAP_W, type Floor } from "./dungeon.ts";
import { isVisible, PLAYER_ATK, PLAYER_MAX_HP, type GameState } from "./rules.ts";
import { TILES } from "./tiles.ts";

const TILE_PX = 16;
const MAP_SCALE = 2;
const PANEL_SCALE = 3;
const CELL = TILE_PX * MAP_SCALE;
/** Explored tiles out of sight are drawn at this opacity. */
const REMEMBERED_ALPHA = 0.35;
export const MAP_CANVAS_W = MAP_W * CELL;
export const MAP_CANVAS_H = MAP_H * CELL;

/** The page elements the game draws into; `play.ts` looks them up once. */
export interface View {
  map: CanvasRenderingContext2D;
  player: HTMLElement;
  bestiary: HTMLOListElement;
  hud: HTMLElement;
  log: HTMLUListElement;
  overlay: HTMLElement;
  overlayTitle: HTMLElement;
  overlayStats: HTMLElement;
}

const bitmaps = new Map<string, HTMLCanvasElement>();

/** A 1-bit bitmap drawn once in `color` at `scale` onto its own canvas, cached by `key`. */
function bitmap(key: string, bits: Uint8Array, color: string, scale: number): HTMLCanvasElement {
  const id = `${key}|${color}|${scale}`;
  let canvas = bitmaps.get(id);
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.width = canvas.height = TILE_PX * scale;
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    ctx.fillStyle = color;
    for (let i = 0; i < bits.length; i++) {
      if (bits[i]) ctx.fillRect((i % TILE_PX) * scale, Math.floor(i / TILE_PX) * scale, scale, scale);
    }
    bitmaps.set(id, canvas);
  }
  return canvas;
}

/** Walls are drawn only where they border the floor, so solid rock stays black. */
function bordersFloor(floor: Floor, x: number, y: number): boolean {
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) if (isFloor(floor, x + dx, y + dy)) return true;
  }
  return false;
}

function drawMap(ctx: CanvasRenderingContext2D, s: GameState): void {
  ctx.imageSmoothingEnabled = false;
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, MAP_CANVAS_W, MAP_CANVAS_H);
  const exitHere = s.depth === FLOORS;
  for (let y = 0; y < MAP_H; y++) {
    for (let x = 0; x < MAP_W; x++) {
      if (!s.explored[y * MAP_W + x]) continue;
      let name: keyof typeof TILES;
      if (!isFloor(s.floor, x, y)) {
        if (!bordersFloor(s.floor, x, y)) continue;
        name = "wall";
      } else if (x === s.floor.stairs.x && y === s.floor.stairs.y) {
        name = exitHere ? "exit" : "stairs";
      } else {
        name = "floor";
      }
      ctx.globalAlpha = isVisible(s, x, y) ? 1 : REMEMBERED_ALPHA;
      ctx.drawImage(bitmap(name, TILES[name].bits, TILES[name].color, MAP_SCALE), x * CELL, y * CELL);
    }
  }
  ctx.globalAlpha = 1;
  for (const p of s.potions) {
    if (isVisible(s, p.x, p.y)) ctx.drawImage(bitmap("potion", TILES.potion.bits, TILES.potion.color, MAP_SCALE), p.x * CELL, p.y * CELL);
  }
  for (const e of s.enemies) {
    if (!isVisible(s, e.x, e.y)) continue;
    const species = s.bestiary.species[e.species];
    ctx.drawImage(bitmap(`sprite-${species.seed}`, species.pixels, species.color, MAP_SCALE), e.x * CELL, e.y * CELL);
  }
  const player = s.bestiary.player;
  ctx.drawImage(bitmap(`sprite-${player.seed}`, player.pixels, PLAYER_COLOR, MAP_SCALE), s.player.x * CELL, s.player.y * CELL);
}

/** A panel row: the sprite at panel scale (blank when unknown) and one line per entry of `lines`. */
function entry(key: string, pixels: Uint8Array, color: string, lines: string[], known: boolean): HTMLElement {
  const row = document.createElement("div");
  row.className = known ? "entry" : "entry unknown";
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = TILE_PX * PANEL_SCALE;
  if (known) (canvas.getContext("2d") as CanvasRenderingContext2D).drawImage(bitmap(key, pixels, color, PANEL_SCALE), 0, 0);
  const text = document.createElement("div");
  for (const line of lines) {
    const div = document.createElement("div");
    div.textContent = line;
    text.append(div);
  }
  row.append(canvas, text);
  return row;
}

export function render(view: View, s: GameState): void {
  drawMap(view.map, s);
  const over = s.result !== "playing";
  const player = s.bestiary.player;
  view.player.replaceChildren(entry(`sprite-${player.seed}`, player.pixels, PLAYER_COLOR, ["You", `HP ${Math.max(0, s.player.hp)}/${PLAYER_MAX_HP} · ATK ${PLAYER_ATK}`], true));
  view.bestiary.replaceChildren(
    ...s.bestiary.species.map((species, id) => {
      const known = over || s.discovered[id];
      const lines = known ? [species.boss ? `${species.name} (boss)` : species.name, `HP ${species.hp} · ATK ${species.atk}`] : ["?"];
      const item = document.createElement("li");
      item.append(entry(`sprite-${species.seed}`, species.pixels, species.color, lines, known));
      return item;
    }),
  );
  view.hud.textContent = `Floor ${s.depth} · Turn ${s.turn}`;
  view.log.replaceChildren(
    ...s.log.map((message) => {
      const item = document.createElement("li");
      item.textContent = message;
      return item;
    }),
  );
  view.overlay.hidden = !over;
  if (over) {
    view.overlayTitle.textContent = s.result === "won" ? "You escaped!" : `You died on floor ${s.depth}`;
    const seen = s.discovered.filter(Boolean).length;
    view.overlayStats.textContent = `Turn ${s.turn} · Seen ${seen}/${s.discovered.length}`;
  }
}
