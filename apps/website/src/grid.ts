import type { Sprite } from "gpu-sprite";

export const GRID = 32;
export const COUNT = GRID * GRID;
export const SCALE = 2;
export const GAP = 1;
export const CELL = 16 * SCALE + GAP;
export const CANVAS = GRID * CELL;

export function drawGrid(ctx: CanvasRenderingContext2D, sprites: Sprite[]): void {
  const image = ctx.createImageData(CANVAS, CANVAS);
  const data = image.data;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 12;
    data[i + 1] = 12;
    data[i + 2] = 12;
    data[i + 3] = 255;
  }
  sprites.forEach((sprite, index) => {
    const gx = (index % GRID) * CELL;
    const gy = Math.floor(index / GRID) * CELL;
    for (let p = 0; p < 256; p++) {
      if (!sprite.pixels[p]) continue;
      const px = gx + (p % 16) * SCALE;
      const py = gy + Math.floor(p / 16) * SCALE;
      for (let dy = 0; dy < SCALE; dy++) {
        let o = ((py + dy) * CANVAS + px) * 4;
        for (let dx = 0; dx < SCALE; dx++, o += 4) {
          data[o] = 240;
          data[o + 1] = 240;
          data[o + 2] = 240;
        }
      }
    }
  });
  ctx.putImageData(image, 0, 0);
}

/** Map a pointer position on the (possibly CSS-scaled) canvas to a sprite index, or -1. */
export function cellAt(canvas: HTMLCanvasElement, clientX: number, clientY: number): number {
  const rect = canvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * CANVAS;
  const y = ((clientY - rect.top) / rect.height) * CANVAS;
  const col = Math.floor(x / CELL);
  const row = Math.floor(y / CELL);
  if (col < 0 || row < 0 || col >= GRID || row >= GRID) return -1;
  return row * GRID + col;
}

export function drawSprite(ctx: CanvasRenderingContext2D, sprite: Sprite, scale: number): void {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, 16 * scale, 16 * scale);
  ctx.fillStyle = "#f0f0f0";
  for (let p = 0; p < 256; p++) {
    if (sprite.pixels[p]) ctx.fillRect((p % 16) * scale, Math.floor(p / 16) * scale, scale, scale);
  }
}
