export const SIDE = 16;
export const PIXELS = SIDE * SIDE;

export interface SpriteMetrics {
  /** Foreground pixels. */
  fg: number;
  /** Connected components of the foreground, 8-connectivity (the quality gate's measure). */
  components: number;
  /** Columns from the first to the last one holding a foreground pixel, inclusive; 0 when empty. */
  width: number;
}

/** Metrics of a 16×16 row-major 0/1 sprite. */
export function spriteMetrics(pixels: Uint8Array): SpriteMetrics {
  if (pixels.length !== PIXELS) throw new RangeError(`expected ${PIXELS} pixels, got ${pixels.length}`);
  let fg = 0;
  let minX = SIDE;
  let maxX = -1;
  for (let i = 0; i < PIXELS; i++) {
    if (!pixels[i]) continue;
    fg++;
    const x = i % SIDE;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
  }
  const seen = new Uint8Array(PIXELS);
  let components = 0;
  for (let start = 0; start < PIXELS; start++) {
    if (!pixels[start] || seen[start]) continue;
    components++;
    seen[start] = 1;
    const stack = [start];
    while (stack.length > 0) {
      const p = stack.pop() as number;
      const x = p % SIDE;
      const y = (p - x) / SIDE;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= SIDE || ny >= SIDE) continue;
          const q = ny * SIDE + nx;
          if (pixels[q] && !seen[q]) {
            seen[q] = 1;
            stack.push(q);
          }
        }
      }
    }
  }
  return { fg, components, width: maxX < 0 ? 0 : maxX - minX + 1 };
}
