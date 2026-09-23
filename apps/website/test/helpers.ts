/** A 16×16 sprite from 16 rows of 16 characters: "#" is foreground, anything else background. */
export function spriteFrom(rows: string[]): Uint8Array {
  if (rows.length !== 16 || rows.some((r) => r.length !== 16)) throw new Error("spriteFrom needs 16 rows of 16 characters");
  return Uint8Array.from(rows.join(""), (c) => (c === "#" ? 1 : 0));
}

/** A solid block of foreground pixels at columns [x, x + w) and rows [y, y + h) of an empty sprite. */
export function blockSprite(x: number, y: number, w: number, h: number): Uint8Array {
  const pixels = new Uint8Array(256);
  for (let row = y; row < y + h; row++) {
    for (let col = x; col < x + w; col++) pixels[row * 16 + col] = 1;
  }
  return pixels;
}

/** A one-component sprite with exactly `fg` pixels: full rows of `w` from the top-left, then a partial row. */
export function spriteWithFg(fg: number, w = 10): Uint8Array {
  const pixels = new Uint8Array(256);
  for (let i = 0; i < fg; i++) pixels[Math.floor(i / w) * 16 + (i % w)] = 1;
  return pixels;
}
