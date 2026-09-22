export const SPRITE_SIDE = 16;
export const SPRITE_PIXELS = SPRITE_SIDE * SPRITE_SIDE;

/** A 16x16 1-bit sprite. `pixels` is row-major, 0 = background, 1 = foreground. */
export interface Sprite {
  width: 16;
  height: 16;
  pixels: Uint8Array;
}

/**
 * Clear every foreground pixel that has no foreground pixel among its 8 neighbours.
 *
 * One pass: neighbours are counted on the input image, not on the partially cleaned
 * one. Pixels outside the 16x16 grid count as background (no wrap-around between rows).
 * Mirrors `packages/training/sprite.py`'s `despeckle`. Returns a new array.
 */
export function despeckle(pixels: Uint8Array): Uint8Array {
  if (pixels.length !== SPRITE_PIXELS) {
    throw new RangeError(`expected ${SPRITE_PIXELS} pixels, got ${pixels.length}`);
  }
  const out = new Uint8Array(SPRITE_PIXELS);
  for (let y = 0; y < SPRITE_SIDE; y++) {
    for (let x = 0; x < SPRITE_SIDE; x++) {
      const i = y * SPRITE_SIDE + x;
      if (!pixels[i]) continue;
      let neighbours = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dy === 0 && dx === 0) continue;
          const ny = y + dy;
          const nx = x + dx;
          if (ny < 0 || ny >= SPRITE_SIDE || nx < 0 || nx >= SPRITE_SIDE) continue;
          if (pixels[ny * SPRITE_SIDE + nx]) neighbours++;
        }
      }
      out[i] = neighbours > 0 ? 1 : 0;
    }
  }
  return out;
}

/**
 * Decode 256 logits starting at `offset` into a final sprite: `pixel = logit > 0`,
 * then clear isolated foreground pixels (see `despeckle`).
 */
export function logitsToSprite(logits: Float32Array, offset = 0): Sprite {
  const raw = new Uint8Array(SPRITE_PIXELS);
  for (let i = 0; i < SPRITE_PIXELS; i++) {
    raw[i] = logits[offset + i] > 0 ? 1 : 0;
  }
  return { width: 16, height: 16, pixels: despeckle(raw) };
}

/** Decode a 64-char hex string (numpy packbits, MSB first) into 256 pixels. Used by tests. */
export function hexBitsToPixels(hex: string): Uint8Array {
  if (hex.length !== SPRITE_PIXELS / 4) throw new RangeError(`expected ${SPRITE_PIXELS / 4} hex chars, got ${hex.length}`);
  const pixels = new Uint8Array(SPRITE_PIXELS);
  for (let byte = 0; byte < SPRITE_PIXELS / 8; byte++) {
    const value = parseInt(hex.slice(byte * 2, byte * 2 + 2), 16);
    for (let bit = 0; bit < 8; bit++) {
      pixels[byte * 8 + bit] = (value >> (7 - bit)) & 1;
    }
  }
  return pixels;
}
