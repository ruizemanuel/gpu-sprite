/** A seeded uint32 generator. Every stream of a run is one of these, so a run replays exactly from its seed. */
export interface Rng {
  /** Next uint32. */
  next(): number;
  /** Integer in [min, max], both inclusive. */
  int(min: number, max: number): number;
}

/** mulberry32: a small, fast 32-bit PRNG with a single uint32 of state. */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  const next = (): number => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
  return { next, int: (min, max) => min + (next() % (max - min + 1)) };
}

function fnv1a(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

/**
 * Seed of an independent stream of a run, one per purpose ("bestiary", "floor-1", ...): the run
 * seed xor the label's FNV-1a hash, through murmur3's finalizer so nearby run seeds diverge.
 */
export function deriveSeed(runSeed: number, label: string): number {
  let h = (runSeed ^ fnv1a(label)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}

export const MAX_RUN_SEED = 0xffffffff;

/** A run seed from the `seed` URL parameter: decimal digits only, 0 to 4294967295. Anything else is `undefined`. */
export function parseRunSeed(text: string): number | undefined {
  if (!/^\d{1,10}$/.test(text)) return undefined;
  const value = Number(text);
  return value <= MAX_RUN_SEED ? value : undefined;
}
