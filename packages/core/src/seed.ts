/** Latent dimension of the shipped decoder. */
export const LATENT_DIM = 32;
/** Version of the seed → latent algorithm. Bump if the algorithm changes. */
export const SEED_VERSION = 1;

const GOLDEN = 0x9e3779b9;
const DRAWS_PER_DIM = 12;
const TWO_32 = 4294967296;

/** Reduce any safe integer to an unsigned 32-bit seed (two's complement for negatives). */
export function toUint32Seed(seed: number): number {
  if (!Number.isSafeInteger(seed)) {
    throw new TypeError(`seed must be a safe integer, got ${String(seed)}`);
  }
  return seed >>> 0;
}

/**
 * Deterministic seed → latent. xorshift32 drives an Irwin-Hall approximation
 * of a standard normal (sum of 12 uniforms minus 6). Uses only integer ops,
 * additions and one division so Python and TypeScript agree bit for bit.
 */
export function seedToLatent(seed: number): Float32Array {
  let x = toUint32Seed(seed);
  if (x === 0) x = GOLDEN;
  const z = new Float32Array(LATENT_DIM);
  for (let i = 0; i < LATENT_DIM; i++) {
    let acc = 0;
    for (let k = 0; k < DRAWS_PER_DIM; k++) {
      x ^= x << 13;
      x >>>= 0;
      x ^= x >>> 17;
      x >>>= 0;
      x ^= x << 5;
      x >>>= 0;
      acc += x / TWO_32;
    }
    z[i] = acc - 6;
  }
  return z;
}
