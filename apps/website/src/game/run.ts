import { buildBestiary, CANDIDATES_PER_BATCH, candidateSeeds, MAX_BATCHES, type Bestiary, type Candidate } from "./bestiary.ts";

/** The part of gpu-sprite's `generateMany` result the game reads. */
export interface GeneratedBatch {
  sprites: { pixels: Uint8Array }[];
  backend: string;
  adapter?: string;
}

export interface BestiaryRun {
  /** `undefined` when MAX_BATCHES batches did not yield a player and 8 creatures. */
  bestiary: Bestiary | undefined;
  /** Candidates generated in total. */
  candidates: number;
  /** Time spent inside `generate`, summed over batches. */
  ms: number;
  backend: string;
  adapter?: string;
}

/**
 * Generates batches of the run's candidate seeds until they yield a bestiary, at most MAX_BATCHES.
 * Returns `undefined` as soon as `isCurrent()` is false after a batch: a replaced run stops there
 * and never reports a result.
 */
export async function generateBestiary(
  runSeed: number,
  generate: (seeds: number[]) => Promise<GeneratedBatch>,
  now: () => number,
  isCurrent: () => boolean,
): Promise<BestiaryRun | undefined> {
  const candidates: Candidate[] = [];
  let ms = 0;
  let backend = "";
  let adapter: string | undefined;
  let bestiary: Bestiary | undefined;
  for (let batch = 0; batch < MAX_BATCHES && !bestiary; batch++) {
    const seeds = candidateSeeds(runSeed, (batch + 1) * CANDIDATES_PER_BATCH).slice(batch * CANDIDATES_PER_BATCH);
    const started = now();
    const result = await generate(seeds);
    ms += now() - started;
    if (!isCurrent()) return undefined;
    backend = result.backend;
    adapter = result.adapter;
    result.sprites.forEach((sprite, i) => candidates.push({ seed: seeds[i], pixels: sprite.pixels }));
    bestiary = buildBestiary(candidates);
  }
  return { bestiary, candidates: candidates.length, ms, backend, adapter };
}
