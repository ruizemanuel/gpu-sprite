import { deriveSeed, mulberry32 } from "./rng.ts";
import { spriteMetrics, type SpriteMetrics } from "./sprite-metrics.ts";

/** A generated sprite and the seed that produced it. */
export interface Candidate {
  seed: number;
  pixels: Uint8Array;
}

export interface Species {
  seed: number;
  pixels: Uint8Array;
  name: string;
  /** CSS color the sprite is drawn in. */
  color: string;
  fg: number;
  width: number;
  hp: number;
  atk: number;
  boss: boolean;
}

export interface Bestiary {
  player: Candidate;
  /** The run's creatures, sorted by foreground pixels (ties by candidate order). Indexes are species ids. */
  species: Species[];
}

export const CANDIDATES_PER_BATCH = 256;
export const MAX_BATCHES = 4;
export const SPECIES_COUNT = 8;
export const MIN_FG = 52;
export const MAX_FG = 115;
export const MAX_COMPONENTS = 2;
/** Species ids that roam floors 1, 2 and 3; the boss is on floor 3 too, placed by the dungeon. */
export const FLOOR_GROUPS: readonly (readonly number[])[] = [[0, 1, 2], [3, 4, 5], [6]];
export const BOSS = 7;
export const PLAYER_COLOR = "#f0f0f0";

const SYLLABLES = ["gor", "vak", "mul", "bit", "zar", "nok", "ti", "ra", "ul", "eth", "kro", "sa", "mi", "dun", "fel", "ox"];

/** The first `count` sprite seeds of a run's "bestiary" stream; batch b is the slice [b * 256, (b + 1) * 256). */
export function candidateSeeds(runSeed: number, count: number): number[] {
  const rng = mulberry32(deriveSeed(runSeed, "bestiary"));
  return Array.from({ length: count }, () => rng.next());
}

/** Fragmented, near-empty and overly dense sprites do not read as creatures. */
export function passesFilter(m: SpriteMetrics): boolean {
  return m.components <= MAX_COMPONENTS && m.fg >= MIN_FG && m.fg <= MAX_FG;
}

export function hpFor(fg: number): number {
  return Math.ceil(fg / 12);
}

export function atkFor(width: number): number {
  return width <= 9 ? 1 : width <= 13 ? 2 : 3;
}

export function colorFor(seed: number): string {
  return `hsl(${seed % 360}, 70%, 60%)`;
}

/** 2 or 3 syllables drawn by the sprite's seed, capitalized; draws again until the name is not in `taken`. */
export function nameFor(seed: number, taken: ReadonlySet<string>): string {
  const rng = mulberry32(seed);
  for (;;) {
    const count = rng.int(2, 3);
    let name = "";
    for (let i = 0; i < count; i++) name += SYLLABLES[rng.int(0, SYLLABLES.length - 1)];
    name = name[0].toUpperCase() + name.slice(1);
    if (!taken.has(name)) return name;
  }
}

/**
 * The player (first accepted candidate) and the 8 species (next 8), in candidate order, skipping
 * candidates that fail the filter or repeat an accepted sprite. `undefined` when fewer than 9 pass.
 */
export function buildBestiary(candidates: readonly Candidate[]): Bestiary | undefined {
  const accepted: { candidate: Candidate; metrics: SpriteMetrics; order: number }[] = [];
  const seen = new Set<string>();
  for (const candidate of candidates) {
    const metrics = spriteMetrics(candidate.pixels);
    if (!passesFilter(metrics)) continue;
    const key = candidate.pixels.join("");
    if (seen.has(key)) continue;
    seen.add(key);
    accepted.push({ candidate, metrics, order: accepted.length });
    if (accepted.length === 1 + SPECIES_COUNT) break;
  }
  if (accepted.length < 1 + SPECIES_COUNT) return undefined;
  const [player, ...rest] = accepted;
  rest.sort((a, b) => a.metrics.fg - b.metrics.fg || a.order - b.order);
  const taken = new Set<string>();
  const species = rest.map(({ candidate, metrics }, id): Species => {
    const name = nameFor(candidate.seed, taken);
    taken.add(name);
    const boss = id === BOSS;
    const hp = hpFor(metrics.fg);
    return {
      seed: candidate.seed,
      pixels: candidate.pixels,
      name,
      color: colorFor(candidate.seed),
      fg: metrics.fg,
      width: metrics.width,
      hp: boss ? 2 * hp : hp,
      atk: atkFor(metrics.width),
      boss,
    };
  });
  return { player: player.candidate, species };
}
