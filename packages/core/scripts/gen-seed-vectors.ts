import { writeFileSync, mkdirSync } from "node:fs";
import { LATENT_DIM, SEED_VERSION, seedToLatent } from "../src/seed.ts";

const seeds = [0, 1, 2, 7, 42, 1337, 65535, 65536, 123456789, 2147483647, 2147483648, 4294967295, -1, -42, 999999999, 31337];
const vectors = seeds.map((seed) => ({ seed, z: Array.from(seedToLatent(seed)) }));
const out = new URL("../test/fixtures/seed-vectors.json", import.meta.url);
mkdirSync(new URL("../test/fixtures/", import.meta.url), { recursive: true });
writeFileSync(out, JSON.stringify({ seedVersion: SEED_VERSION, latentDim: LATENT_DIM, vectors }, null, 2) + "\n");
console.log(`wrote ${vectors.length} vectors to ${out.pathname}`);
