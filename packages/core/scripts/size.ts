import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { brotliCompressSync, constants } from "node:zlib";

const LIMIT = 40_000;
const file = fileURLToPath(new URL("../dist/gpu-sprite.min.js", import.meta.url));
const raw = readFileSync(file);
const brotli = brotliCompressSync(raw, { params: { [constants.BROTLI_PARAM_QUALITY]: 11 } });
console.log(`gpu-sprite.min.js: ${statSync(file).size} bytes minified, ${brotli.length} bytes brotli (limit ${LIMIT})`);
const sizeJson = JSON.stringify({ minified: statSync(file).size, brotli: brotli.length, limit: LIMIT });
writeFileSync(fileURLToPath(new URL("../dist/size.json", import.meta.url)), sizeJson);
const publicDir = fileURLToPath(new URL("../../../apps/website/public/", import.meta.url));
mkdirSync(publicDir, { recursive: true });
writeFileSync(publicDir + "size.json", sizeJson);
if (brotli.length >= LIMIT) {
  console.error(`size gate failed: ${brotli.length} >= ${LIMIT}`);
  process.exit(1);
}
