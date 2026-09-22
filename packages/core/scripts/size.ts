import { readFileSync, statSync } from "node:fs";
import { brotliCompressSync, constants } from "node:zlib";
import { fileURLToPath } from "node:url";

const LIMIT = 40_000;
const file = fileURLToPath(new URL("../dist/gpu-sprite.min.js", import.meta.url));
const raw = readFileSync(file);
const brotli = brotliCompressSync(raw, { params: { [constants.BROTLI_PARAM_QUALITY]: 11 } });
console.log(`gpu-sprite.min.js: ${statSync(file).size} bytes minified, ${brotli.length} bytes brotli (limit ${LIMIT})`);
if (brotli.length >= LIMIT) {
  console.error(`size gate failed: ${brotli.length} >= ${LIMIT}`);
  process.exit(1);
}
