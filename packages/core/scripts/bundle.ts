import { build } from "esbuild";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));

await build({
  entryPoints: [root + "src/index.ts"],
  bundle: true,
  format: "esm",
  minify: true,
  target: "es2022",
  outfile: root + "dist/gpu-sprite.min.js",
  legalComments: "none",
});

await build({
  entryPoints: [root + "test/browser/entry.ts"],
  bundle: true,
  format: "iife",
  globalName: "GpuSpriteTest",
  minify: false,
  target: "es2022",
  outfile: root + "dist/gpu-sprite.test.iife.js",
  legalComments: "none",
});

console.log("bundled dist/gpu-sprite.min.js");
