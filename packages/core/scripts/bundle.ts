import { build } from "esbuild";
import { existsSync } from "node:fs";
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

// The browser-test entry is created in Task 10; until then only the library bundle is built.
if (existsSync(root + "test/browser/entry.ts")) {
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
}

console.log("bundled dist/gpu-sprite.min.js");
