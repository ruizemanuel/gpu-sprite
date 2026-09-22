import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  resolve: {
    alias: { "gpu-sprite": fileURLToPath(new URL("../../packages/core/src/index.ts", import.meta.url)) },
  },
  build: { target: "es2022", outDir: "dist", emptyOutDir: true },
  server: { port: 4174, strictPort: true },
  preview: { port: 4175, strictPort: true },
});
