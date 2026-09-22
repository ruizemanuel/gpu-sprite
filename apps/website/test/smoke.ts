import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { build, preview } from "vite";

const root = fileURLToPath(new URL("../", import.meta.url));
await build({ root, logLevel: "error" });
const server = await preview({ root, preview: { port: 4175, strictPort: true }, logLevel: "error" });
const url = server.resolvedUrls?.local[0] ?? "http://localhost:4175/";

const browser = await chromium.launch({ headless: true, channel: "chromium", args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 1200 } });
const errors: string[] = [];
page.on("pageerror", (e) => errors.push(e.message));
const failures: string[] = [];
const screenshotPath = fileURLToPath(new URL("../../../test-results/website-smoke.png", import.meta.url));
try {
  await page.goto(url);
  await page.waitForFunction(() => (window as unknown as { __gpuSprite?: { count: number } }).__gpuSprite?.count === 1024, null, { timeout: 30_000 });
  const statusText = await page.textContent("#status");
  if (!statusText?.includes("cpu:")) failures.push(`status did not report cpu render: ${statusText}`);

  await page.click("#grid", { position: { x: 20, y: 20 } });
  if (await page.isHidden("#detail")) failures.push("detail panel did not open");

  // A VAE latent dimension can be inactive, so moving slider 0 alone may not change the
  // sprite. Try sliders in order until one changes the rendered detail canvas.
  let changedSliderIndex = -1;
  const before = await page.evaluate(() => (document.querySelector("#detail-canvas") as HTMLCanvasElement).toDataURL());
  const sliderCount = await page.evaluate(() => document.querySelectorAll("#sliders input").length);
  for (let i = 0; i < sliderCount; i++) {
    await page.evaluate((index) => {
      const slider = document.querySelectorAll("#sliders input")[index] as HTMLInputElement;
      slider.value = "3";
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    }, i);
    await page.waitForTimeout(50);
    const after = await page.evaluate(() => (document.querySelector("#detail-canvas") as HTMLCanvasElement).toDataURL());
    if (after !== before) {
      changedSliderIndex = i;
      break;
    }
  }
  if (changedSliderIndex < 0) failures.push(`moving any of the ${sliderCount} latent sliders did not change the detail sprite`);
  else console.log(`slider index: ${changedSliderIndex} changed the detail sprite`);

  await page.click("#regenerate");
  // `render()` is fire-and-forget from the click handler and can overlap with a render triggered
  // by a backend toggle; wait for this specific render (base 1024) to actually finish, not just
  // for `state.base` to have been bumped synchronously, before moving on.
  await page.waitForFunction(
    () => {
      const s = (window as unknown as { __gpuSprite: { renderedBase: number; rendering: boolean } }).__gpuSprite;
      return s.renderedBase === 1024 && !s.rendering;
    },
    null,
    { timeout: 30_000 },
  );

  const webgpuEnabled = await page.evaluate(() => !(document.querySelector("#webgpu-radio") as HTMLInputElement).disabled);
  if (webgpuEnabled) {
    await page.check("#webgpu-radio");
    await page.waitForFunction(
      () => {
        const s = (window as unknown as { __gpuSprite: { rendering: boolean } }).__gpuSprite;
        const statusText = (document.querySelector("#status") as HTMLElement).textContent;
        return !s.rendering && Boolean(statusText?.includes("webgpu:"));
      },
      null,
      { timeout: 30_000 },
    );
    console.log("webgpu toggle: ok");
  } else {
    console.log(`webgpu toggle: disabled (${await page.getAttribute("#webgpu-radio", "title")})`);
  }

  mkdirSync(fileURLToPath(new URL("../../../test-results/", import.meta.url)), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });
} finally {
  await browser.close();
  await server.close();
}
if (errors.length) failures.push(`page errors: ${errors.join(" | ")}`);
if (failures.length) {
  for (const f of failures) console.error(`FAIL ${f}`);
  process.exit(1);
}
console.log("PASS website smoke");
