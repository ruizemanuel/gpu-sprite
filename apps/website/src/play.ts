import { defineGenerator } from "gpu-sprite";
import { SPECIES_COUNT, type Bestiary } from "./game/bestiary.ts";
import { render, type View } from "./game/render.ts";
import { parseRunSeed } from "./game/rng.ts";
import { newGame, step, type Action, type GameState } from "./game/rules.ts";
import { generateBestiary, type BestiaryRun } from "./game/run.ts";

const $ = <T extends HTMLElement>(selector: string): T => document.querySelector(selector) as T;

const mapCanvas = $<HTMLCanvasElement>("#map");
const view: View = {
  map: mapCanvas.getContext("2d") as CanvasRenderingContext2D,
  player: $("#player"),
  bestiary: $("#bestiary"),
  hud: $("#hud"),
  log: $("#log"),
  overlay: $("#overlay"),
  overlayTitle: $("#overlay-title"),
  overlayStats: $("#overlay-stats"),
};
const message = $("#message");
const seedLabel = $("#seed");
const generation = $("#generation");
const copyLink = $<HTMLButtonElement>("#copy-link");

const up: Action = { type: "move", dir: "up" };
const down: Action = { type: "move", dir: "down" };
const left: Action = { type: "move", dir: "left" };
const right: Action = { type: "move", dir: "right" };
const KEYS: Record<string, Action> = {
  ArrowUp: up, w: up, W: up,
  ArrowDown: down, s: down, S: down,
  ArrowLeft: left, a: left, A: left,
  ArrowRight: right, d: right, D: right,
  " ": { type: "wait" },
};

const generator = defineGenerator({ backend: "auto" });
let runSeed = 0;
let bestiary: Bestiary | undefined;
let game: GameState | undefined;
let backend = "";
/** Bumped by every run start, so a slower, older generation cannot overwrite a newer run. */
let runToken = 0;

/** Test hook, like `window.__gpuSprite` on the grid page. */
function publish(): void {
  (window as unknown as { __gpuSpriteGame: unknown }).__gpuSpriteGame = {
    seed: game?.runSeed ?? null,
    creatures: bestiary?.species.length ?? 0,
    turn: game?.turn ?? 0,
    backend,
    result: game?.result ?? "none",
  };
}

function draw(): void {
  if (game) render(view, game);
  publish();
}

function showMessage(text: string): void {
  game = undefined;
  view.overlay.hidden = true;
  message.textContent = text;
  message.hidden = false;
  publish();
}

/** The shared " in … ms on …" tail of both the success and failure generation lines. */
function backendTail(run: BestiaryRun): string {
  return ` in ${run.ms.toFixed(1)} ms on ${run.backend}${run.adapter ? ` (${run.adapter})` : ""}`;
}

async function startRun(seed: number): Promise<void> {
  const token = ++runToken;
  runSeed = seed;
  bestiary = undefined;
  game = undefined;
  seedLabel.textContent = String(seed);
  message.hidden = true;
  view.overlay.hidden = true;
  generation.textContent = "Generating bestiary…";
  publish();
  let run: BestiaryRun | undefined;
  try {
    run = await generateBestiary(seed, (seeds) => generator.generateMany(seeds), () => performance.now(), () => token === runToken);
  } catch (err) {
    if (token === runToken) {
      generation.textContent = "";
      showMessage(`Could not generate sprites: ${(err as Error).message}`);
    }
    return;
  }
  if (!run) return; // replaced by a newer run
  if (!run.bestiary) {
    generation.textContent = `${run.candidates} candidates → no bestiary${backendTail(run)}`;
    showMessage("Could not build a bestiary for this seed");
    return;
  }
  backend = run.backend;
  bestiary = run.bestiary;
  generation.textContent = `${run.candidates} candidates → ${1 + SPECIES_COUNT} creatures${backendTail(run)}`;
  game = newGame(seed, run.bestiary);
  mapCanvas.focus();
  draw();
}

function newRun(): void {
  const seed = crypto.getRandomValues(new Uint32Array(1))[0];
  history.replaceState(null, "", `?seed=${seed}`);
  void startRun(seed);
}

function retry(): void {
  if (!bestiary) return;
  game = newGame(runSeed, bestiary);
  mapCanvas.focus();
  draw();
}

window.addEventListener("keydown", (event) => {
  if (!game || event.ctrlKey || event.metaKey || event.altKey) return;
  if (game.result !== "playing") {
    if (event.key === "r" || event.key === "R") retry();
    else if (event.key === "n" || event.key === "N") newRun();
    else if (KEYS[event.key]) event.preventDefault();
    return;
  }
  const action = KEYS[event.key];
  if (!action) return;
  event.preventDefault();
  game = step(game, action);
  draw();
});

$("#retry").addEventListener("click", retry);
$("#new-run").addEventListener("click", newRun);
$("#new-run-end").addEventListener("click", newRun);
copyLink.addEventListener("click", () => {
  // `navigator.clipboard` is undefined on insecure origins; route that through the same failure path
  // instead of throwing an unhandled error.
  Promise.resolve()
    .then(() => navigator.clipboard.writeText(location.href))
    .then(
      () => (copyLink.textContent = "Copied"),
      () => (copyLink.textContent = "Copy failed"),
    );
  setTimeout(() => (copyLink.textContent = "Copy link"), 1500);
});

const param = new URLSearchParams(location.search).get("seed");
if (param === null) {
  newRun();
} else {
  const seed = parseRunSeed(param);
  if (seed === undefined) {
    seedLabel.textContent = "none";
    generation.textContent = "";
    showMessage(`Invalid seed "${param}"`);
  } else {
    void startRun(seed);
  }
}
