import { LATENT_DIM, type Generator, type Sprite } from "gpu-sprite";
import { drawSprite } from "./grid.ts";

export class DetailPanel {
  private readonly root: HTMLElement;
  private readonly generator: Generator;
  private readonly onChange: ((sprite: Sprite) => void) | undefined;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly inputs: HTMLInputElement[] = [];
  private z = new Float32Array(LATENT_DIM);

  constructor(root: HTMLElement, generator: Generator, onChange?: (sprite: Sprite) => void) {
    this.root = root;
    this.generator = generator;
    this.onChange = onChange;
    this.ctx = (root.querySelector("#detail-canvas") as HTMLCanvasElement).getContext("2d")!;
    const sliders = root.querySelector("#sliders")!;
    for (let i = 0; i < LATENT_DIM; i++) {
      const input = document.createElement("input");
      input.type = "range";
      input.min = "-3";
      input.max = "3";
      input.step = "0.01";
      input.value = "0";
      input.setAttribute("aria-label", `latent ${i}`);
      input.addEventListener("input", () => {
        this.z[i] = Number(input.value);
        void this.redraw();
      });
      sliders.appendChild(input);
      this.inputs.push(input);
    }
    root.querySelector("#close-detail")!.addEventListener("click", () => this.hide());
  }

  async show(seed: number, z: Float32Array): Promise<void> {
    this.z = Float32Array.from(z);
    this.inputs.forEach((input, i) => (input.value = String(z[i])));
    (this.root.querySelector("#detail-seed") as HTMLElement).textContent = `seed ${seed}`;
    const copy = this.root.querySelector("#copy-seed") as HTMLButtonElement;
    copy.onclick = () => void navigator.clipboard?.writeText(String(seed));
    this.root.hidden = false;
    await this.redraw();
  }

  hide(): void {
    this.root.hidden = true;
  }

  private async redraw(): Promise<void> {
    const sprite = await this.generator.fromLatent(this.z);
    drawSprite(this.ctx, sprite, 12);
    this.onChange?.(sprite);
  }
}
