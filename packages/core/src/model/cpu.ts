import type { DecodedLayer } from "./types.ts";

/**
 * Reference runtime. Float32 accumulation, sequential over inputs, matching
 * packages/training/quantize.py:forward_sequential operation for operation.
 */
export class CpuModel {
  readonly inputDim: number;
  readonly outputDim: number;
  private readonly layers: DecodedLayer[];

  constructor(layers: DecodedLayer[]) {
    if (layers.length === 0) throw new Error("model has no layers");
    this.layers = layers;
    this.inputDim = layers[0].in;
    this.outputDim = layers[layers.length - 1].out;
  }

  forward(input: Float32Array, batch: number): Float32Array {
    if (input.length < batch * this.inputDim) {
      throw new RangeError(`input has ${input.length} values, need ${batch * this.inputDim}`);
    }
    let x = input;
    for (const layer of this.layers) {
      const y = new Float32Array(batch * layer.out);
      for (let b = 0; b < batch; b++) {
        const inBase = b * layer.in;
        const outBase = b * layer.out;
        for (let o = 0; o < layer.out; o++) {
          let acc = layer.bias[o];
          const wBase = o * layer.in;
          for (let i = 0; i < layer.in; i++) {
            acc = Math.fround(acc + Math.fround(layer.weights[wBase + i] * x[inBase + i]));
          }
          y[outBase + o] = layer.relu && acc < 0 ? 0 : acc;
        }
      }
      x = y;
    }
    return x;
  }
}
