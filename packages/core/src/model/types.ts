export type Activation = "relu" | "none";

export interface LayerSpec {
  in: number;
  out: number;
  activation: Activation;
  /** Per-tensor symmetric int6 scale, float32-representable. */
  scale: number;
  /** Float32-representable biases rounded to 3 decimals, length `out`. */
  bias: number[];
}

export interface ModelSpec {
  format: 2;
  seedVersion: 1;
  latent: number;
  checkpoint: string;
  layers: LayerSpec[];
  /** One character per int6 weight (see `decode.ts`), all layers concatenated, each row-major (out, in). */
  weights: string;
}

export interface DecodedLayer {
  in: number;
  out: number;
  relu: boolean;
  /** Dequantized float32 weights, row-major (out, in). */
  weights: Float32Array;
  bias: Float32Array;
}
