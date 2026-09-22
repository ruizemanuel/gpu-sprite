export type Activation = "relu" | "none";

export interface LayerSpec {
  in: number;
  out: number;
  activation: Activation;
  /** Per-tensor symmetric int8 scale, float32-representable. */
  scale: number;
  /** Float32-representable biases, length `out`. */
  bias: number[];
}

export interface ModelSpec {
  format: 1;
  seedVersion: 1;
  latent: number;
  checkpoint: string;
  layers: LayerSpec[];
  /** base64 of int8 weights, all layers concatenated, each row-major (out, in). */
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
