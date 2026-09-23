import { LATENT_DIM, SEED_VERSION } from "../seed.ts";
import type { DecodedLayer, ModelSpec } from "./types.ts";

/** Character `i` encodes the int6 weight `i - 31`: the base64 alphabet without "/". Mirrors `quantize.py`. */
const WEIGHT_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+";

function decodeInt6(payload: string): Int8Array {
  const out = new Int8Array(payload.length);
  for (let i = 0; i < payload.length; i++) {
    const v = WEIGHT_ALPHABET.indexOf(payload[i]);
    if (v < 0) throw new Error(`invalid weight character at ${i}`);
    out[i] = v - 31;
  }
  return out;
}

/** Decode the embedded int6 payload into dequantized float32 layers. Done once per generator instance. */
export function decodeModel(spec: ModelSpec): DecodedLayer[] {
  if (spec.format !== 2) throw new Error(`unsupported model format ${String(spec.format)}`);
  if (spec.seedVersion !== SEED_VERSION) throw new Error(`model seedVersion ${String(spec.seedVersion)} != ${SEED_VERSION}`);
  if (spec.latent !== LATENT_DIM) throw new Error(`model latent ${String(spec.latent)} != ${LATENT_DIM}`);
  if (spec.layers[0]?.in !== spec.latent) throw new Error(`model layers[0].in ${String(spec.layers[0]?.in)} != latent ${String(spec.latent)}`);
  const q = decodeInt6(spec.weights);
  const expected = spec.layers.reduce((n, l) => n + l.in * l.out, 0);
  if (q.length !== expected) throw new Error(`weights payload has ${q.length} values, layers need ${expected}`);
  const layers: DecodedLayer[] = [];
  let offset = 0;
  for (const layer of spec.layers) {
    const count = layer.in * layer.out;
    const scale = Math.fround(layer.scale);
    const weights = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      weights[i] = Math.fround(q[offset + i] * scale);
    }
    offset += count;
    if (layer.bias.length !== layer.out) throw new Error(`layer bias length ${layer.bias.length} != out ${layer.out}`);
    layers.push({ in: layer.in, out: layer.out, relu: layer.activation === "relu", weights, bias: Float32Array.from(layer.bias) });
  }
  return layers;
}
