import type { DecodedLayer, ModelSpec } from "./types.ts";

function base64ToInt8(b64: string): Int8Array {
  const bin = atob(b64);
  const out = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    out[i] = (bin.charCodeAt(i) << 24) >> 24;
  }
  return out;
}

/** Decode the embedded int8 payload into dequantized float32 layers. Done once per generator instance. */
export function decodeModel(spec: ModelSpec): DecodedLayer[] {
  if (spec.format !== 1) throw new Error(`unsupported model format ${String(spec.format)}`);
  const q = base64ToInt8(spec.weights);
  const expected = spec.layers.reduce((n, l) => n + l.in * l.out, 0);
  if (q.length !== expected) throw new Error(`weights payload has ${q.length} bytes, layers need ${expected}`);
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
