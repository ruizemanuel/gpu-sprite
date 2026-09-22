export const WORKGROUP_SIZE = 256;

/**
 * One dense layer. One workgroup per batch item, one thread per output unit.
 * Params: batch, inDim, outDim, activation (1 = relu).
 */
export const DENSE_SHADER = /* wgsl */ `
struct Params { batch: u32, inDim: u32, outDim: u32, activation: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> xs: array<f32>;
@group(0) @binding(4) var<storage, read_write> ys: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let b = wg.x;
  let o = lid.x;
  if (b >= params.batch || o >= params.outDim) {
    return;
  }
  var acc: f32 = bias[o];
  let inBase = b * params.inDim;
  let wBase = o * params.inDim;
  for (var i: u32 = 0u; i < params.inDim; i = i + 1u) {
    acc = acc + weights[wBase + i] * xs[inBase + i];
  }
  if (params.activation == 1u) {
    acc = max(acc, 0.0);
  }
  ys[b * params.outDim + o] = acc;
}
`;
