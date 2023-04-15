@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var blitSampler: sampler;

@fragment
fn main(@location(0) texCoord: vec2f) -> @location(0) vec4f {
  return textureSample(inputTexture, blitSampler, texCoord);
}
