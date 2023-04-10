const BPM = 160.0;
const PI = 3.141592654;
const TAU = 6.283185307;

fn timeToBeat(t: f32) -> f32 {
  return t / 60.0 * BPM;
}

fn beatToTime(b: f32) -> f32 {
  return b / BPM * 60.0;
}

fn sine(phase: f32) -> f32 {
  return sin(TAU * phase);
}

fn rand(co: vec2<f32>) -> f32 {
  return fract(sin(dot(co, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn noise(phase: f32) -> vec4<f32> {
  let uv = phase / vec2<f32>(0.512, 0.487);
  return vec4<f32>(rand(uv));
}

fn kick(time: f32) -> f32 {
  let amp = exp(-5.0 * time);
  let phase = 120.0 * time - 15.0 * exp(-60.0 * time);
  return amp * sine(phase);
}

fn hiHat(time: f32) -> vec2<f32> {
  let amp = exp(-40.0 * time);
  return amp * noise(time * 110.0).xy;
}

@group(0) @binding(0) var outputTexture: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  
  if (globalId.x >= 4096 || globalId.y >= 4096) {
    return;
  }

  let time = f32(4096 * globalId.y + globalId.x) / 44100.0;
  let beat = timeToBeat(time);

  var res = vec2<f32>(0.6 * kick(beatToTime(beat % 1.0)));

  res += 0.3 * hiHat(beatToTime((beat + 0.5) % 1.0));

  textureStore(
    outputTexture,
    vec2<u32>(globalId.x, globalId.y),
    vec4<f32>(clamp(res, vec2<f32>(-1.0), vec2<f32>(1.0)), 0.0, 1.0));
}
