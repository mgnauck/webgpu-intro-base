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

fn rand(co: vec2f) -> f32 {
  // Bad rand, ripped from somewhere
  return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn noise(phase: f32) -> vec4f {
  let uv = phase / vec2f(0.512, 0.487);
  return vec4f(rand(uv));
}

fn kick(time: f32) -> f32 {
  let amp = exp(-5.0 * time);
  let phase = 120.0 * time - 15.0 * exp(-60.0 * time);
  return amp * sine(phase);
}

fn hiHat(time: f32) -> f32 {
  let amp = exp(-40.0 * time);
  return amp * noise(time * 110.0).x;
}

@group(0) @binding(0) var outputTexture: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn audioMain(@builtin(global_invocation_id) globalId: vec3u) {
  
  if (globalId.x >= 4096 || globalId.y >= 4096) {
    return;
  }

  // Calculate current time (= current sample) from x/y of our buffer
  let time = f32(4096 * globalId.y + globalId.x) / 44100.0;
  let beat = timeToBeat(time);

  // At the moment samples are calculated in mono and then written to left/right

  // Kick
  var result = vec2f(0.6 * kick(beatToTime(beat % 1.0)));

  // Hihat
  result += vec2f(0.3 * hiHat(beatToTime((beat + 0.5) % 1.0)));

  textureStore(
    outputTexture,
    // Position in output buffer
    vec2u(globalId.x, globalId.y),
    // Write 2 floats between -1 and 1 to output buffer (stereo)
    vec4f(clamp(result, vec2f(-1), vec2f(1)), 0, 1));
}
