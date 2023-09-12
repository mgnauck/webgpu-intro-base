const BUFFER_DIM = 256; // Hardcoded :( This must equal the AUDIO_BUFFER_DIM in main.js
const BPM = 160.0;
const PI = 3.141592654;
const TAU = 6.283185307;

fn timeToBeat(t: f32) -> f32
{
  return t / 60.0 * BPM;
}

fn beatToTime(b: f32) -> f32
{
  return b / BPM * 60.0;
}

fn sine(phase: f32) -> f32
{
  return sin(TAU * phase);
}

// Suboptimal random (ripped from somewhere)
fn rand(co: vec2f) -> f32
{
  return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn noise(phase: f32) -> vec4f
{
  let uv = phase / vec2f(0.512, 0.487);
  return vec4f(rand(uv));
}

fn kick(time: f32) -> f32
{
  let amp = exp(-5.0 * time);
  let phase = 120.0 * time - 15.0 * exp(-60.0 * time);
  return amp * sine(phase);
}

fn hihat(time: f32) -> f32
{
  let amp = exp(-40.0 * time);
  return amp * noise(time * 110.0).x;
}

@group(0) @binding(0) var<storage, read_write> buffer: array<vec2f>;

@compute @workgroup_size(4, 4, 4)
fn audioMain(@builtin(global_invocation_id) globalId: vec3u)
{
  if(globalId.x >= BUFFER_DIM || globalId.y >= BUFFER_DIM || globalId.z >= BUFFER_DIM) {
    return;
  }

  // Calculate current sample from given buffer id
  let sample = dot(globalId, vec3u(1, BUFFER_DIM, BUFFER_DIM * BUFFER_DIM));
  
  let time = f32(sample) / 44100.0;
  let beat = timeToBeat(time);

  // Samples are calculated in mono and then written to left/right

  // Kick
  var result = vec2f(0.6 * kick(beatToTime(beat % 1.0)));

  // Hihat
  result += vec2f(0.3 * hihat(beatToTime((beat + 0.5) % 1.0)));

  // Write 2 floats between -1 and 1 to output buffer (stereo)
  buffer[sample] = clamp(result, vec2f(-1), vec2f(1));
}
