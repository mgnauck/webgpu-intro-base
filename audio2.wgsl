struct AudioParameters
{
  bufferDim: u32, // Cubic root of audio buffer size to match grid dimension of compute shader invocations
  sampleRate: u32 // Sample rate as per WebAudio context
}

const BPM = 60.0; // 160.0;
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

// TBD relevant note constants below
const F1 = 9;
const D1 = 6;

// convert a note to it's frequency representation
fn noteToFreq(note: u32) -> f32
{
  return (440.f / 32.f) * pow(2.f, ((f32(note) - 9.f) / 12.f));
}

fn pling(time: f32, freq : f32) -> f32
{
  return sin(time * TAU * freq) * exp(-6 * time);
}

@group(0) @binding(0) var<uniform> params: AudioParameters;
@group(0) @binding(1) var<storage, read_write> buffer: array<vec2f>;

@compute @workgroup_size(4, 4, 4)
fn audioMain(@builtin(global_invocation_id) globalId: vec3u)
{
  if(globalId.x >= params.bufferDim || globalId.y >= params.bufferDim || globalId.z >= params.bufferDim) {
    return;
  }

  // Calculate current sample from given buffer id
  let sample = dot(globalId, vec3u(1, params.bufferDim, params.bufferDim * params.bufferDim));
  
  let time = f32(sample) / f32(params.sampleRate);
  let beat = timeToBeat(time);

  // 4/4 beat, 16 entries
  let beatIndex = u32(beat * 4.0 % 16.0);
  //let beatTime = time; // beatToTime(beat); // % (4.0 / 16.0));

  // KICK,D1,4

  const notePattern = array<u32, 16>(60,60,0,0, 60,0,0,0, 60,60,0,0, 60,0,0,0);
                                  // ---------> t
                                  //            t=0...
                                  // t0..............................tX
                                  //   t0
                                  //      t0
                                  //        t0
                                  // 1 0 0 0  ....
                                  // 1 1 1 1 1 1 1 1 1 0 0 0 ...
  var result = vec2(0.f);

  for(var i=0;i<16;i++)
  {
    let note = notePattern[beatIndex];
    let noteNext = notePattern[(beatIndex+1)%16];
    let noteFreq = 440.f; // noteToFreq(note);
    let noteTime = (time + f32(i)*4.f/16.f) % 4.f; // max note length in seconds
    let noteOn = sign(f32(note)) * (1.f-sign(f32(noteNext)));

    result += vec2f(0.1 * pling(noteTime, noteFreq) * noteOn);
  }

  // Samples are calculated in mono and then written to left/right

  // Kick
  //var result = vec2f(0.6 * kick(beatToTime(beat % 1.0)));

  // Hihat
  //result += vec2f(0.3 * hihat(beatToTime((beat + 0.5) % 1.0)));

  // Write 2 floats between -1 and 1 to output buffer (stereo)
  buffer[sample] = clamp(result, vec2f(-1), vec2f(1));
}
