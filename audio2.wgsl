struct AudioParameters
{
  bufferDim: u32, // Cubic root of audio buffer size to match grid dimension of compute shader invocations
  sampleRate: u32 // Sample rate as per WebAudio context
}

const BPM = 120.0; // 160.0;
const PI = 3.141592654;
const TAU = 6.283185307;
const TIME_PER_BEAT = 60.0 / BPM / 4.0;
const TIME_PER_PATTERN = 60.0 / BPM * 4.0;

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

fn adsr(time: f32, att: f32, dec: f32, sus: f32, susL: f32, rel : f32) -> f32
{
  var amp = 0.f;
  if(time <= att) 
  {
    amp = time / att;
  } 
  else if(time <= att + dec)
  {
    amp = ((susL - 1.0) / dec) * (time - att) + 1.0;
  }
  else if(time <= att + dec + sus)
  {
    amp = susL;
  }
  else if(time <= att + dec + sus + rel)
  {
    amp = -(susL / rel) * (time - att - dec - sus) + susL;
  }

  return amp;
}

fn bass(time: f32, freq: f32) -> f32
{
  var pitch = freq + adsr(time, 0.2f, 0.80f, 0.1f, 0.2f, 0.1f) * 5.f;
  var env = adsr(time, 0.01f, 0.396f, 0.01f, 0.1f, 0.1f);
  var bass = sin(time * TAU * pitch);

  return bass * env;
}

fn hihat(time: f32, freq: f32) -> f32
{
  let amp = exp(-70.0 * time);
  return amp * noise(time * freq).x;
}

fn kick(time: f32, freq: f32) -> f32
{
  var pitch = freq + adsr(time, 0.01f, 0.30f, 0.01f, 0.1f, 0.01f) * 65.f;
  var env = adsr(time, 0.01f, 0.196f, 0.01f, 0.0f, 0.001f);
  var kick = sin(time * TAU * pitch);
  
  return kick * env;
}

// TBD relevant note constants below
const D4 = 42;
const B3 = 39;
const A3 = 37;
const G3 = 35;
const F3 = 33;
const A2 = 25;
const F1 = 9;
const D1 = 6;

// convert a note to it's frequency representation
fn noteToFreq(note: u32) -> f32
{
  return (440.0 / 32.0) * pow(2.0, (f32(note) - 9.0) / 12.0);
}

fn pling(time: f32, freq : f32) -> f32
{
  return sin(time * TAU * freq) * exp(-6 * time);
}

// FIXME: workaround to support glsl style modulo 
// appears % in wgsl is not the same as mod in glsl
// https://github.com/gpuweb/gpuweb/issues/3987#issuecomment-1528783750
fn modulo_euclidean (a: f32, b: f32) -> f32 
{
	var m = a % b;
	if (m < 0.0) 
  {
		if (b < 0.0) 
    {
			m -= b;
		} 
    else 
    {
			m += b;
		}
	}
	return m;
}

// this should be rewritten into something more generalized
const kickPatternLength = 16;
const kickPattern = array<u32, kickPatternLength>
(D1,0,0,0, D1,0,D1,0, 0,0,D1,0, D1,0,0,D1);

const hihatPatternLength = 16;
const hihatPattern = array<u32, hihatPatternLength>
(0,0,D1,0, 0,0,D1,0, 0,0,D1,0, 0,0,D1,D1);

const bassPatternLength = 16;
const bassPattern = array<u32, bassPatternLength>
(0,0,F3,0, 0,0,F3,0, 0,F3,0,0, F3,G3,A3,B3);

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

  // Samples are calculated in mono and then written to left/right
  var result = vec2(0.0);

  // kick
  // FIXME: generalize pattern stuff
  for(var i=0;i<kickPatternLength;i++)
  {
    let note = kickPattern[i];
    let noteFreq = noteToFreq(note);
    let beatTime = f32(i) * TIME_PER_BEAT;
    let noteTime = modulo_euclidean(time - beatTime, TIME_PER_PATTERN);
    let noteOn = sign(f32(note));

    result += vec2f(0.5 * kick(noteTime, noteFreq) * noteOn);
  }

  // hihat
  // FIXME: generalize pattern stuff
  for(var i=0;i<hihatPatternLength;i++)
  {
    let note = hihatPattern[i];
    let noteFreq = noteToFreq(note);
    let beatTime = f32(i) * TIME_PER_BEAT;
    let noteTime = modulo_euclidean(time - beatTime, TIME_PER_PATTERN);
    let noteOn = sign(f32(note));

    // FIXME: hihat doesn't really have a frequency right now
    result += vec2f(0.15 * hihat(noteTime, noteFreq) * noteOn);
  }

  // bass
  // FIXME: generalize pattern stuff
  for(var i=0;i<bassPatternLength;i++)
  {
    let note = bassPattern[i];
    let noteFreq = noteToFreq(note);
    let beatTime = f32(i) * TIME_PER_BEAT;
    let noteTime = modulo_euclidean(time - beatTime, TIME_PER_PATTERN);
    let noteOn = sign(f32(note));

    result += vec2f(0.25 * bass(noteTime, noteFreq) * noteOn);
  }

  // Write 2 floats between -1 and 1 to output buffer (stereo)
  buffer[sample] = clamp(result, vec2f(-1), vec2f(1));
}
