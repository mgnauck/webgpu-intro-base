struct AudioParameters
{
  bufferDim: u32, // Cubic root of audio buffer size to match grid dimension of compute shader invocations
  sampleRate: u32 // Sample rate as per WebAudio context
}

const BPM = 120.0;
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
const DIS1 = 27;
const D1 = 26;
const F1 = 9;
const NONE = -1;

// convert a note to it's frequency representation
fn noteToFreq(note: f32) -> f32
{
  return 440.0 * pow(2.0, (f32(note) - 69.0) / 12.0);
}

fn pling(time: f32, freq : f32) -> f32
{
  return sin(time * TAU * freq) * exp(-6 * time);
}

// Workaround to support euclidean modulo (glsl)
// https://github.com/gpuweb/gpuweb/issues/3987#issuecomment-1528783750
fn modulo_euclidean(a: f32, b: f32) -> f32 
{
	let m = a % b;
  return select(m, m + abs(b), m < 0.0);
}

const KICK_CHANNEL = 0;
const HIHAT_CHANNEL = 1;
const BASS_CHANNEL = 2;
const CHANNELS = 4;
const ROWS = 16;
const PATTERN = array<array<f32,CHANNELS>, ROWS>
( 
  array<f32,CHANNELS>( DIS1, NONE, NONE, NONE ), 
  array<f32,CHANNELS>( NONE, NONE,   G3, NONE ), 
  array<f32,CHANNELS>( NONE,   D4, NONE, NONE ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 

  array<f32,CHANNELS>( DIS1, NONE, NONE, NONE ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 
  array<f32,CHANNELS>( NONE,   D4, NONE, NONE ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 

  array<f32,CHANNELS>( DIS1, NONE, NONE, NONE ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 
  array<f32,CHANNELS>( NONE,   D4, NONE, NONE ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 

  array<f32,CHANNELS>( DIS1, NONE, NONE, NONE ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 
  array<f32,CHANNELS>( NONE,   D4, NONE, NONE ), 
  array<f32,CHANNELS>( DIS1,   D4, NONE, NONE ), 
);

@group(0) @binding(0) var<uniform> params: AudioParameters;
@group(0) @binding(1) var<storage, read_write> buffer: array<vec2f>;

@compute @workgroup_size(4, 4, 4)
fn audioMain(@builtin(global_invocation_id) globalId: vec3u)
{
  // Make sure workgroups align properly with buffer size, i.e. do not run beyond buffer dimension
  if(globalId.x >= params.bufferDim || globalId.y >= params.bufferDim || globalId.z >= params.bufferDim) {
    return;
  }

  // Calculate current sample from given buffer id
  let sample = dot(globalId, vec3u(1, params.bufferDim, params.bufferDim * params.bufferDim));
  let time = f32(sample) / f32(params.sampleRate);

  // Samples are calculated in mono and then written to left/right
  var output = vec2(0.0);

  for(var i=0;i<ROWS;i++)
  {
    let beatTime = f32(i) * TIME_PER_BEAT;
    let noteTime = modulo_euclidean(time - beatTime, TIME_PER_PATTERN);

    // kick
    let kickNote = PATTERN[i][KICK_CHANNEL];
    let kickNoteFreq = noteToFreq(kickNote);
    let kickNoteOn = sign(kickNote+1.0);

    output += vec2f(0.25 * kick(noteTime, kickNoteFreq) * kickNoteOn);

    // hihat
    let hihatNote = PATTERN[i][HIHAT_CHANNEL];
    let hihatNoteFreq = noteToFreq(hihatNote);
    let hihatNoteOn = sign(hihatNote+1.0);

    output += vec2f(0.25 * hihat(noteTime, hihatNoteFreq) * hihatNoteOn);

    // bass
    let bassNote = PATTERN[i][BASS_CHANNEL];
    let bassNoteFreq = noteToFreq(bassNote);
    let bassNoteOn = sign(bassNote+1.0);

    // this line kills it w/o error message :(
    // output += vec2f(0.25 * bass(noteTime, bassNoteFreq) * bassNoteOn);
  }

  // Write 2 floats between -1 and 1 to output buffer (stereo)
  buffer[sample] = clamp(output, vec2f(-1), vec2f(1));
}
