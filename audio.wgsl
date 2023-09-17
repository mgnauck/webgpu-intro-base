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

// TODO: translate into function
// https://graphtoy.com/?f1(x,t)=max(0,min(x/1,max(0.5,1-(x-1)*(1-0.5)/1)*min(1,1-max(0,(x-1-1-1)/1))))&v1=true&f2(x,t)=&v2=false&f3(x,t)=&v3=false&f4(x,t)=&v4=false&f5(x,t)=&v5=false&f6(x,t)=&v6=false&grid=1&coords=2.0657836566653387,-0.2538551861155832,3.823569812524288
fn adsr(time: f32, att: f32, dec: f32, sus: f32, susL: f32, rel : f32) -> f32
{
  return 
    max(  0.0,
          min( 
            time / att, 
            max( 
              susL, 
              1.0 - (time - att) * (1.0 - susL) / dec) *
              min(1.0, 1.0 - max(0.0, (time - att - dec - sus) / rel)
            )
          )
    );
}

fn bass(time: f32, freq: f32) -> f32
{
  let dist = 1.0 + 0.20 * sin(time * TAU * 5.0);
  let phase = freq * time + 5.0 * sin(time*TAU*1.0); 
  let env = exp(-4 * time);
  let bass = atan2(sin(TAU * phase), dist);

  return bass * env;
}

fn hihat(time: f32, freq: f32) -> f32
{
  let dist = 1.0 + 0.5 * sin(time * TAU * 3.0);
  let env = exp(-20.0 * time);
  let hihat = atan2( noise(time * freq).x, dist);
  return hihat * env; 
}

// inspired by:
// https://www.shadertoy.com/view/7ljczz
fn kick(time: f32, freq: f32) -> f32
{
  let dist = 1.0 + 0.5 * sin(time * TAU * 50.0);
  let phase = freq * time - 8.0 * exp( -20.0 * time ) - 3.0 * exp( -600.0 * time );
  let env = exp( -5.0 * time );
  let kick = atan2( sin(TAU * phase), dist);

  return kick * env;
}

fn clap(time: f32, freq : f32) -> f32
{
  // TODO
  let clap = 0.0;
  let env = 0.0;

  return clap * env;
}

// TBD relevant note constants below
const D4 = 62;
const B3 = 39;
const A3 = 37;
const G3 = 35;
const FIS3 = 34;
const F3 = 33;
const FIS1 = 30;
const E1 = 28;
const A2 = 25;
const FIS2 = 22;
const DIS1 = 27;
const D1 = 26;
const F1 = 9;
const NONE = -1;

// convert a note to it's frequency representation
fn noteToFreq(note: f32) -> f32
{
  return 440.0 * pow(2.0, (f32(note) - 69.0) / 12.0);
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
const CLAP_CHANNEL = 3;
const CHANNELS = 4;
const ROWS = 16;
const PATTERN = array<array<f32,CHANNELS>, ROWS>
( 
  array<f32,CHANNELS>( FIS1,   D4, NONE,   33 ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 
  array<f32,CHANNELS>( NONE,   D4,   33, NONE ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 

  array<f32,CHANNELS>( FIS1,   D4, NONE, NONE ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 
  array<f32,CHANNELS>( NONE,   D4,   33, NONE ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 

  array<f32,CHANNELS>( FIS1,   D4, NONE, NONE ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 
  array<f32,CHANNELS>( NONE,   D4, NONE, NONE ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 

  array<f32,CHANNELS>( FIS1,   D4,   33, NONE ), 
  array<f32,CHANNELS>( NONE, NONE, NONE, NONE ), 
  array<f32,CHANNELS>( NONE,   D4,   36, NONE ), 
  array<f32,CHANNELS>( FIS1,   D4, NONE, NONE ), 
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
    let kickNoteNext = PATTERN[(i+1)%ROWS][KICK_CHANNEL];
    let kickNoteFreq = noteToFreq(kickNote);
    let kickNoteOn = 1.0 * sign(kickNote+1.0);

    if(time > TIME_PER_PATTERN * 1)
    {
      output += vec2f(0.30 * kick(noteTime, kickNoteFreq) * kickNoteOn);
    }

    // hihat
    let hihatNote = PATTERN[i][HIHAT_CHANNEL];
    let hihatNoteFreq = noteToFreq(hihatNote);
    let hihatNoteOn = sign(hihatNote+1.0);
    output += vec2f(0.05 * hihat(noteTime, hihatNoteFreq) * hihatNoteOn);

    // bass
    let bassNote = PATTERN[i][BASS_CHANNEL];
    let bassNoteFreq = noteToFreq(bassNote);
    let bassNoteOn = sign(bassNote+1.0);
    output += vec2f(0.4 * bass(noteTime, bassNoteFreq) * bassNoteOn);

    // clap
    //let clapNote = PATTERN[i][CLAP_CHANNEL];
    //let clapNoteFreq = noteToFreq(clapNote);
    //let clapNoteOn = sign(clapNote+1.0);
    //output += vec2f(0.3 * clap(noteTime, clapNoteFreq) * clapNoteOn);
  }

  // Write 2 floats between -1 and 1 to output buffer (stereo)
  buffer[sample] = clamp(output, vec2f(-1), vec2f(1));
}
