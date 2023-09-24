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
const PATTERN_COUNT = 2;
const PATTERN_ROW_COUNT = 16;
const KICK = 0;
const HIHAT = 1;
const BASS = 2;
const DURCH = 3;

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
            smoothstep(0.0, 1.0, time / att),
            max( 
              susL, 
              smoothstep(0.0, 1.0, 1.0 - (time - att) * (1.0 - susL) / dec)) *
              min(1.0, 1.0 - max(0.0, (time - att - dec - sus) / rel)
            )
          )
    );
}

//////////////////////// INSTRUMENTS 

fn bass(time: f32, freq: f32) -> f32
{
  let phase = freq * time; 
  let env = exp(-3.0 * time);
  let bass = atan2( sin(TAU * phase), 1.0-0.25);

  return bass * env;
}

fn hihat(time: f32, freq: f32) -> f32
{
  let dist = 0.75;
  let out = noise(time * freq).x;
  let env = exp(-90.0 * time);
  let hihat = atan2(out, 1.0 - dist);
  return hihat * env; 
}

// inspired by:
// https://www.shadertoy.com/view/7lpatternIndexczz
fn kick(time: f32, freq: f32) -> f32
{
  let dist = 0.65;
  let phase = freq * time - 8.0 * exp(-20.0 * time) - 3.0 * exp(-800.0 * time);
  let env = exp(-4.0 * time);
  let kick = atan2(sin(TAU * phase), 1.0 - dist);

  return kick * env;
}

fn clap(time: f32, freq : f32) -> f32
{
  // TODO
  let clap = 0.0;
  let env = 0.0;

  return clap * env;
}

fn simple(time: f32, freq: f32) -> f32
{
  let lfo = 0.25 * sin(TAU * time * 0.01);
  let phase = time * (freq + lfo);
  let phase2 = time * freq;
  let env = exp(-2.0 * time);
  let o1 = sin(TAU * phase);
  let o2 = 2 * (fract(phase) - 0.5);

  return o1 * o2 * env;
}

fn simple2(time: f32, freq: f32) -> f32
{
  let c = 0.2;
  let r = 0.3;
  
  var v0 = 0.0;
  var v1 = 0.0;
  const cnt = 12;
  for(var i=0;i<cnt;i++)
  {
    let last = f32(cnt-i)*(1.0/f32(params.sampleRate));
    let t = time - last;
    let inp = simple(t, freq);
    v0 = (1.0-r*c)*v0  -  (c)*v1  + (c)*inp;
    v1 = (1.0-r*c)*v1  +  (c)*v0;
  }

  return v1;
}

// convert a note to it's frequency representation
fn noteToFreq(note: f32) -> f32
{
  return 440.0 * pow(2.0, (note - 69.0) / 12.0);
}

fn sine(time: f32, freq: f32) -> f32
{
  return sin(time * TAU * freq);
}

fn pling(time: f32, freq: f32) -> f32
{
  return sin(time * TAU * freq) * exp(-1.0*time);
}

fn sqr(time: f32, freq: f32) -> f32
{
  return (1.0 - 0.5) * 2.0 * atan(sin(time*freq) / 0.5) / PI;
}

fn sample1(time: f32, freq: f32) -> f32
{
  let lfo = sin(time * TAU * 0.1) * 1.0;
  let lfo2 = 0.5; // 1.0 + 0.5 * sin(time * TAU * 1) * 0.1;

  let voices = 13.0;
  var out = 0.0;
  for(var v=1.0;v<=voices;v+=1.0)
  {
    let detune = sin((time + v) * TAU * 0.25) * 0.25;
    out += 1.0/voices * sine(time, (freq+detune+lfo)*v);
  }

  out = atan2(out, 1.0-lfo2);
  let env = exp(-3.0*time);

  return out * env;
}

fn sample2(time: f32, freq: f32) -> f32
{
  return sample1(time, freq);
}

fn addSample(idx: u32, time: f32, pat: u32, dur: f32, freq: f32, amp: f32) -> f32
{
  let sampleTime = time - f32(pat) * TIME_PER_BEAT;

  if( sampleTime < 0.0 || sampleTime > dur )
  {
    return 0.0;
  }

  // if the duration causes the sound to shutdown we want
  // at least a quick ramp down to zero to not cause a click
  // this seems to work but better double check again!
  let env = amp * smoothstep(0.0, 0.05, dur-sampleTime);

  if(idx == KICK)
  {
    return kick(sampleTime, freq) * env;
  }
  else if(idx == HIHAT)
  {
    return hihat(sampleTime, freq) * env;
  }
  {
    return 0.0;
  }
}

fn isPattern(time: f32, start: u32, end: u32) -> bool
{
  let patternIndex = u32(time / TIME_PER_PATTERN) % PATTERN_COUNT; 

  return patternIndex >= start && patternIndex < end;
}

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
  let patternTime = time % TIME_PER_PATTERN;

  // Samples are calculated in mono and then written to left/right
  var output = vec2(0.0);

  // pattern 0
  if(isPattern(time, 0, 1))
  {
    output += addSample(KICK, patternTime,  0, 1.0, 55.0, 0.5 );
    output += addSample(KICK, patternTime,  4, 1.0, 55.0, 0.4 );
    output += addSample(KICK, patternTime,  8, 1.0, 55.0, 0.3 );
    output += addSample(KICK, patternTime, 12, 1.0, 55.0, 0.4 );

    output += addSample(HIHAT, patternTime,  0, 1.0, 55.0, 0.5 );
    output += addSample(HIHAT, patternTime,  2, 1.0, 55.0, 0.25 );
    output += addSample(HIHAT, patternTime,  4, 1.0, 55.0, 0.5 );
    output += addSample(HIHAT, patternTime,  6, 1.0, 55.0, 0.25 );
    output += addSample(HIHAT, patternTime,  8, 1.0, 55.0, 0.5 );
    output += addSample(HIHAT, patternTime, 10, 1.0, 55.0, 0.25 );
    output += addSample(HIHAT, patternTime, 12, 1.0, 55.0, 0.5 );
    output += addSample(HIHAT, patternTime, 14, 1.0, 55.0, 0.25 );
    output += addSample(HIHAT, patternTime, 15, 1.0, 55.0, 0.1 );
  }
  // pattern 1
  if(isPattern(time, 1, 2))
  {
    output += addSample(KICK, patternTime,  0, 1.0, 55.0, 0.5 );
    output += addSample(KICK, patternTime,  4, 1.0, 55.0, 0.4 );
    output += addSample(KICK, patternTime,  8, 1.0, 55.0, 0.3 );
    output += addSample(KICK, patternTime, 12, 1.0, 55.0, 0.4 );
    output += addSample(KICK, patternTime, 15, 1.0, 55.0, 0.5 );

    output += addSample(HIHAT, patternTime,  0, 1.0, 55.0, 0.5 );
    output += addSample(HIHAT, patternTime,  2, 1.0, 55.0, 0.25 );
    output += addSample(HIHAT, patternTime,  4, 1.0, 55.0, 0.5 );
    output += addSample(HIHAT, patternTime,  6, 1.0, 55.0, 0.25 );
    output += addSample(HIHAT, patternTime,  8, 1.0, 55.0, 0.5 );
    output += addSample(HIHAT, patternTime, 10, 1.0, 55.0, 0.25 );
    output += addSample(HIHAT, patternTime, 12, 1.0, 55.0, 0.5 );
    output += addSample(HIHAT, patternTime, 14, 1.0, 55.0, 0.25 );
    output += addSample(HIHAT, patternTime, 15, 1.0, 55.0, 0.1 );
  }

  // Write 2 floats between -1 and 1 to output buffer (stereo)
  buffer[sample] = clamp(output, vec2f(-1), vec2f(1));
}
