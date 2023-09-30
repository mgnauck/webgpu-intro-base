const BPM = 130.0;
const PI = 3.141592654;
const TAU = 6.283185307;
const TIME_PER_BEAT = 60.0 / BPM / 4.0;
const TIME_PER_PATTERN = 60.0 / BPM * 4.0;
const PATTERN_COUNT = 4;
const KICK = 0;
const HIHAT = 1;
const BASS = 2;
const DURCH = 3;

// Suboptimal random (ripped from somewhere)
fn rand2(co: vec2f) -> f32
{
  return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn noise(phase: f32) -> vec4f
{
  let uv = phase / vec2f(0.512, 0.487);
  return vec4f(rand2(uv));
}

fn hihat(time: f32, freq: f32) -> f32
{
  let dist = 0.85;
  let out = noise(time * freq).x;
  let env = exp(-90.0 * time);
  let hihat = atan2(out, 1.0 - dist);
  return hihat * env; 
}

fn bass3(time: f32, freq: f32) -> f32
{
  if(time < 0.0)
  {
    return 0.0;
  }

  let phase = freq * time; 
  let env = exp(-1.0 * time);
  let bass = atan2(sin(TAU * phase), 1.0);

  return bass * env;
}

// inspired by:
// https://www.shadertoy.com/view/7lpatternIndexczz
fn kick(time: f32, freq: f32) -> f32
{
  if(time < 0.0)
  {
    return 0.0;
  }

  let phase = freq * time - 8.0 * exp(-20.0 * time) - 3.0 * exp(-800.0 * time);
  let env = exp(-5.0 * time);
  let kick = atan2(sin(TAU * phase), 0.4);

  return kick * env;
}

fn sine(time: f32, freq: f32) -> f32
{
  return sin(time * TAU * freq);
}

fn sample1(gTime: f32, time: f32, freq: f32) -> f32
{
  let lfo = sin(gTime * TAU * 0.001) * 0.5;
  let lfo2 = sin(gTime * TAU * lfo) * 0.5;

  let voices = 15.0;
  var out = 0.0;
  for(var v=1.0;v<=voices;v+=1.0)
  {
    let lfo3 = sin((gTime * v) * TAU * 0.05) * 0.1;
    let detune = lfo3 * v;
    let f0 = freq * (v*0.5);
    out += (1.0/voices) * sine(time, f0+detune+lfo-55.0);
  }

  out = atan2(out, 1.0-lfo2*0.2);
  let env = exp(-2.0*time);

  return out * env;
}

fn sample1lpf(gTime: f32, time: f32, freq: f32) -> f32
{
  let c = 0.8 - (sin(gTime * TAU * 0.001) * 0.3);
  let r = 0.8 - cos(gTime * TAU * 0.1) * 0.4;
  let dt = 1.0/f32(params[0]);

  var v0 = 0.0;
  var v1 = 0.0;
  for(var j=0u; j<96; j++)
  {
    let input = sample1(gTime, time - f32(j)*dt, freq);
    v0 = (1.0-r*c)*v0 - (c)*v1 + (c)*input;
    v1 = (1.0-r*c)*v1 + (c)*v0;
   }
  
  return v1;
}

fn addSample(idx: u32, gTime: f32, time: f32, pat: u32, dur: f32, freq: f32, amp: f32) -> f32
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
  else if(idx == BASS)
  {
    return bass3(sampleTime, freq) * env;
  }
  else if(idx == DURCH)
  {
    return sample1lpf(gTime, sampleTime, freq) * env;
  }
  
  return 0.0;
}

fn isPattern(time: f32, start: u32, end: u32) -> bool
{
  let patternIndex = u32(time / TIME_PER_PATTERN) % PATTERN_COUNT; 

  return patternIndex >= start && patternIndex < end;
}

@group(0) @binding(0) var<storage> params: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer: array<vec2f>;

@compute @workgroup_size(4, 4, 4)
fn cM(@builtin(global_invocation_id) globalId: vec3u)
{
  // Calculate current sample from given buffer id
  let sample = dot(globalId, vec3u(1, 256, 256 * 256));
  let time = f32(sample) / f32(params[0]);
  let patternTime = time % TIME_PER_PATTERN;

  // Samples are calculated in mono and then written to left/right
  var output = vec2(0.0);

  output += addSample(KICK, time, patternTime,  0, 1.0, 55.0, 0.4 );
  output += addSample(KICK, time, patternTime,  4, 1.0, 55.0, 0.4 );
  output += addSample(KICK, time, patternTime,  8, 1.0, 55.0, 0.4 );
  output += addSample(KICK, time, patternTime, 12, 1.0, 55.0, 0.4 );

  output += addSample(HIHAT, time, patternTime,  0, 0.1, 55.0, 0.05);
  output += addSample(HIHAT, time, patternTime,  4, 0.1, 55.0, 0.25);
  output += addSample(HIHAT, time, patternTime, 12, 0.1, 55.0, 0.15);
  output += addSample(HIHAT, time, patternTime, 15, 0.2, 55.0, 0.20);

  output += addSample(BASS, time, patternTime,   2, 0.5, 110.0, 0.4 );
  output += addSample(BASS, time, patternTime,   6, 0.5, 110.0, 0.3 );
  output += addSample(BASS, time, patternTime,  10, 0.5, 110.0, 0.2 );
  output += addSample(BASS, time, patternTime,  14, 0.5, 110.0, 0.4 );

  output += addSample(DURCH, time, patternTime,  0, 0.5, 110.00, 0.9 );

/*
  if(isPattern(time, 0, 2))
  {
    output += addSample(KICK, time, patternTime,  0, 1.0, 55.0, 0.5 );
    output += addSample(KICK, time, patternTime,  6, 1.0, 55.0, 0.4 );
    output += addSample(KICK, time, patternTime, 12, 1.0, 55.0, 0.5 );
    output += addSample(HIHAT, time, patternTime,  0, 1.0, 55.0, 0.15);
    output += addSample(HIHAT, time, patternTime,  4, 0.25, 55.0, 0.25);
    output += addSample(HIHAT, time, patternTime, 12, 0.25, 55.0, 0.15);
  }
  if(isPattern(time, 0, 2))
  {
    output += addSample(BASS, time, patternTime,  0, 1.0, 110.0, 0.66 );
    output += addSample(BASS, time, patternTime,  4, 4.0, 55.0, 0.56 );
    output += addSample(BASS, time, patternTime,  6, 1.0, 55.0+27.5, 0.56 );
    output += addSample(BASS, time, patternTime, 10, 1.0, 55.0, 0.56 );
    output += addSample(BASS, time, patternTime, 12, 1.0, 55.0+27.5, 0.56 );
  }
*/

/*
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
*/

  // Write 2 floats between -1 and 1 to output buffer (stereo)
  buffer[sample] = clamp(output, vec2f(-1), vec2f(1));
}