const BPM = 130.0;
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
fn rand2(co: vec2f) -> f32
{
  return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn noise(phase: f32) -> vec4f
{
  let uv = phase / vec2f(0.512, 0.487);
  return vec4f(rand2(uv));
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
  if(time<0)
  {
    return 0.0;
  }

  let dist = 0.55;
  let phase = freq * time - 8.0 * exp(-20.0 * time) - 3.0 * exp(-800.0 * time);
  let env = exp(-3.0 * time);
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
    let last = f32(cnt-i)*(1.0/f32(params[0]));
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
  if(time<0.0) {
    return 0.0;
  }
  return sin(time * TAU * freq) * exp(-10.0*time);
}

fn sqr(time: f32, freq: f32) -> f32
{
  return (1.0 - 0.5) * 2.0 * atan(sin(time*freq) / 0.5) / PI;
}

fn sample1(time: f32, freq: f32) -> f32
{
  let lfo = sin(time * TAU * 0.1) * 8.0;
  let lfo2 = sin(time * TAU * lfo) * 0.5;

  let voices = 15.0;
  var out = 0.0;
  for(var v=1.0;v<=voices;v+=1.0)
  {
    let lfo3 = sin((time * v) * TAU * 0.05) * 1.0;
    let detune = lfo3 + v;
    let f0 = freq * (v*0.5);
    out += (1.0/voices) * sine(time, f0+detune+lfo-55.0);
  }

  out = atan2(out, 1.0-lfo2);
  let env = exp(-4.0*time);

  return out * env;
}

fn sample1lpf(gTime: f32, time: f32, freq: f32) -> f32
{
  let c = 0.8 - (sin(gTime * TAU * 0.001) * 0.3);
  let r = 0.4 + cos(gTime * TAU * 0.01) * 0.4;
  let dt = 1.0/f32(params[0]);

  var v0 = 0.0;
  var v1 = 0.0;
  for(var j=0u; j<128; j++)
  {
    let input = sample1(time - f32(j)*dt, freq);
    v0 = (1.0-r*c)*v0 - (c)*v1 + (c)*input;
    v1 = (1.0-r*c)*v1 + (c)*v0;
   }
  
  return v1;
}

fn sample2(time: f32, freq: f32) -> f32
{
  return sample1(time, freq);
}

fn rand(n: f32) -> f32 
{ 
  return fract(sin(n) * 43758.5453123); 
}

fn noise3030(time: f32) -> f32
{
  let t = time;

const NUM_LAYERS: u32 = 8; // Adjust for the number of noise layers
const LAYER_INTENSITY: f32 = 0.3; // Adjust for the intensity of each noise layer
const LAYER_SPEED: f32 = 1.0; // Adjust for the speed of each noise layer
const FREQUENCY_VARIATION: f32 = 0.1; // Adjust for frequency variation
const AMPLITUDE_VARIATION: f32 = 0.2; // Adjust for amplitude variation

 // Generate multiple noise layers
  var wind_sample: f32 = 0.0;

  for (var i = 0u; i < NUM_LAYERS; i++) {
      // Vary the intensity and speed of each layer
      let layer_intensity = LAYER_INTENSITY * f32(i + 1);
      let layer_speed = LAYER_SPEED * f32(i + 1);

      // Introduce frequency and amplitude variations
      let frequency_variation = FREQUENCY_VARIATION * (2.0 * rand(t * layer_speed) - 1.0);
      let amplitude_variation = 1.0 - AMPLITUDE_VARIATION * rand(t * layer_speed);

      // Generate a noise-like sample for each layer with variations
      let layer_sample = layer_intensity * amplitude_variation *
          (2.0 * rand(t * layer_speed + frequency_variation) - 1.0);


      // Combine the layer samples
      wind_sample += layer_sample;
  }

  return wind_sample;
}

fn saw(time: f32, freq: f32) -> f32
{
  if(time < 0.0)
  {
    return 0.0;
  }
  //return sin(time * TAU * freq);
  return (time * freq) % 1.0;
}

fn pad(time: f32, freq: f32) -> f32
{
  let env = 1.0;
  let detune = 0.8; // 2.0; // 0.01;
  let lfo = sin(time * TAU * 0.001) * 1.0;

  var notes: array<f32, 7> = array<f32, 7>
  (
    -36.0, -24.0, -12.0, 0.0, 12.0, 24.0, 36.0
//      0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 12.0
  );

  var detunes: array<f32, 7> = array<f32, 7>
  (
    -3.0 * detune,
    -2.0 * detune,
    -detune,
    0,
    detune,
    2.0 * detune,
    3.0 * detune
  );

  var output = 0.0;
  for (var i=0u; i<7u; i++) 
  {
    let curFreq = freq * exp2(notes[i]/12.0) + detunes[i];
    output += saw(time, curFreq + lfo);
  }
  output += noise(time).x*0.1;

  return clamp(atan2(output * 0.15, 0.4) * env, -1.0, 1.0);
}

fn sample999(time: f32, freq: f32) -> f32
{
  let c = 0.6 + sin(time * TAU * 0.01) * 0.4;
  let r = 0.4 + cos(time * TAU * 0.001) * 0.3;
  let dt = 1.0/f32(params[0]);

  var v0 = 0.0;
  var v1 = 0.0;
  for(var j=0u; j<128; j++)
  {
    let input = pad(time - f32(j)*dt, freq);
    v0 = (1.0-r*c)*v0 - (c)*v1 + (c)*input;
    v1 = (1.0-r*c)*v1 + (c)*v0;
   }
  
  return v1;
}

fn didge2(time: f32, freq: f32) -> f32
{
  if(time <0.0)
  {
    return 0.0;
  }

  var noiseFrequency: f32 = freq; // Adjust this value for the noise frequency
  var noiseAmplitude: f32 = 0.8; // Adjust this value for noise amplitude
  var turbulenceSpeed: f32 = 0.75; // Adjust this value for turbulence speed

  // Generate wind noise using Perlin noise
  var perlinNoise: f32 = perlinNoise2D(vec2f(time * turbulenceSpeed * noiseFrequency, 0.0));
  
  return noiseAmplitude * perlinNoise;
}

fn didge(time: f32, freq: f32) -> f32
{
  let lfo = sin(time * TAU * 0.001) * 2.0;
  let freq2 = freq + lfo;

  var windNoise = didge2(time, freq2);

  return windNoise;
}

fn perlinNoise2D(p: vec2f) -> f32 
{
    let i = floor(p);
    let f = fract(p);

    // Random values at the corners of the cell
    var a: f32 = rand3(i);
    var b: f32 = rand3(i + vec2f(1.0, 0.0));
    var c: f32 = rand3(i + vec2f(0.0, 1.0));
    var d: f32 = rand3(i + vec2f(1.0, 1.0));

    // Interpolate along x and y using smoothstep
    var u: f32 = f.x * f.x * (3.0 - 2.0 * f.x);
    var v: f32 = f.y * f.y * (3.0 - 2.0 * f.y);

    // Interpolate along x first
    var ab: f32 = mix(a, b, u);
    var cd: f32 = mix(c, d, u);

    // Interpolate along y
    return mix(ab, cd, v);
}

fn rand3(p: vec2f) -> f32 
{
    return fract(
      sin(dot(vec2f(p), vec2f(12.9898, 78.233))) * 43758.5453);
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

  //output += simple2(time, 55.0);

//  output += didge(time, 440.0);
//  output += sample999(time, 55.0); // noise3030(time);
//  output += sample1(time, 55.0); // noise3030(time);

//  output += addSample(KICK, patternTime,  0, 1.0, 55.0, 0.5 );
//  output += addSample(KICK, patternTime,  6, 1.0, 55.0, 0.5 );
//  output += addSample(BASS, patternTime,  0, 1.0, 55.0, 0.6 );
//  output += addSample(BASS, patternTime,  4, 100.0, 66.0, 0.6 );
//  output += addSample(HIHAT, patternTime,  4, 1.0, 55.0, 0.05);
//  output += addSample(HIHAT, patternTime,  8, 1.0, 55.0, 0.15);

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
