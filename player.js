const FULLSCREEN = false;
const BPM = 120;

const CANVAS_WIDTH = 1024; // Careful, this is also hardcoded in the shader!!
const CANVAS_HEIGHT = CANVAS_WIDTH / 1.6;

const BUFFER_DIM = 256; // Used for audio buffer and grid buffer

let audioContext;
let audioBufferSourceNode;

let device;
let uniformBuffer;
let gridBuffer = [];
let rulesBuffer;
let bindGroup = [];
let pipelineLayout;
let computePipeline;
let renderPipeline;
let renderPassDescriptor;

let canvas;
let context;

let view = [];
let grid = new Uint32Array(3 + (BUFFER_DIM ** 3));
let rules = new Uint32Array(1 + 2 * 27);
let updateDelay = 0.5;

let startTime;
let timeInBeats = 0;
let lastSimulationUpdateTime = 0;
let simulationIteration = 0;
let activeRuleSet;
let activeSimulationEventIndex = -1;
let activeCameraEventIndex = -1;

const RULES = [
  0, // not in use, leave it here, we need it for enable/disable magic
  2023103542460421n, // clouds-5, key 0
  34359738629n, // 4/4-5, key 1
  97240207056901n, // amoeba-5, key 2
  962072678154n, // pyro-10, key 3
  36507219973n, // framework-5, key 4
  96793530464266n, // spiky-10, key 5
  1821066142730n, // builder-10, key 6
  96793530462218n, // ripple-10, key 7
  37688665960915591n, // shells-7, key 8
  30064771210n, // pulse-10, key 9
  4294970885n, // more-builds-5, key )
];

const SIMULATION_EVENTS = [
{ time: 0, obj: { ruleSet: 3, delta: -0.320 } },
{ time: 40, obj: { ruleSet: 4, delta: 0.320 } },
{ time: 60, obj: { ruleSet: 3, delta: 0.05 } },
{ time: 80, obj: { ruleSet: 1, delta: 0.125 } },
{ time: 120, obj: { ruleSet: 8, delta: -0.130 } },
{ time: 180, obj: { ruleSet: -8 } },
];

const CAMERA_EVENTS = [
{ time: 0, obj: [ 42, 1.5708, 0.0000 ] },
{ time: 40, obj: [ 320, -3.7292, 0.7250 ] },
{ time: 60, obj: [ 240, -4.4042, -0.7000 ] },
{ time: 80, obj: [ 200, -5.7792, 0.8000 ] },
{ time: 120, obj: [ 170, -2.7960, -0.7000 ] },
{ time: 180, obj: [ 220, -1.3600, 0.5000] },
];

const AUDIO_SHADER = `
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
  let hihat = atan2(out, 1.0-dist);
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
`;

const VISUAL_SHADER = `
struct Uniforms
{
  radius: f32,
  phi: f32,
  theta: f32,
  time: f32,
  //ruleSet: f32
}

struct Grid
{
  mul: vec3i,
  arr: array<u32>
}

struct RuleSet
{
  states: u32,
  arr: array<u32>
}

struct Hit
{
  index: i32,
  pos: vec3f,
  norm: vec3f,
  state: u32,
  dist: f32,
  maxDist: f32,
  col: vec3f
}

const WIDTH = 1024;
const HEIGHT = WIDTH / 1.6;
const EPSILON = 0.001;

// TODO Edit! For now these values just align the brightness due to changing state count.
const ruleSetTints = array<vec3f, 11>(
  vec3f(1.0), // clouds-5
  vec3f(1.0), // 44-5
  vec3f(1.0), // amoeba-5
  vec3f(0.25), // pyro-10
  vec3f(1.0), // framework-5
  vec3f(0.25), // spiky-10
  vec3f(0.25), // builder-10
  vec3f(0.25), // ripple-10
  vec3f(0.5), // shells-7
  vec3f(0.25), // pulse-10
  vec3f(1.0)); // more-builds-5

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage> grid: Grid;
@group(0) @binding(2) var<storage, read_write> outputGrid: Grid;
@group(0) @binding(3) var<storage> rules: RuleSet;

fn getCell(x: i32, y: i32, z: i32) -> u32
{
  // Consider only states 0 and 1. Cells in refactory period do NOT count as active neighbours, i.e. are counted as 0.
  return u32(1 - min(abs(1 - i32(grid.arr[grid.mul.z * z + grid.mul.y * y + x])), 1));
}

fn getMooreNeighbourCountWrap(pos: vec3i) -> u32
{
  let res = grid.mul.y;
  let dec = vec3i((pos.x - 1) % res, (pos.y - 1) % res, (pos.z - 1) % res);
  let inc = vec3i((pos.x + 1) % res, (pos.y + 1) % res, (pos.z + 1) % res);

  return  getCell(pos.x, inc.y, pos.z) +
          getCell(inc.x, inc.y, pos.z) +
          getCell(dec.x, inc.y, pos.z) +
          getCell(pos.x, inc.y, inc.z) +
          getCell(pos.x, inc.y, dec.z) +
          getCell(inc.x, inc.y, inc.z) +
          getCell(inc.x, inc.y, dec.z) +
          getCell(dec.x, inc.y, inc.z) +
          getCell(dec.x, inc.y, dec.z) +
          getCell(inc.x, pos.y, pos.z) +
          getCell(dec.x, pos.y, pos.z) +
          getCell(pos.x, pos.y, inc.z) +
          getCell(pos.x, pos.y, dec.z) +
          getCell(inc.x, pos.y, inc.z) +
          getCell(inc.x, pos.y, dec.z) +
          getCell(dec.x, pos.y, inc.z) +
          getCell(dec.x, pos.y, dec.z) +
          getCell(pos.x, dec.y, pos.z) +
          getCell(inc.x, dec.y, pos.z) +
          getCell(dec.x, dec.y, pos.z) +
          getCell(pos.x, dec.y, inc.z) +
          getCell(pos.x, dec.y, dec.z) +
          getCell(inc.x, dec.y, inc.z) +
          getCell(inc.x, dec.y, dec.z) +
          getCell(dec.x, dec.y, inc.z) +
          getCell(dec.x, dec.y, dec.z);
}

fn evalState(pos: vec3i, states: u32)
{
  let index = dot(pos, grid.mul);
  let value = grid.arr[index];

  switch(value) {
    case 0: {
      outputGrid.arr[index] = rules.arr[27 + getMooreNeighbourCountWrap(pos)];
    }
    case 1: {
      // Dying state 1 goes to 2 (or dies directly by being moduloed to 0, in case there are only 2 states)
      outputGrid.arr[index] = (1 + 1 - rules.arr[getMooreNeighbourCountWrap(pos)]) % rules.states;
    }
    default {
      // Refactory period
      outputGrid.arr[index] = min(value + 1, rules.states) % rules.states; 
    }
  }
}

@compute @workgroup_size(4,4,4)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u)
{
  evalState(vec3i(globalId), rules.states);
}

fn minComp(v: vec3f) -> f32
{
  return min(v.x, min(v.y, v.z));
}

fn maxComp(v: vec3f) -> f32
{
  return max(v.x, max(v.y, v.z));
}

fn intersectGround(d: f32, ori: vec3f, dir: vec3f, t: ptr<function, f32>) -> bool
{
  *t = step(EPSILON, abs(dir.y)) * ((d - ori.y) / dir.y);
  return bool(step(EPSILON, *t));
}

fn intersectAabb(minExt: vec3f, maxExt: vec3f, ori: vec3f, invDir: vec3f, tmin: ptr<function, f32>, tmax: ptr<function, f32>) -> bool
{
  let t0 = (minExt - ori) * invDir;
  let t1 = (maxExt - ori) * invDir;
  
  *tmin = maxComp(min(t0, t1));
  *tmax = minComp(max(t0, t1));

  return *tmin <= *tmax && *tmax > 0.0;
}

fn traverseGrid(ori: vec3f, invDir: vec3f, tmax: f32, hit: ptr<function, Hit>) -> bool
{
  let mulf = vec3f(grid.mul);
  var stepDir = sign(invDir);
  var t = (vec3f(0.5) + 0.5 * stepDir - fract(ori)) * invDir;
  var mask = vec3f(0);

  (*hit).dist = 0.0;
  (*hit).index = i32(dot(mulf, floor(vec3f(mulf.y * 0.5) + ori)));
  
  loop {
    (*hit).state = grid.arr[(*hit).index];
    if((*hit).state > 0) {
      (*hit).norm = mask * -stepDir;
      return true;
    }

    (*hit).dist = minComp(t);
    if((*hit).dist >= tmax) {
      return false;
    }
 
    mask.x = f32(t.x <= t.y && t.x <= t.z);
    mask.y = f32(t.y <= t.x && t.y <= t.z);
    mask.z = f32(t.z <= t.x && t.z <= t.y);

    t += mask * stepDir * invDir;
    (*hit).index += i32(dot(mulf, mask * stepDir));
  }
}

fn calcOcclusion(pos: vec3f, index: i32, norm: vec3i) -> f32
{
  let above = index + dot(grid.mul, norm);
  let dir = abs(norm);
  let hori = dot(grid.mul, dir.yzx);
  let vert = dot(grid.mul, dir.zxy);

  let edgeCellStates = vec4f(
    f32(min(1, grid.arr[above + hori])),
    f32(min(1, grid.arr[above - hori])),
    f32(min(1, grid.arr[above + vert])),
    f32(min(1, grid.arr[above - vert])));

  let cornerCellStates = vec4f(
    f32(min(1, grid.arr[above + hori + vert])),
    f32(min(1, grid.arr[above - hori + vert])),
    f32(min(1, grid.arr[above + hori - vert])),
    f32(min(1, grid.arr[above - hori - vert])));

  let uvLocal = fract(pos);
  let uv = vec2f(dot(uvLocal, vec3f(dir.yzx)), dot(uvLocal, vec3f(dir.zxy)));
  let uvInv = vec2f(1) - uv;

  let edgeOcc = edgeCellStates * vec4f(uv.x, uvInv.x, uv.y, uvInv.y);
  let cornerOcc = cornerCellStates * vec4f(uv.x * uv.y, uvInv.x * uv.y, uv.x * uvInv.y, uvInv.x * uvInv.y) * (vec4f(1.0) - edgeCellStates.xzwy) * (vec4f(1.0) - edgeCellStates.zyxw);

  return 1.0 - (edgeOcc.x + edgeOcc.y + edgeOcc.z + edgeOcc.w + cornerOcc.x + cornerOcc.y + cornerOcc.z + cornerOcc.w) * 0.5;
}

fn shade(pos: vec3f, dir: vec3f, hit: ptr<function, Hit>) -> vec3f
{
  // Wireframe, better add AA
  /*let border = vec3f(0.5 - 0.05);
  let wire = (vec3f(1) - abs((*hit).norm)) * abs(fract(pos) - vec3f(0.5));

  if(any(vec3<bool>(step(border, wire)))) {
    return vec3f(0);
  }*/

  let val = f32(rules.states) / f32(min((*hit).state, rules.states));
  let sky = 0.4 + (*hit).norm.y * 0.6;
  let col = vec3f(0.005) + /*ruleSetTints[u32(uniforms.ruleSet)] * */(vec3f(0.5) + pos / f32(grid.mul.y)) * sky * sky * val * val * 0.3 * exp(-3.5 * (*hit).dist / (*hit).maxDist);
  let occ = calcOcclusion(pos, (*hit).index, vec3i((*hit).norm));

  return col * occ * occ * occ;
}

fn background(ori: vec3f, dir: vec3f) -> vec3f
{
  // TODO Make much better background
  let a = 0.5 + abs(dir.y) * 0.5;
  return 0.05 * vec3f(0.3, 0.3, 0.4) * vec3f(pow(a, 42.0)); 
}

fn trace(ori: vec3f, dir: vec3f, hit: ptr<function, Hit>) -> bool
{
  let invDir = 1.0 / dir;
  var tmin: f32;
  var tmax: f32;
  let halfGrid = vec3f(f32(grid.mul.y / 2));

  if(intersectAabb(-halfGrid, halfGrid, ori, invDir, &tmin, &tmax)) {
    tmin = max(tmin + EPSILON, 0.0);
    (*hit).maxDist = tmax - EPSILON - tmin;
    if(traverseGrid(ori + tmin * dir, invDir, (*hit).maxDist, hit)) {
      (*hit).pos = ori + (tmin + (*hit).dist) * dir;
      (*hit).col = shade((*hit).pos, dir, hit);
      return true;
    }
  }

  (*hit).col = background(ori, dir);

  return false;
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn filmicToneACES(x: vec3f) -> vec3f
{
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return saturate(x * (a * x + vec3f(b)) / (x * (c * x + vec3f(d)) + vec3f(e)));
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f
{
  let pos = array<vec2f, 4>(vec2f(-1, 1), vec2f(-1, -1), vec2f(1), vec2f(1, -1));
  return vec4f(pos[vertexIndex], 0, 1);
}

@fragment
fn fragmentMain(@builtin(position) position: vec4f) -> @location(0) vec4f
{
  // Camera
  let FOV = 50.0;
  let tanHalfFov = 0.5 / tan(0.5 * radians(FOV));
  
  let dirEyeSpace = normalize(vec3f((position.xy - vec2f(WIDTH, HEIGHT) * 0.5) / f32(HEIGHT), tanHalfFov));

  let ori = vec3f(uniforms.radius * cos(uniforms.theta) * cos(uniforms.phi), uniforms.radius * sin(uniforms.theta), uniforms.radius * cos(uniforms.theta) * sin(uniforms.phi));

  let fwd = normalize(-ori);
  let ri = normalize(cross(fwd, vec3f(0, 1, 0)));
  let up = cross(ri, fwd);

  var dir = ri * dirEyeSpace.x - up * dirEyeSpace.y + fwd * dirEyeSpace.z;

  var col = vec3f(0.0);
  var hit: Hit;

  // Normal cell grid
  trace(ori, dir, &hit);
  col = hit.col;
 
  /*
  // Normal cell grid with ground reflection
  if(!trace(ori, dir, &hit)) {
    var t: f32;
    if(intersectGround(-30.0, ori, dir, &t)) {
      let newDir = reflect(dir, vec3f(0.0, 1.0, 0.0));
      var newOri = ori + t * dir;
      newOri += newDir * vec3f(5.0 * sin(newOri.x + uniforms.time * 0.0015), 0.0, 1.5 * sin(newOri.y + uniforms.time * 0.003));
      trace(newOri, newDir, &hit);
    }
  }
  col = hit.col;
  */

  /*
  // All cells reflecting
  var iter = 1.0;
  loop {
    let cellHit = trace(ori, dir, &hit);
    col += hit.col;
    if(!cellHit || iter > 2.0) {
      break;
    } 
    dir = reflect(dir, hit.norm);
    ori = hit.pos + EPSILON * dir;
    iter += 1.0;
  }
  col /= iter;
  */ 

  // Amplify
  //col += pow(max(col - vec3f(0.3), vec3f(0.0)), vec3f(1.5)) * 0.5;
  
  col = filmicToneACES(col);

  return vec4f(pow(col, vec3f(0.4545)), 1.0);
}
`;

// https://github.com/bryc/code/blob/master/jshash/PRNGs.md
function splitmix32(a) {
  return function() {
    a |= 0; a = a + 0x9e3779b9 | 0;
    var t = a ^ a >>> 16; t = Math.imul(t, 0x21f0aaad);
    t = t ^ t >>> 15; t = Math.imul(t, 0x735a2d97);
    return ((t = t ^ t >>> 15) >>> 0) / 4294967296;
  }
}

async function createComputePipeline(shaderModule, pipelineLayout, entryPoint)
{
  return device.createComputePipelineAsync({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: entryPoint
    }
  });
}

async function createRenderPipeline(shaderModule, pipelineLayout, vertexEntryPoint, fragmentEntryPoint)
{
  return device.createRenderPipelineAsync({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: vertexEntryPoint
    },
    fragment: {
      module: shaderModule,
      entryPoint: fragmentEntryPoint,
      targets: [{format: "bgra8unorm"}]
    },
    primitive: {topology: "triangle-strip"}
  });
}

function encodeComputePassAndSubmit(commandEncoder, pipeline, bindGroup, count)
{
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(count, count, count);
  passEncoder.end();
}

function encodeRenderPassAndSubmit(commandEncoder, passDescriptor, pipeline, bindGroup)
{
  const passEncoder = commandEncoder.beginRenderPass(passDescriptor);
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.draw(4);
  passEncoder.end();
}

async function createAudioResources()
{
  audioContext = new AudioContext();
  webAudioBuffer = audioContext.createBuffer(2, BUFFER_DIM ** 3, audioContext.sampleRate);

  let audioBuffer = device.createBuffer({
    size: (BUFFER_DIM ** 3) * 2 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

  let audioUniformBuffer = device.createBuffer({
    size: 2 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});

  let audioBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}
    ]});

  let audioBindGroup = device.createBindGroup({
    layout: audioBindGroupLayout,
    entries: [
      {binding: 0, resource: {buffer: audioUniformBuffer}},
      {binding: 1, resource: {buffer: audioBuffer}}
    ]});

  let audioReadBuffer = device.createBuffer({
    size: (BUFFER_DIM ** 3) * 2 * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});

  let audioPipelineLayout = device.createPipelineLayout({bindGroupLayouts: [audioBindGroupLayout]});

  device.queue.writeBuffer(audioUniformBuffer, 0, new Uint32Array([BUFFER_DIM, audioContext.sampleRate]));

  let pipeline = await createComputePipeline(device.createShaderModule({code: AUDIO_SHADER}), audioPipelineLayout, "audioMain");

  let commandEncoder = device.createCommandEncoder();

  encodeComputePassAndSubmit(commandEncoder, pipeline, audioBindGroup, BUFFER_DIM / 4);

  commandEncoder.copyBufferToBuffer(audioBuffer, 0, audioReadBuffer, 0, (BUFFER_DIM ** 3) * 2 * 4);

  device.queue.submit([commandEncoder.finish()]);

  await audioReadBuffer.mapAsync(GPUMapMode.READ);
  let audioData = new Float32Array(audioReadBuffer.getMappedRange());

  let channel0 = webAudioBuffer.getChannelData(0);
  let channel1 = webAudioBuffer.getChannelData(1);

  for(let i=0; i<BUFFER_DIM ** 3; i++) {
    channel0[i] = audioData[(i << 1) + 0];
    channel1[i] = audioData[(i << 1) + 1];
  }

  audioReadBuffer.unmap();

  audioBufferSourceNode = audioContext.createBufferSource();
  audioBufferSourceNode.buffer = webAudioBuffer;
  audioBufferSourceNode.connect(audioContext.destination); 
}

async function createRenderResources()
{
  let bindGroupLayout = device.createBindGroupLayout({
    entries: [ 
      {binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
      {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
      {binding: 3, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
    ]
  });
 
  uniformBuffer = device.createBuffer({
    size: 4 * 4, // radius, phi, theta, time
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});

  for(let i=0; i<2; i++)
    gridBuffer[i] = device.createBuffer({
      size: grid.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  rulesBuffer = device.createBuffer({
    size: rules.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  for(let i=0; i<2; i++) {
    bindGroup[i] = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {binding: 0, resource: {buffer: uniformBuffer}},
        {binding: 1, resource: {buffer: gridBuffer[i]}},
        {binding: 2, resource: {buffer: gridBuffer[1 - i]}},
        {binding: 3, resource: {buffer: rulesBuffer}},
      ]
    });
  }

  pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});

  renderPassDescriptor = {
    colorAttachments: [{
      undefined, // view
      clearValue: {r: 1.0, g: 0.0, b: 0.0, a: 1.0},
      loadOp: "clear",
      storeOp: "store"
    }]
  };

  let shaderModule = device.createShaderModule({code: VISUAL_SHADER});
  computePipeline = await createComputePipeline(shaderModule, pipelineLayout, "computeMain");
  renderPipeline = await createRenderPipeline(shaderModule, pipelineLayout, "vertexMain", "fragmentMain");
}

let last;

function render(time)
{  
  if(last !== undefined) {
    let frameTime = (performance.now() - last);
    document.title = `${(frameTime).toFixed(2)} / ${(1000.0 / frameTime).toFixed(2)}`;
  }
  last = performance.now();

  if(startTime === undefined) {
    audioBufferSourceNode.start(0, 0);
    startTime = audioContext.currentTime;
  }

  timeInBeats = (audioContext.currentTime - startTime) * BPM / 60;

  const commandEncoder = device.createCommandEncoder();
  
  updateSimulation();
  updateCamera();
 
  if(activeRuleSet > 0) {
    if(timeInBeats - lastSimulationUpdateTime > updateDelay) {
      encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup[simulationIteration % 2], BUFFER_DIM / 4); 
      simulationIteration++;
      lastSimulationUpdateTime = (audioContext.currentTime - startTime) * BPM / 60;
    }
  } else
    lastSimulationUpdateTime = timeInBeats;

  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    ...view, timeInBeats //, Math.abs(activeRuleSet) - 1
  ]));

  renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
  encodeRenderPassAndSubmit(commandEncoder, renderPassDescriptor, renderPipeline, bindGroup[simulationIteration % 2]);
  
  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);
}

function updateSimulation()
{
  if(activeSimulationEventIndex + 1 < SIMULATION_EVENTS.length && timeInBeats >= SIMULATION_EVENTS[activeSimulationEventIndex + 1].time) {
    let o = SIMULATION_EVENTS[++activeSimulationEventIndex].obj;

    // Rules
    if(o.ruleSet !== undefined) {
      activeRuleSet = o.ruleSet; // Can be active (positive) or inactive (negative), zero is excluded by definition
      let rulesBitsBigInt = RULES[Math.abs(activeRuleSet)];
      // State count (bit 0-3)
      rules[0] = Number(rulesBitsBigInt & BigInt(0xf));
      // Alive bits (4-31), birth bits (32-59)
      for(let i=0; i<rules.length - 1; i++)
        rules[1 + i] = Number((rulesBitsBigInt >> BigInt(4 + i)) & BigInt(0x1));
      device.queue.writeBuffer(rulesBuffer, 0, rules);
    }

    // Time
    updateDelay = (o.delta === undefined) ? updateDelay : updateDelay + o.delta;
  }
}

function updateCamera()
{
  if(activeCameraEventIndex + 1 < CAMERA_EVENTS.length && timeInBeats >= CAMERA_EVENTS[activeCameraEventIndex + 1].time)
    ++activeCameraEventIndex;

  if(activeCameraEventIndex >= 0 && activeCameraEventIndex + 1 < CAMERA_EVENTS.length) {
    let curr = CAMERA_EVENTS[activeCameraEventIndex];
    let next = CAMERA_EVENTS[activeCameraEventIndex + 1];
    let t = (timeInBeats - curr.time) / (next.time - curr.time);
    for(let i=0; i<3; i++)
      view[i] = curr.obj[i] + (next.obj[i] - curr.obj[i]) * t;
  }
}

function setGrid(area)
{
  for(let i=0; i<grid.length; i++)
    grid[i] = 0;

  grid[0] = 1;
  grid[1] = BUFFER_DIM;
  grid[2] = BUFFER_DIM ** 2;

  const center = BUFFER_DIM * 0.5;
  const d = area * 0.5;

  let rand = splitmix32(4079287172);

  // TODO Make initial grid somewhat more interesting
  for(let k=center - d; k<center + d; k++)
    for(let j=center - d; j<center + d; j++)
      for(let i=center - d; i<center + d; i++)
        grid[3 + (BUFFER_DIM ** 2) * k + BUFFER_DIM * j + i] = rand() > 0.6 ? 1 : 0;

  device.queue.writeBuffer(gridBuffer[0], 0, grid);
  device.queue.writeBuffer(gridBuffer[1], 0, grid);
}

function startRender()
{
  document.querySelector("button").removeEventListener("click", startRender);

  if(FULLSCREEN)
    canvas.requestFullscreen();
  else {
    canvas.style.width = CANVAS_WIDTH;
    canvas.style.height = CANVAS_HEIGHT;
    canvas.style.position = "absolute";
    canvas.style.left = 0;
    canvas.style.top = 0;
  }

  updateSimulation();
  updateCamera();
  requestAnimationFrame(render);
}

async function main()
{
  if(!navigator.gpu)
    alert("No WebGPU");

  const gpuAdapter = await navigator.gpu.requestAdapter();
  device = await gpuAdapter.requestDevice();

  await createAudioResources();
  await createRenderResources();
  setGrid(24);

  document.body.innerHTML = "<button>CLICK<canvas style='width:0;cursor:none'>";
  canvas = document.querySelector("canvas");
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  context = canvas.getContext("webgpu");
  context.configure({device, format: "bgra8unorm", alphaMode: "opaque"});

  document.querySelector("button").addEventListener("click", startRender);
}

main();
