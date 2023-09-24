const FULLSCREEN = false;
const BPM = 120;

const ASPECT = 1.6;
const CANVAS_WIDTH = 1024; // Careful, this is also hardcoded in the shader!!
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;
const FOV = 50.0;

const AUDIO_BUFFER_SIZE = 4096 * 4096;

const MAX_GRID_RES = 256;

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

let radius, phi, theta;

let rand;
let gridRes;
let updateDelay = 0.5;
let grid;
let rules;

let startTime;
let timeInBeats = 0;
let lastSimulationUpdateTime = 0;
let simulationIteration = 0;
let gridBufferUpdateOffset = 0;
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
{ time: 0, obj: { ruleSet: 3, delta: -0.320, seed: 4079287172, gridRes: MAX_GRID_RES, area: 24 } },
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

  for(var i=0; i<ROWS; i++)
  {
    let beatTime = f32(i) * TIME_PER_BEAT;
    let noteTime = modulo_euclidean(time - beatTime, TIME_PER_PATTERN);
    let currentRow = PATTERN[i];

    // kick
    let kickNote = currentRow[KICK_CHANNEL];
    //let kickNoteNext = PATTERN[(i+1)%ROWS][KICK_CHANNEL];
    let kickNoteFreq = noteToFreq(kickNote);
    let kickNoteOn = 1.0 * sign(kickNote + 1.0);

    if(time > TIME_PER_PATTERN * 1)
    {
      output += vec2f(0.30 * kick(noteTime, kickNoteFreq) * kickNoteOn);
    }

    // hihat
    let hihatNote = currentRow[HIHAT_CHANNEL];
    let hihatNoteFreq = noteToFreq(hihatNote);
    let hihatNoteOn = sign(hihatNote + 1.0);
    output += vec2f(0.05 * hihat(noteTime, hihatNoteFreq) * hihatNoteOn);

    // bass
    let bassNote = currentRow[BASS_CHANNEL];
    let bassNoteFreq = noteToFreq(bassNote);
    let bassNoteOn = sign(bassNote + 1.0);
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
`;

const VISUAL_SHADER = `
struct Uniforms
{
  right: vec3f,
  tanHalfFov: f32,
  up: vec3f,
  time: f32,
  forward: vec3f,
  ruleSet: f32,
  eye: vec3f,
  freeValue2: f32
}

struct Grid
{
  mul: vec3i,
  zOfs: u32,
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
  var pos = vec3i(i32(globalId.x), i32(globalId.y), i32(globalId.z + grid.zOfs));
  if(pos.z >= grid.mul.y) {
    return;
  }

  evalState(pos, rules.states);
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
  let col = vec3f(0.005) + ruleSetTints[u32(uniforms.ruleSet)] * (vec3f(0.5) + pos / f32(grid.mul.y)) * sky * sky * val * val * 0.3 * exp(-3.5 * (*hit).dist / (*hit).maxDist);
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
  // Framebuffer y-down in webgpu
  let dirEyeSpace = normalize(vec3f((position.xy - vec2f(WIDTH, HEIGHT) * 0.5) / f32(HEIGHT), uniforms.tanHalfFov));
  var dir = uniforms.right * dirEyeSpace.x - uniforms.up * dirEyeSpace.y + uniforms.forward * dirEyeSpace.z;
  var ori = uniforms.eye;

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

function vec3Add(a, b)
{
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vec3Negate(v)
{
  return [-v[0], -v[1], -v[2]];
}

function vec3Scale(v, s)
{
  return [v[0] * s, v[1] * s, v[2] * s];
}

function vec3Cross(a, b)
{
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function vec3Normalize(v)
{
  let invLen = 1.0 / Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  return [v[0] * invLen, v[1] * invLen, v[2] * invLen];
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

function encodeComputePassAndSubmit(commandEncoder, pipeline, bindGroup, countX, countY, countZ)
{
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(countX, countY, countZ);
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
  webAudioBuffer = audioContext.createBuffer(2, AUDIO_BUFFER_SIZE, audioContext.sampleRate);

  let audioBuffer = device.createBuffer({
    size: AUDIO_BUFFER_SIZE * 2 * 4, 
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
    size: AUDIO_BUFFER_SIZE * 2 * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});

  let audioPipelineLayout = device.createPipelineLayout({bindGroupLayouts: [audioBindGroupLayout]});

  device.queue.writeBuffer(audioUniformBuffer, 0, new Uint32Array([Math.ceil(Math.cbrt(AUDIO_BUFFER_SIZE)), audioContext.sampleRate]));

  let pipeline = await createComputePipeline(device.createShaderModule({code: AUDIO_SHADER}), audioPipelineLayout, "audioMain");

  let commandEncoder = device.createCommandEncoder();

  let count = Math.ceil(Math.cbrt(AUDIO_BUFFER_SIZE) / 4);
  encodeComputePassAndSubmit(commandEncoder, pipeline, audioBindGroup, count, count, count);

  commandEncoder.copyBufferToBuffer(audioBuffer, 0, audioReadBuffer, 0, AUDIO_BUFFER_SIZE * 2 * 4);

  device.queue.submit([commandEncoder.finish()]);

  await audioReadBuffer.mapAsync(GPUMapMode.READ);
  let audioData = new Float32Array(audioReadBuffer.getMappedRange());

  let channel0 = webAudioBuffer.getChannelData(0);
  let channel1 = webAudioBuffer.getChannelData(1);

  for(let i=0; i<AUDIO_BUFFER_SIZE; i++) {
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
 
  // Right, up, dir, eye, fov, time, simulation step, programmable value/padding
  uniformBuffer = device.createBuffer({
    size: 16 * 4, 
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
  //setPerformanceTimer();
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
 
  if(!idle && activeRuleSet > 0) {
    if(timeInBeats - lastSimulationUpdateTime > updateDelay) {
      const count = Math.ceil(gridRes / 4);
      encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup[simulationIteration % 2], count, count, count); 
      simulationIteration++;
      lastSimulationUpdateTime = ((AUDIO ? audioContext.currentTime : time / 1000.0) - startTime) * BPM / 60;
    }
  } else {
    lastSimulationUpdateTime = timeInBeats;
  }

  let view = calcView();
  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    ...view.right,
    0.5 / Math.tan(0.5 * FOV * Math.PI / 180.0),
    ...view.up,
    timeInBeats,
    ...view.dir,
    Math.abs(activeRuleSet) - 1,
    ...view.eye,
    1.0 // free value
  ]));

  renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
  encodeRenderPassAndSubmit(commandEncoder, renderPassDescriptor, renderPipeline, bindGroup[simulationIteration % 2]);
  
  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);
}

function setPerformanceTimer(timerName)
{
  let begin = performance.now();
  device.queue.onSubmittedWorkDone()
    .then(function() {
      let end = performance.now();
      let frameTime = end - begin;
      document.title = `${(frameTime).toFixed(2)} / ${(1000.0 / frameTime).toFixed(2)}`;
    }).catch(function(err) {
      console.log(err);
    });
}

function updateSimulation()
{
  if(activeSimulationEventIndex + 1 < SIMULATION_EVENTS.length && timeInBeats >= SIMULATION_EVENTS[activeSimulationEventIndex + 1].time) {
    let eventObj = SIMULATION_EVENTS[++activeSimulationEventIndex].obj;
    setGrid(eventObj);
    setRules(eventObj);
    setTime(eventObj);
  }
}

function updateCamera()
{
  if(activeCameraEventIndex + 1 < CAMERA_EVENTS.length && timeInBeats >= CAMERA_EVENTS[activeCameraEventIndex + 1].time)
    ++activeCameraEventIndex;

  if(activeCameraEventIndex >= 0) {
    let curr = CAMERA_EVENTS[activeCameraEventIndex];
    let vals = curr.obj;
    if(activeCameraEventIndex + 1 < CAMERA_EVENTS.length) {
      let next = CAMERA_EVENTS[activeCameraEventIndex + 1];
      vals = vec3Add(vals, vec3Scale(vec3Add(next.obj, vec3Negate(vals)), (timeInBeats - curr.time) / (next.time - curr.time)));
    }
    radius = vals[0];
    phi = vals[1];
    theta = vals[2];
    // TODO Apply unsteady cam again
  }
}

function calcView()
{
  // TODO If we calc this in the shader (on every fragment), then we can throw out all vector math in the JS
  let e = [radius * Math.cos(theta) * Math.cos(phi), radius * Math.sin(theta), radius * Math.cos(theta) * Math.sin(phi)];
  let d = vec3Normalize(vec3Negate(e));
  let r = vec3Normalize(vec3Cross(d, [0, 1, 0]));  
  return { eye: e, dir: d, right: r, up: vec3Cross(r, d) };
}

// TODO Can go!
function resetView()
{
  radius = gridRes * 0.5;
  phi = Math.PI * 0.5;
  theta = 0;
}

// TODO This will likely be thrown out completely (we just need to initially create the grid)
function setGrid(obj)
{
  if(obj.gridRes === undefined)
    return;

  for(let i=0; i<grid.length; i++)
    grid[i] = 0;

  rand = splitmix32(obj.seed);
  gridRes = obj.gridRes;

  grid[0] = 1;
  grid[1] = gridRes;
  grid[2] = gridRes * gridRes;

  const center = gridRes * 0.5; 
  const d = obj.area * 0.5;

  // TODO Make initial grid somewhat more interesting
  for(let k=center - d; k<center + d; k++)
    for(let j=center - d; j<center + d; j++)
      for(let i=center - d; i<center + d; i++)
        grid[3 + gridRes * gridRes * k + gridRes * j + i] = rand() > 0.6 ? 1 : 0;

  device.queue.writeBuffer(gridBuffer[0], 0, grid);
  device.queue.writeBuffer(gridBuffer[1], 0, grid);
}

// TODO Pull the next two functions into updateSimulation()
function setRules(obj)
{
  // TODO Check if bitwise encoding is in fact larger than simply keeping the bitfield as raw array
  if(obj.ruleSet !== undefined) {
    activeRuleSet = obj.ruleSet; // Can be active (positive) or inactive (negative), zero is excluded by definition
    let rulesBitsBigInt = RULES[Math.abs(activeRuleSet)];
    // State count (bit 0-3)
    rules[0] = Number(rulesBitsBigInt & BigInt(0xf));
    // Alive bits (4-31), birth bits (32-59)
    for(let i=0; i<rules.length - 1; i++)
      rules[1 + i] = Number((rulesBitsBigInt >> BigInt(4 + i)) & BigInt(0x1));
    device.queue.writeBuffer(rulesBuffer, 0, rules);
  }
}

function setTime(obj)
{
  if(obj.delta !== undefined)
    updateDelay += obj.delta;
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
  resetView();
  updateCamera();

  requestAnimationFrame(render);
}

async function main()
{
  if(!navigator.gpu)
    throw new Error("WebGPU is not supported on this browser.");

  const gpuAdapter = await navigator.gpu.requestAdapter();
  if(!gpuAdapter)
    throw new Error("Can not use WebGPU. No GPU adapter available.");

  device = await gpuAdapter.requestDevice();
  if(!device)
    throw new Error("Failed to request logical device.");

  await createAudioResources();

  grid = new Uint32Array(3 + MAX_GRID_RES * MAX_GRID_RES * MAX_GRID_RES);
  rules = new Uint32Array(1 + 2 * 27);

  await createRenderResources();

  document.body.innerHTML = "<button>CLICK<canvas style='width:0;cursor:none'>";
  canvas = document.querySelector("canvas");
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  let presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  if(presentationFormat !== "bgra8unorm")
    throw new Error(`Expected canvas pixel format of bgra8unorm but was '${presentationFormat}'.`);

  context = canvas.getContext("webgpu");
  context.configure({device, format: presentationFormat, alphaMode: "opaque"});

  document.querySelector("button").addEventListener("click", startRender);
}

main();
