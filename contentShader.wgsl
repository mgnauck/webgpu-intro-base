struct Uniforms
{
  right: vec3f,
  tanHalfFov: f32,
  up: vec3f,
  time: f32,
  forward: vec3f,
  freeValue1: f32,
  eye: vec3f,
  freeValue2: f32
}

struct Grid
{
  mul: vec3i,
  pad1: i32,
  minc: vec3u,
  pad2: u32,
  maxc: vec3u,
  arr: array<u32>
}

struct OutputGrid
{
  mul: vec3i,
  pad1: i32,
  minx: atomic<u32>,
  miny: atomic<u32>,
  minz: atomic<u32>,
  pad2: u32,
  maxx: atomic<u32>,
  maxy: atomic<u32>,
  maxz: atomic<u32>,
  arr: array<u32>
}

struct Rules
{
  kind: u32,
  states: u32,
  arr: array<u32>
}

struct Result
{
  index: i32,
  norm: vec3f,
  state: u32,
  dist: f32,
  distNorm: f32
}

const WIDTH = 1024;
const HEIGHT = WIDTH / 1.6;
const EPSILON = 0.001;
const PI = 3.141;
const TWO_PI = PI * 2.0;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage> grid: Grid;
@group(0) @binding(2) var<storage, read_write> outputGrid: OutputGrid;
@group(0) @binding(3) var<storage> rules: Rules;

fn maxComp(v: vec3f) -> f32
{
  return max(v.x, max(v.y, v.z));
}

fn minComp(v: vec3f) -> f32
{
  return min(v.x, min(v.y, v.z));
}

fn getCell(x: i32, y: i32, z: i32) -> u32
{
  // Consider only states 0 and 1. Cells in refactory period do NOT count as active neighbours, i.e. are counted as 0.
  return 1 - min(abs(1 - grid.arr[grid.mul.z * z + grid.mul.y * y + x]), 1);
}

fn getNeumannNeighbourCountWrap(pos: vec3i) -> u32
{
  let res = grid.mul.y;
  let dec = vec3i((pos.x - 1) % res, (pos.y - 1) % res, (pos.z - 1) % res);
  let inc = vec3i((pos.x + 1) % res, (pos.y + 1) % res, (pos.z + 1) % res);

  return  getCell(inc.x, pos.y, pos.z) +
          getCell(dec.x, pos.y, pos.z) +
          getCell(pos.x, inc.y, pos.z) +
          getCell(pos.x, dec.y, pos.z) +
          getCell(pos.x, pos.y, inc.z) +
          getCell(pos.x, pos.y, dec.z);
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

fn evalMultiState(pos: vec3i, states: u32)
{
  let index = dot(pos, grid.mul);
  let value = grid.arr[index];

  switch(value) {
    case 0: {
      let newValue = rules.arr[getMooreNeighbourCountWrap(pos)];
      outputGrid.arr[index] = newValue;
      if(newValue == 1) {
        let s = grid.mul.y - 1;
        atomicMin(&outputGrid.minx, u32(max(0, pos.x - 1)));
        atomicMin(&outputGrid.miny, u32(max(0, pos.y - 1)));
        atomicMin(&outputGrid.minz, u32(max(0, pos.z - 1)));
        atomicMax(&outputGrid.maxx, u32(min(pos.x + 1, s)));
        atomicMax(&outputGrid.maxy, u32(min(pos.y + 1, s)));
        atomicMax(&outputGrid.maxz, u32(min(pos.z + 1, s)));
      }
    }
    case 1: {
      // Dying state 1 goes to 2
      outputGrid.arr[index] = 1 + 1 - rules.arr[27 + getMooreNeighbourCountWrap(pos)];
    }
    default {
      // Refactory period
      outputGrid.arr[index] = (value + 1) % rules.states;
    }
  }
}

@compute @workgroup_size(4,4,4)
fn c(@builtin(global_invocation_id) globalId: vec3u)
{
  let halfExtent = vec3f(grid.maxc - grid.minc) * 0.5;
  if(maxComp(abs(vec3f(globalId - grid.minc) - halfExtent) / halfExtent) <= 1.0) {
    switch(rules.kind) {
      case 1: {
        // TODO handle different automaton
      }
      default: {
        evalMultiState(vec3i(globalId), rules.states);
      }
    }
  }
}

fn intersectAabb(minExt: vec3f, maxExt: vec3f, ori: vec3f, invDir: vec3f, tmin: ptr<function, f32>, tmax: ptr<function, f32>) -> bool
{
  let t0 = (minExt - ori) * invDir;
  let t1 = (maxExt - ori) * invDir;
  
  *tmin = maxComp(min(t0, t1));
  *tmax = minComp(max(t0, t1));

  return *tmin <= *tmax && *tmax > 0.0;
}

fn traverseGrid(ori: vec3f, invDir: vec3f, tmax: f32, res: ptr<function, Result>) -> bool
{
  let gridMul = vec3f(grid.mul);
  let stepDir = sign(invDir);
  var t = (vec3f(0.5) + 0.5 * stepDir - fract(ori)) * invDir;
  var mask: vec3f;

  (*res).index = i32(dot(gridMul, floor(ori)));
  
  loop {
    (*res).dist = minComp(t);
    if((*res).dist >= tmax) {
      return false;
    }
 
    mask.x = f32(t.x <= t.y && t.x <= t.z);
    mask.y = f32(t.y <= t.x && t.y <= t.z);
    mask.z = f32(t.z <= t.x && t.z <= t.y);

    t += mask * stepDir * invDir;
    (*res).index += i32(dot(gridMul, mask * stepDir));
 
    (*res).state = grid.arr[(*res).index];
    if((*res).state > 0) {
      (*res).norm = mask * -stepDir;
      return true;
    }
  }
}

// iq/rgba
fn palette(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f
{
  return a + b * cos(TWO_PI * (c * t + d));
}

fn shade(pos: vec3f, dir: vec3f, res: ptr<function, Result>) -> vec3f
{
  /*
  // Wireframe, better add AA
  let border = vec3f(0.5 - 0.02);
  let wire = (vec3f(1) - abs((*res).norm)) * abs(fract(pos) - vec3f(0.5));

  if(any(vec3<bool>(step(border, wire)))) {
    return vec3f(0);
  }
  */

  let val = f32(rules.states) / f32((*res).state);  

  // Position in cube and z-distance with sky and state
  let sky = (0.4 + (*res).norm.y * 0.6);
  let col = pos / f32(grid.mul.y) * val * val * 0.33 * sky * exp(-4 * (*res).distNorm);

  /*
  // Distance from center into palette scaled by state and dist
  let halfGrid = f32(grid.mul.y) * 0.5;
  let v = pos - vec3f(halfGrid);
  let centerDist = length(v) / halfGrid;
  let col = palette(centerDist, vec3f(0.3), vec3f(0.2), vec3f(0.6, 0.3, 0.3), vec3f(0.3, 0.5, 0.3)) * val * val * 0.5 * exp(-3.5 * (*res).distNorm);
  */

  let occ = calcOcclusion(pos, (*res).index, vec3i((*res).norm));
  return col * occ * occ * occ;
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

@vertex
fn v(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f
{
  let pos = array<vec2f, 4>(vec2f(-1, 1), vec2f(-1, -1), vec2f(1), vec2f(1, -1));
  return vec4f(pos[vertexIndex], 0, 1);
}

@fragment
fn f(@builtin(position) position: vec4f) -> @location(0) vec4f
{
  // Framebuffer y-down in webgpu
  let origin = uniforms.eye;
  let dirEyeSpace = normalize(vec3f((position.xy - vec2f(WIDTH, HEIGHT) * 0.5) / f32(HEIGHT), uniforms.tanHalfFov));
  let dir = uniforms.right * dirEyeSpace.x - uniforms.up * dirEyeSpace.y + uniforms.forward * dirEyeSpace.z;

  var col = vec3f(0.3, 0.3, 0.6) * 0.005;
  let invDir = 1.0 / dir;
  var tmin: f32;
  var tmax: f32;
  var res: Result;

  if(intersectAabb(vec3f(grid.minc), vec3f(grid.maxc), origin, invDir, &tmin, &tmax)) {
    let tminGrid = max(tmin - EPSILON, 0.0);
    let tmaxGrid = tmax - EPSILON - tminGrid;
    if(traverseGrid(origin + tminGrid * dir, invDir, tmaxGrid, &res)) {
      res.distNorm = res.dist / tmaxGrid;
      col = shade(origin + (tminGrid + res.dist) * dir, dir, &res);
    } /*else
    {
      // Visualize grid box
      col = vec3f(0.005);
    }*/
  }

  // Amplify
  col += pow(max(col - vec3f(0.3), vec3f(0.0)), vec3f(1.5)) * 0.5;

  return vec4f(pow(col, vec3f(0.4545)), 1.0);
}
