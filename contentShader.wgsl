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
  minx: u32,
  miny: u32,
  minz: u32,
  maxx: u32,
  maxy: u32,
  maxz: u32,
  arr: array<u32>
}

struct OutputGrid
{
  mul: vec3i,
  // TODO Add proper padding so we can use vec3f in Grid struct
  minx: atomic<u32>,
  miny: atomic<u32>,
  minz: atomic<u32>,
  maxx: atomic<u32>,
  maxy: atomic<u32>,
  maxz: atomic<u32>,
  arr: array<u32>
}

// TODO Use as binding 3
struct Rules
{
  ilk: u32,
  arr: array<u32>
}

const WIDTH = 800;
const HEIGHT = WIDTH / 1.6;

const EPSILON = 0.001;
const HEMISPHERE = vec3f(0.3, 0.3, 0.6);

const rules = array<array<u32, 27>, 2>(
  array<u32, 27>(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), // birth
  array<u32, 27>(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  // live
);

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage> grid: Grid;
@group(0) @binding(2) var<storage, read_write> outputGrid: OutputGrid;

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
      let newValue = rules[0][getMooreNeighbourCountWrap(pos)];
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
      outputGrid.arr[index] = 1 + 1 - rules[1][getMooreNeighbourCountWrap(pos)];
    }
    default {
      // Refactory period
      outputGrid.arr[index] = (value + 1) % states;
    }
  }
}

@compute @workgroup_size(4,4,4)
fn c(@builtin(global_invocation_id) globalId: vec3u)
{
  let gridMin = vec3f(f32(grid.minx), f32(grid.miny), f32(grid.minz));
  let gridMax = vec3f(f32(grid.maxx), f32(grid.maxy), f32(grid.maxz));
  let halfExtent = (gridMax - gridMin) * 0.5;
  if(maxComp(abs(vec3f(globalId) - (gridMin + halfExtent)) / halfExtent) <= 1.0) {
    evalMultiState(vec3i(globalId), 5);
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

fn traverseGrid(ori: vec3f, invDir: vec3f, tmax: f32, dist: ptr<function, f32>, norm: ptr<function, vec3f>) -> u32
{
  let stepDir = sign(invDir);
  var cell = floor(ori);
  var t = (vec3f(0.5) + 0.5 * stepDir - fract(ori)) * invDir;
  
  var mask : vec3f;
  let gridMul = vec3f(grid.mul);

  while(*dist < tmax) {
    mask.x = f32(t.x <= t.y && t.x <= t.z);
    mask.y = f32(t.y <= t.x && t.y <= t.z);
    mask.z = f32(t.z <= t.x && t.z <= t.y);
 
    t += mask * stepDir * invDir;
    cell += mask * stepDir;

    *dist = dot(mask, (vec3f(0.5) - 0.5 * stepDir + cell - ori) * invDir);
    let state = grid.arr[u32(dot(gridMul, cell))];
    if(state > 0) {
      *norm = -mask * stepDir;
      return state;
    }
  }

  return 0;
}

fn shade(pos: vec3f, dir: vec3f, norm: vec3f, dist: f32, state: u32) -> vec3f
{
  let border = vec3f(0.5 - 0.05);
  let wire = (vec3f(1) - abs(norm)) * abs(fract(pos) - vec3f(0.5));

  if(any(vec3<bool>(step(border, wire)))) {
    return vec3f(0);
  }

  // TODO Fake AO
  // TODO Consider distance to center in shading

  let val = 5.0 / f32(state);
  let sky = (0.4 + norm.y * 0.6);
  return pos / f32(grid.mul.y) * sky * val * exp(4 * -dist);
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

  var col = HEMISPHERE * 0.005;

  let invDir = 1.0 / dir; 
  var tmin: f32;
  var tmax: f32;
  var t: f32;
  var norm: vec3f;

  if(intersectAabb(vec3f(f32(grid.minx), f32(grid.miny), f32(grid.minz)), vec3f(f32(grid.maxx), f32(grid.maxy), f32(grid.maxz)), origin, invDir, &tmin, &tmax)) {
    tmin = max(tmin - EPSILON, -EPSILON);
    let state = traverseGrid(origin + tmin * dir, invDir, tmax - tmin, &t, &norm);
    if(state > 0) {
      col = shade(origin + (tmin + t) * dir, dir, norm, (tmin + t) / tmax, state);
    } /*else
    {
      // Visualize grid box
      col = vec3f(0.1);
    }*/
  }

  return vec4f(pow(col, vec3f(0.4545)), 1.0);
}
