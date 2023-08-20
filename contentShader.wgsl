struct Uniforms
{
  right: vec3f,
  gridRes: f32,
  up: vec3f,
  verticalFovInDeg: f32,
  forward: vec3f,
  time: f32,
  eye: vec3f,
  freeValue: f32
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
@group(0) @binding(1) var<storage> grid : array<u32>;
@group(0) @binding(2) var<storage, read_write> outputGrid : array<u32>;

fn maxComp(v: vec3f) -> f32
{
  return max(v.x, max(v.y, v.z));
}

fn minComp(v: vec3f) -> f32
{
  return min(v.x, min(v.y, v.z));
}

fn getCell(x: i32, y: i32, z: i32, gridMul: ptr<function, vec3i>) -> u32
{
  // Consider only states 0 and 1. Cells in refactory period do NOT count as active neighbours.
  return 1 - min(abs(1 - grid[(*gridMul).z * z + (*gridMul).y * y + x]), 1);
}

fn getNeumannNeighbourCountWrap(pos: vec3i, gridMul: ptr<function, vec3i>) -> u32
{
  let res = vec3i((*gridMul).y);
  let dec = (pos - vec3i(1)) % res;
  let inc = (pos + vec3i(1)) % res;

  return  getCell(inc.x, pos.y, pos.z, gridMul) +
          getCell(dec.x, pos.y, pos.z, gridMul) +
          getCell(pos.x, inc.y, pos.z, gridMul) +
          getCell(pos.x, dec.y, pos.z, gridMul) +
          getCell(pos.x, pos.y, inc.z, gridMul) +
          getCell(pos.x, pos.y, dec.z, gridMul);
}

fn getMooreNeighbourCountWrap(pos: vec3i, gridMul: ptr<function, vec3i>) -> u32
{
  let res = vec3i((*gridMul).y);
  let dec = (pos - vec3i(1)) % res;
  let inc = (pos + vec3i(1)) % res;

  return  getCell(pos.x, inc.y, pos.z, gridMul) +
          getCell(inc.x, inc.y, pos.z, gridMul) +
          getCell(dec.x, inc.y, pos.z, gridMul) +
          getCell(pos.x, inc.y, inc.z, gridMul) +
          getCell(pos.x, inc.y, dec.z, gridMul) +
          getCell(inc.x, inc.y, inc.z, gridMul) +
          getCell(inc.x, inc.y, dec.z, gridMul) +
          getCell(dec.x, inc.y, inc.z, gridMul) +
          getCell(dec.x, inc.y, dec.z, gridMul) +
          getCell(inc.x, pos.y, pos.z, gridMul) +
          getCell(dec.x, pos.y, pos.z, gridMul) +
          getCell(pos.x, pos.y, inc.z, gridMul) +
          getCell(pos.x, pos.y, dec.z, gridMul) +
          getCell(inc.x, pos.y, inc.z, gridMul) +
          getCell(inc.x, pos.y, dec.z, gridMul) +
          getCell(dec.x, pos.y, inc.z, gridMul) +
          getCell(dec.x, pos.y, dec.z, gridMul) +
          getCell(pos.x, dec.y, pos.z, gridMul) +
          getCell(inc.x, dec.y, pos.z, gridMul) +
          getCell(dec.x, dec.y, pos.z, gridMul) +
          getCell(pos.x, dec.y, inc.z, gridMul) +
          getCell(pos.x, dec.y, dec.z, gridMul) +
          getCell(inc.x, dec.y, inc.z, gridMul) +
          getCell(inc.x, dec.y, dec.z, gridMul) +
          getCell(dec.x, dec.y, inc.z, gridMul) +
          getCell(dec.x, dec.y, dec.z, gridMul);
}

fn evalMultiState(pos: vec3i, states: u32, gridMul: ptr<function, vec3i>)
{
  let index = dot(pos, *gridMul);
  let value = grid[index];

  if(value <= 1) {
    let count = getMooreNeighbourCountWrap(pos, gridMul);
    outputGrid[index] = u32(abs(i32(value + value) - i32(rules[value][count])));
  } else { 
    outputGrid[index] = (value + 1) % states;
  }
}

@compute @workgroup_size(4,4,4)
fn c(@builtin(global_invocation_id) globalId: vec3u)
{
  let currPos = vec3i(globalId);
  let gridRes = i32(uniforms.gridRes);
  var gridMul = vec3i(1, gridRes, gridRes * gridRes);
 
  evalMultiState(currPos, 5, &gridMul);
}

fn intersectAabb(minExt: vec3f, maxExt: vec3f, ori: vec3f, invDir: vec3f, tmin: ptr<function, f32>, tmax: ptr<function, f32>) -> bool
{
  let t0 = (minExt - ori) * invDir;
  let t1 = (maxExt - ori) * invDir;
  
  *tmin = maxComp(min(t0, t1));
  *tmax = minComp(max(t0, t1));

  return *tmin <= *tmax && *tmax > 0.0;
}

fn traverseGrid(ori: vec3f, invDir: vec3f, tmax: f32, gridRes: f32, dist: ptr<function, f32>, norm: ptr<function, vec3f>) -> bool
{
  let gridMul = vec3f(1.0, gridRes, gridRes * gridRes);
  let stepDir = sign(invDir);
  var cell = floor(ori);
  var t = (vec3f(0.5) + 0.5 * stepDir - fract(ori)) * invDir;

  while(*dist < tmax) {
    let mask = vec3f(f32(t.x <= t.y && t.x <= t.z), f32(t.y <= t.x && t.y <= t.z), f32(t.z <= t.x && t.z <= t.y));
 
    t += mask * stepDir * invDir;
    cell += mask * stepDir;

    *dist = dot(mask, (vec3f(0.5) - 0.5 * stepDir + cell - ori) * invDir);

    if(grid[u32(dot(gridMul, cell))] > 0) {
      *norm = -mask * stepDir;
      return true;
    }
  }

  return false;
}

fn shade(pos: vec3f, dir: vec3f, norm: vec3f, dist: f32) -> vec3f
{
  let border = vec3f(0.5 - 0.05);
  let wire = (vec3f(1) - abs(norm)) * abs(fract(pos) - vec3f(0.5));

  if(any(vec3<bool>(step(border, wire)))) {
    return vec3f(0);
  }

  // TODO Fake AO
  // TODO Consider state in shading
  // TODO Consider distance to center in shading

  var sky = (0.4 + norm.y * 0.6);
  return pos / uniforms.gridRes * sky * exp(4 * -dist);
}

fn renderBackground(o: vec3f, d: vec3f) -> vec3f
{
  return HEMISPHERE * 0.001;
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
  let origin = uniforms.eye;
  let dirEyeSpace = vec3f((position.xy - vec2f(WIDTH, HEIGHT) * 0.5) / f32(HEIGHT), 0.5 / tan(radians(0.5 * uniforms.verticalFovInDeg)));
  let dir = uniforms.right * dirEyeSpace.x - uniforms.up * dirEyeSpace.y + uniforms.forward * dirEyeSpace.z;

  let invDir = 1.0 / dir; 
  var col = renderBackground(origin, dir);
  var tmin: f32;
  var tmax: f32;
  var t: f32;
  var norm: vec3f;

  if(intersectAabb(vec3f(0), vec3f(uniforms.gridRes), origin, invDir, &tmin, &tmax)) {
    tmin = max(tmin - EPSILON, -EPSILON);
    if(traverseGrid(origin + tmin * dir, invDir, tmax - tmin, uniforms.gridRes, &t, &norm)) {
      col = shade(origin + (tmin + t) * dir, dir, norm, (tmin + t) / tmax);
    }
  }

  return vec4f(pow(col, vec3f(0.4545)), 1.0);
}
