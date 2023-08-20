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

const states = 5u;
// TODO Separate rules for dead and alive cells
const rules = array<u32, 27>(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

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

fn getCell(cellX: u32, cellY: u32, cellZ: u32, gridMul: ptr<function, vec3u>) -> u32
{
  let v = grid[(cellX % (*gridMul).y) + (cellY % (*gridMul).y) * (*gridMul).y + (cellZ % (*gridMul).y) * (*gridMul).z]; 
  // Consider only states 0 and 1. Cells in refactory period do NOT count as active neighbours.
  return 1 - min(abs(1 - v), 1);
}

fn getNeumannNeighbourCountWrap(cell: vec3u, gridMul: ptr<function, vec3u>) -> u32
{
  return  getCell(cell.x + 1, cell.y,     cell.z, gridMul) +
          getCell(cell.x - 1, cell.y,     cell.z, gridMul) +
          getCell(cell.x,     cell.y + 1, cell.z, gridMul) +
          getCell(cell.x,     cell.y - 1, cell.z, gridMul) +
          getCell(cell.x,     cell.y,     cell.z + 1, gridMul) +
          getCell(cell.x,     cell.y,     cell.z - 1, gridMul);
}

fn getMooreNeighbourCountWrap(cell: vec3u, gridMul: ptr<function, vec3u>) -> u32
{
  return  getCell(cell.x,     cell.y + 1, cell.z, gridMul) +
          getCell(cell.x + 1, cell.y + 1, cell.z, gridMul) +
          getCell(cell.x - 1, cell.y + 1, cell.z, gridMul) +
          getCell(cell.x,     cell.y + 1, cell.z + 1, gridMul) +
          getCell(cell.x,     cell.y + 1, cell.z - 1, gridMul) +
          getCell(cell.x + 1, cell.y + 1, cell.z + 1, gridMul) +
          getCell(cell.x + 1, cell.y + 1, cell.z - 1, gridMul) +
          getCell(cell.x - 1, cell.y + 1, cell.z + 1, gridMul) +
          getCell(cell.x - 1, cell.y + 1, cell.z - 1, gridMul) +
          getCell(cell.x + 1, cell.y,     cell.z, gridMul) +
          getCell(cell.x - 1, cell.y,     cell.z, gridMul) +
          getCell(cell.x,     cell.y,     cell.z + 1, gridMul) +
          getCell(cell.x,     cell.y,     cell.z - 1, gridMul) +
          getCell(cell.x + 1, cell.y,     cell.z + 1, gridMul) +
          getCell(cell.x + 1, cell.y,     cell.z - 1, gridMul) +
          getCell(cell.x - 1, cell.y,     cell.z + 1, gridMul) +
          getCell(cell.x - 1, cell.y,     cell.z - 1, gridMul) +
          getCell(cell.x,     cell.y - 1, cell.z, gridMul) +
          getCell(cell.x + 1, cell.y - 1, cell.z, gridMul) +
          getCell(cell.x - 1, cell.y - 1, cell.z, gridMul) +
          getCell(cell.x,     cell.y - 1, cell.z + 1, gridMul) +
          getCell(cell.x,     cell.y - 1, cell.z - 1, gridMul) +
          getCell(cell.x + 1, cell.y - 1, cell.z + 1, gridMul) +
          getCell(cell.x + 1, cell.y - 1, cell.z - 1, gridMul) +
          getCell(cell.x - 1, cell.y - 1, cell.z + 1, gridMul) +
          getCell(cell.x - 1, cell.y - 1, cell.z - 1, gridMul);
}

fn getMooreNeighbourCount(cell: vec3u, gridMul: ptr<function, vec3u>) -> u32
{
  let minBound = vec3u(step(vec3f(0), vec3f(cell) - vec3f(1)));
  let maxBound = vec3u(step(vec3f(0), vec3f(f32((*gridMul).y)) - vec3f(cell) + vec3f(1)));
  return  maxBound.y * getCell(cell.x, cell.y + 1, cell.z, gridMul) +
          maxBound.x * maxBound.y * getCell(cell.x + 1, cell.y + 1, cell.z, gridMul) +
          minBound.x * maxBound.y * getCell(cell.x - 1, cell.y + 1, cell.z, gridMul) +
          maxBound.y * maxBound.z * getCell(cell.x, cell.y + 1, cell.z + 1, gridMul) +
          maxBound.y * minBound.z * getCell(cell.x, cell.y + 1, cell.z - 1, gridMul) +
          maxBound.x * maxBound.y * maxBound.z * getCell(cell.x + 1, cell.y + 1, cell.z + 1, gridMul) +
          maxBound.x * maxBound.y * minBound.z * getCell(cell.x + 1, cell.y + 1, cell.z - 1, gridMul) +
          minBound.x * maxBound.y * maxBound.z * getCell(cell.x - 1, cell.y + 1, cell.z + 1, gridMul) +
          minBound.x * maxBound.y * minBound.z * getCell(cell.x - 1, cell.y + 1, cell.z - 1, gridMul) +
          maxBound.x * getCell(cell.x + 1, cell.y, cell.z, gridMul) +
          minBound.x * getCell(cell.x - 1, cell.y, cell.z, gridMul) +
          maxBound.z * getCell(cell.x, cell.y, cell.z + 1, gridMul) +
          minBound.z * getCell(cell.x, cell.y, cell.z - 1, gridMul) +
          maxBound.x * maxBound.z * getCell(cell.x + 1, cell.y, cell.z + 1, gridMul) +
          maxBound.x * minBound.z * getCell(cell.x + 1, cell.y, cell.z - 1, gridMul) +
          minBound.x * maxBound.z * getCell(cell.x - 1, cell.y, cell.z + 1, gridMul) +
          minBound.x * minBound.z * getCell(cell.x - 1, cell.y, cell.z - 1, gridMul) +
          minBound.y * getCell(cell.x, cell.y - 1, cell.z, gridMul) +
          maxBound.x * minBound.z * getCell(cell.x + 1, cell.y - 1, cell.z, gridMul) +
          minBound.x * minBound.y * getCell(cell.x - 1, cell.y - 1, cell.z, gridMul) +
          minBound.y * maxBound.z * getCell(cell.x, cell.y - 1, cell.z + 1, gridMul) +
          minBound.y * minBound.z * getCell(cell.x, cell.y - 1, cell.z - 1, gridMul) +
          maxBound.x * minBound.y * maxBound.z * getCell(cell.x + 1, cell.y - 1, cell.z + 1, gridMul) +
          maxBound.x * minBound.y * minBound.z * getCell(cell.x + 1, cell.y - 1, cell.z - 1, gridMul) +
          minBound.x * minBound.y * maxBound.z * getCell(cell.x - 1, cell.y - 1, cell.z + 1, gridMul) +
          minBound.x * minBound.y * minBound.z * getCell(cell.x - 1, cell.y - 1, cell.z - 1, gridMul);
}

@compute @workgroup_size(4,4,4)
fn c(@builtin(global_invocation_id) globalId: vec3u)
{
  let gridRes = u32(uniforms.gridRes);
  var gridMul = vec3u(1, gridRes, gridRes * gridRes);
 
  let index = dot(globalId, gridMul);
  let value = grid[index];
  let count = getMooreNeighbourCount(globalId, &gridMul);

  // TODO Fix general rule handling
/*
  switch(value) {
    case 0: {
      outputGrid[index] = rules[count];
    }
    case 1: {
      outputGrid[index] += 1 - rules[count];
    }
    default: {
      outputGrid[index] = (value + 1) % states;
    }
  }*/
  if(value == 1) {
    if(count == 4) {
      outputGrid[index] = 1;
    } else {
      outputGrid[index] = 2;
    }
  }
  if(value == 0 && count == 4) {
    outputGrid[index] = 1;
  }
  if(value > 1) {
    outputGrid[index] = (value + 1) % states;
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
