// render voxel grid compute shader

struct Uniforms
{
  cameraToWorld: mat4x4f,
  gridRes: f32,
  time: f32,
  freeValue: vec2f,
}

const WIDTH = 800;
const HEIGHT = 500;
const EPSILON = 0.001;

fn maxComp(v: vec3f) -> f32
{
  return max(v.x, max(v.y, v.z));
}

fn minComp(v: vec3f) -> f32
{
  return min(v.x, min(v.y, v.z));
}

fn insideAabb(minExt: vec3f, maxExt: vec3f, p: vec3f) -> bool
{
  return p.x >= minExt.x && p.y >= minExt.y && p.z >= minExt.z && p.x < maxExt.x && p.y < maxExt.y && p.z < maxExt.z;
}

fn intersectAabb(minExt: vec3f, maxExt: vec3f, ori: vec3f, dir: vec3f, tmin: ptr<function, f32>, tmax: ptr<function, f32>) -> bool
{
  let invDir = 1.0 / dir;
 
  let t0 = (minExt - ori) * invDir;
  let t1 = (maxExt - ori) * invDir;
  
  *tmin = maxComp(min(t0, t1));
  *tmax = minComp(max(t0, t1));

  return *tmin <= *tmax && *tmax > 0.0;
}

fn traverseGrid(pos: vec3f, dir: vec3f, tmax: f32, gridRes: f32, dist: ptr<function, f32>) -> bool
{
  let gridOfs = vec3f(1.0, gridRes, gridRes * gridRes);
  let stepDir = sign(dir);
  var currCell = floor(pos);
  var delta = abs(length(dir) / dir);
  var t = (step(vec3f(0), stepDir) - stepDir * fract(pos)) * delta;

  *dist = 0.0;

  loop {
    if(grid[i32(dot(currCell, gridOfs))] > 0) { 
      return true;
    }

    let i = (u32(t.z <= t.x && t.z <= t.y) << 1) | u32(t.y <= t.x && t.y <= t.z);

    t[i] += delta[i];
    currCell[i] += stepDir[i];

    if(currCell[i] < 0.0 || currCell[i] >= gridRes) {
      return false;
    }

    *dist = t[i];

    if(*dist > tmax) {
      return false;
    }
  }
}

fn traverseGrid2(pos: vec3f, dir: vec3f, tmax: f32, gridRes: f32, t: ptr<function, f32>, col: ptr<function, vec3f>) -> bool
{
  let invDir = 1.0 / dir;
  let stepDir = step(vec3f(0), invDir);
  let gridOfs = vec3f(1.0, gridRes, gridRes * gridRes);
 
  *t = 0.0;

  loop {
    let p = pos + *t * dir;
    let cell = floor(p);

    if(minComp(cell) < 0.0 || maxComp(cell) >= gridRes) {
      *col = vec3f(1.0, 0.0, 0.0);
      return false;
    }

    if(grid[i32(dot(cell, gridOfs))] > 0) {
      *col = vec3f(0.0, 1.0, 0.0);
      return true;
    }

    let delta = minComp((stepDir - fract(p)) * invDir);
    *t += max(delta, EPSILON);

    if(*t > tmax) {
      *col = vec3f(0.0, 0.0, 1.0);
      return false;
    }
  }
}

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage> grid : array<u32>; 

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3u)
{
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) {
    return;
  }

  let time = uniforms.time; 
  let verticalFovInDeg = 60.0;
  
  let fragCoord = vec2f(f32(globalId.x), f32(globalId.y));
  let uv = (fragCoord - vec2f(WIDTH, HEIGHT) * 0.5) / f32(HEIGHT);

  let dirEyeSpace = normalize(vec3f(uv, -0.5 / tan(radians(0.5 * verticalFovInDeg))));
  let dir = (uniforms.cameraToWorld * vec4f(dirEyeSpace, 0.0)).xyz;
  let origin = vec4f(uniforms.cameraToWorld[3]).xyz;

  let s = uniforms.gridRes;
  var col = vec3f(0);
  var pos = origin;

  var tmin: f32;
  var tmax: f32;
  var t: f32;
  var hit: bool;

  // TODO Extend all rays to start in aabb during intersection test + calc proper tmax

  if(insideAabb(vec3(0), vec3f(s), origin)) {
    tmax = sqrt(s*s + s*s);
    hit = traverseGrid2(pos, dir, tmax, s, &t, &col);
  } else if(intersectAabb(vec3(0), vec3f(s), origin, dir, &tmin, &tmax)) {
      pos = origin + (tmin + EPSILON) * dir;
      hit = traverseGrid2(pos, dir, tmax, s, &t, &col);  
  }

  if(hit) {
    col = col * vec3f(1.0 - t / (tmax - tmin));
  }
  
  col = pow(col, vec3f(0.4545));

  textureStore(outputTexture, vec2u(globalId.x, globalId.y), vec4f(col, 1.0));
}

// blit vertex and fragment shader

struct Output {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> Output
{
  let pos = array<vec2f, 4>(
    vec2f(-1, 1), vec2f(-1, -1), vec2f(1), vec2f(1, -1));

  var output: Output;

  let h = vec2f(0.5);

  output.position = vec4f(pos[vertexIndex], 0, 1);
  output.texCoord = pos[vertexIndex] * h + h;

  return output;
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;

@fragment
fn fragmentMain(@location(0) texCoord: vec2f) -> @location(0) vec4f {
  return textureLoad(
    inputTexture, 
    vec2u(texCoord * vec2f(WIDTH, HEIGHT)),
    0);
}
