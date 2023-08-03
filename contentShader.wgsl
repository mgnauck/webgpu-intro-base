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
  var delta = abs(1.0 / dir);
  var t = (step(vec3f(0), stepDir) - stepDir * fract(pos)) * delta;

  *dist = 0.0;

  while(*dist < tmax) {
    if(grid[i32(dot(gridOfs, currCell))] > 0) { 
      return true;
    }

    let i = (u32(t.z <= t.x && t.z <= t.y) << 1) | u32(t.y <= t.x && t.y <= t.z);

    t[i] += delta[i];
    currCell[i] += stepDir[i];

    *dist = t[i];
  }

  return false;
}

fn traverseGrid2(pos: vec3f, dir: vec3f, tmin: f32, tmax: f32, gridRes: f32, t: ptr<function, f32>) -> bool
{
  let invDir = 1.0 / dir;
  let stepDir = step(vec3f(0), invDir);
  let gridOfs = vec3f(1.0, gridRes, gridRes * gridRes);

  *t = tmin;

  while(*t < tmax) {
    let p = pos + *t * dir;
    let cell = floor(p);

    if(grid[i32(dot(gridOfs, cell))] > 0) {
      return true;
    }

    let delta = minComp((stepDir - fract(p)) * invDir);
    *t += max(delta, EPSILON);
  }

  return false;
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

  var col = vec3f(0);

  var tmin: f32;
  var tmax: f32;
  if(intersectAabb(vec3f(0), vec3f(uniforms.gridRes), origin, dir, &tmin, &tmax)) {
    var t: f32;
    if(traverseGrid2(origin, dir, max(tmin, 0.0) + EPSILON, tmax - EPSILON, uniforms.gridRes, &t)) { 
    //if(traverseGrid(origin + (max(tmin, 0.0) + EPSILON) * dir, dir, tmax - EPSILON, uniforms.gridRes, &t)) {
      col += vec3f(0.0, 0.0, 1.0);
    }
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
