struct Uniforms
{
  cameraToWorld: mat4x4f,
  gridRes: f32,
  time: f32,
  freeValue: vec2f,
}

// TODO Replace by string literals
const WIDTH = 800;
const HEIGHT = WIDTH / 1.6;

const EPSILON = 0.001;
const HEMISPHERE = vec3f(0.3, 0.3, 0.6);

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

fn calcLightContribution(pos: vec3f, dir: vec3f, norm: vec3f, dist: f32) -> vec3f
{
  let border = vec3f(0.5 - 0.075);
  let wire = (vec3f(1) - abs(norm)) * abs(fract(pos) - vec3f(0.5));

  if(any(vec3<bool>(step(border, wire)))) {
    return vec3f(0);
  }

  var sky = (0.4 - norm.y * 0.6);
  return HEMISPHERE * sky * exp(4 * -dist);
}

fn renderBackground(o: vec3f, d: vec3f) -> vec3f
{
  return HEMISPHERE * 0.001;
}

@compute @workgroup_size(4,4,4)
fn c(@builtin(global_invocation_id) globalId: vec3u)
{
  let gridRes = u32(uniforms.gridRes);

  if(globalId.x >= gridRes || globalId.y >= gridRes || globalId.y >= gridRes) {
    return;
  }

  let gridMul = vec3u(1, gridRes, gridRes * gridRes); 
  let index = dot(gridMul, globalId);

  // TODO
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
  let time = uniforms.time; 
  let verticalFovInDeg = 50.0 - uniforms.freeValue.x;
  
  let fragCoord = position.xy;
  let uv = (fragCoord - vec2f(WIDTH, HEIGHT) * 0.5) / f32(HEIGHT);

  let origin = vec4f(uniforms.cameraToWorld[3]).xyz;
  let dirEyeSpace = normalize(vec3f(uv, -0.5 / tan(radians(0.5 * verticalFovInDeg))));
  let dir = (uniforms.cameraToWorld * vec4f(dirEyeSpace, 0.0)).xyz;
  let invDir = 1.0 / dir;
 
  var col = renderBackground(origin, dir);
  var tmin: f32;
  var tmax: f32;
  var t: f32;
  var norm: vec3f;

  if(intersectAabb(vec3f(0), vec3f(uniforms.gridRes), origin, invDir, &tmin, &tmax)) {
    tmin = max(tmin, 0.0) - EPSILON;
    if(traverseGrid(origin + tmin * dir, invDir, tmax - tmin, uniforms.gridRes, &t, &norm)) {
      col = calcLightContribution(origin + (tmin + t) * dir, dir, norm, (tmin + t) / tmax);
    }
  }

  return vec4f(pow(col, vec3f(0.4545)), 1.0);
}
