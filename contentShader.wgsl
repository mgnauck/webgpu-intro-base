// render voxel grid compute shader

struct Uniforms
{
  cameraToWorld: mat4x4f,
  gridRes: f32,
  time: f32,
  freeValue: vec2f,
}

struct Light {
  position: vec3f,
  color: vec3f
}

struct Material {
  ambient: vec3f,
  diffuse: vec3f,
  specular: vec3f,
  exponent: f32
}

const WIDTH = 800;
const HEIGHT = 500;
const EPSILON = 0.001;

const MAX_LIGHT_COUNT = 1;

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage> grid : array<u32>; 

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
  let gridOfs = vec3f(1.0, gridRes, gridRes * gridRes); 
  let stepDir = sign(invDir);
  var cell = floor(ori);
  var t = (vec3f(0.5) + 0.5 * stepDir - fract(ori)) * invDir;

  while(*dist < tmax) {
    let mask = vec3f(f32(t.x <= t.y && t.x <= t.z), f32(t.y <= t.x && t.y <= t.z), f32(t.z <= t.x && t.z <= t.y));
 
    t += mask * stepDir * invDir;
    cell += mask * stepDir;

    *dist = dot(mask, (vec3f(0.5) - 0.5 * stepDir + cell - ori) * invDir);
    
    if(grid[u32(dot(gridOfs, cell))] > 0) {
      *norm = -mask * stepDir;
      return true;
    }
  }

  return false;
}

fn calcLightContribution(eyePos: vec3f, eyeDir: vec3f, n: vec3f, m: Material, lights: array<Light, MAX_LIGHT_COUNT>) -> vec3f {
  var col = m.ambient;
  for(var i=0u; i<MAX_LIGHT_COUNT; i++) {
    let light = lights[i];
    
    let lv = normalize(light.position - eyePos);
    let diffuseFactor = max(dot(n, lv), 0.0);

    let lv2 = light.position - eyePos;
    let dist = dot(lv2, lv2);
    let attenuation = 1.0 / (1.0 + 0.00006 * dist);
   
    //col += attenuation * diffuseFactor * m.diffuse * light.color;
    
    let rv = reflect(lv, n);
    let reflectedLightAngle = dot(rv, eyeDir);
    var specularFactor = 0.0;
    if(diffuseFactor > 0.0 && reflectedLightAngle > 0.0) {
      specularFactor = pow(reflectedLightAngle, m.exponent);
    }
    
    col += attenuation * (diffuseFactor * m.diffuse + specularFactor * m.specular) * light.color;
    
  }
  return col;
}

fn renderBackground(o: vec3f, d: vec3f) -> vec3f {
  return vec3f();
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3u)
{
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) {
    return;
  }

 var lights = array<Light, MAX_LIGHT_COUNT>(
  Light(vec3f(uniforms.gridRes, 2.0 * uniforms.gridRes, uniforms.gridRes), vec3f(0.7, 0.85, 1.0)),
  );

  const defaultMaterial = Material(vec3f(0.005), vec3f(0.01, 0.6, 0.3), vec3f(0.3), 28.0);

  let time = uniforms.time; 
  let verticalFovInDeg = 60.0;
  
  let fragCoord = vec2f(f32(globalId.x), f32(globalId.y));
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
    tmax = tmax - EPSILON - tmin;
    if(traverseGrid(origin + tmin * dir, invDir, tmax, uniforms.gridRes, &t, &norm)) {
      //col = abs(norm); // * (1.0 - t / tmax);
      col = calcLightContribution(origin + t * dir, dir, norm, defaultMaterial, lights) * (1.0 - t / tmax);
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
