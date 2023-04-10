struct Uniforms {
  resolution: vec2<f32>,
  time: f32,
}

const INFINITY = 999999.0;

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

fn rotateX(v: vec3<f32>, a: f32) -> vec3<f32> {
  let c: f32 = cos(a);
  let s: f32 = sin(a);
  return vec3<f32>(v.x, v.y * c - v.z * s, v.z * c + v.y * s);
}

fn rotateY(v: vec3<f32>, a: f32) -> vec3<f32> {
  let c: f32 = cos(a);
  let s: f32 = sin(a);
  return vec3<f32>(v.x * c - v.z * s, v.y, v.z * c + v.x * s);
}

fn rotateZ(v: vec3<f32>, a: f32) -> vec3<f32> {
  let c: f32 = cos(a);
  let s: f32 = sin(a);
  return vec3<f32>(v.x * c - v.y * s, v.y * c + v.x * s, v.z);
}

// https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39
fn rand2d(n: vec2<f32>) -> f32 {
  return fract(sin(dot(n, vec2<f32>(12.9898, 4.1414))) * 43758.5453);
}

fn noise2d(n: vec2<f32>) -> f32 {
  let d = vec2<f32>(0., 1.);
  let b = floor(n);
  let f = smoothstep(vec2<f32>(0.), vec2<f32>(1.), fract(n));
  return mix(mix(rand2d(b), rand2d(b + d.yx), f.x), mix(rand2d(b + d.xy), rand2d(b + d.yy), f.x), f.y);
}

fn fSphere(p: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
  return length(p - c) - r;
}

fn fBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3<f32>(0.))) + min(max(q.x, max(q.y, q.z)), 0.);
}

fn fBoxFrame(p: vec3<f32>, b: vec3<f32>, e: f32) -> f32 {
  let q = abs(p) - b;
  let w = abs(q + e) - e;
  return min(min(
      length(max(vec3<f32>(q.x, w.y, w.z), vec3<f32>(0.))) + min(max(q.x, max(w.y, w.z)), 0.),
      length(max(vec3<f32>(w.x, q.y, w.z), vec3<f32>(0.))) + min(max(w.x, max(q.y, w.z)), 0.)),
      length(max(vec3<f32>(w.x, w.y, q.z), vec3<f32>(0.))) + min(max(w.x, max(w.y, q.z)), 0.));
}

fn f(p: vec3<f32>) -> f32 {
  //return fSphere(p, vec3<f32>(0.0, 0.0, 0.0), 1.0);
  return fBoxFrame(p, vec3<f32>(1.0, 1.0, 1.0), 0.1);
}

fn trace(o: vec3<f32>, d: vec3<f32>, pixelRadius: f32, tmax: f32, maxIterations: u32) -> f32 {

  var t = 0.0;
  
  var candidateError = INFINITY;
  var candidateT = t;

  for(var i = 0u; i<maxIterations; i++) {

    let radius = f(o + t * d);
    
    let error = radius / t;

    if(error < candidateError) {

      if(error < pixelRadius) {
        return t;
      }

      candidateT = t;
      candidateError = error;
    }

    t += radius;

    if(t > tmax) {
      return INFINITY;
    }
  }

  if(candidateError > pixelRadius) {
    return INFINITY;
  }

  return candidateT;
}

fn renderBackground(o: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(0.3, 0.3, 0.6);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {

  if (globalId.x >= 800 || globalId.y >= 500) {
    return;
  }

  var uv = vec2<f32>(
    f32(globalId.x) / 800.0,
    f32(globalId.y) / 500.0);

  let time: f32 = uniforms.time;

  let rotX = sin(time * 0.6) * 1.5;
  let rotY = sin(time * 0.4) * 1.7;
  let rotZ = sin(time * 0.3) * 1.9;

  uv = uv * 2.0 - 1.0;
  uv.x *= 800.0 / 500.0;

  var o = vec3<f32>(0.0, 0.0, 2.0);
  var d = normalize(vec3<f32>(uv, -1.0));

  d = rotateZ(d, rotZ);
  d = rotateX(d, rotX);
  d = rotateY(d, rotY);

  o = rotateX(o, rotX);
  o = rotateY(o, rotY);

  var col = renderBackground(o, d);

  let t = trace(o, d, 0.01, 32.0, 64u);

  if(t < INFINITY) {
    col = vec3<f32>(1.0, 0.0, 0.0);
  }

  textureStore(
    outputTexture,
    vec2<u32>(globalId.x, globalId.y),
    vec4<f32>(col, 1.0));
}
