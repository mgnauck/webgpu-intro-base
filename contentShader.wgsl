struct Uniforms {
  resolution: vec2f,
  time: f32,
}

const INFINITY = 999999.0;

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

fn rotateX(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(v.x, v.y * c - v.z * s, v.z * c + v.y * s);
}

fn rotateY(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(v.x * c - v.z * s, v.y, v.z * c + v.x * s);
}

fn rotateZ(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(v.x * c - v.y * s, v.y * c + v.x * s, v.z);
}

fn maxComponent(v: vec3f) -> f32 {
  return max(max(v.x, v.y), v.z);
}

// https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39
fn rand2d(n: vec2f) -> f32 {
  return fract(sin(dot(n, vec2f(12.9898, 4.1414))) * 43758.5453);
}

fn noise2d(n: vec2f) -> f32 {
  let d = vec2f(0, 1);
  let b = floor(n);
  let f = smoothstep(vec2f(0), vec2f(1), fract(n));
  return mix(mix(rand2d(b), rand2d(b + d.yx), f.x), mix(rand2d(b + d.xy), rand2d(b + d.yy), f.x), f.y);
}

fn fSphere(p: vec3f, c: vec3f, r: f32) -> f32 {
  return length(p - c) - r;
}

fn fBox(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0))) + min(maxComponent(q), 0);
}

fn f(p: vec3f) -> f32 {
  let d = fSphere(p, vec3f(), 1.25);
  return min(d, -fBox(p, vec3f(1)));
}

fn trace(o: vec3f, d: vec3f, pixelRadius: f32, tmax: f32, maxIterations: u32) -> f32 {

  var t = 0.0;
  
  var candidateError = INFINITY;
  var candidateT = 0.0;

  var omega = 1.2;
  var previousRadius = 0.0;
  var stepLength = 0.0;
  var functionSign = 1.0;

  if(f(o) < 0.0) {
    functionSign = -1.0;
  }

  for(var i=0u; i<maxIterations; i++) {

    let signedRadius = functionSign * f(o + t * d);
    let radius = abs(signedRadius);
     
    if(omega > 1.0 && (radius + previousRadius) < stepLength) {

      stepLength -= omega * stepLength;
      omega = 1.0;

    } else {
      
      stepLength = signedRadius * omega;

      let error = radius / t;
      if(error < candidateError) {

        if(error < pixelRadius) {
          return t;
        }

        candidateT = t;
        candidateError = error;
      }
    }

    previousRadius = radius;

    t += stepLength;

    if(t > tmax) {
      return INFINITY;
    }
  }

  if(candidateError > pixelRadius) {
    return INFINITY;
  }

  return candidateT;
}

fn renderBackground(o: vec3f, d: vec3f) -> vec3f {
  return vec3f();
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3u) {

  let width = uniforms.resolution.x;
  let height = uniforms.resolution.y;

  if (globalId.x >= u32(width) || globalId.y >= u32(height)) {
    return;
  }

  let time = uniforms.time;

  var uv = vec2f(
    f32(globalId.x) / width,
    f32(globalId.y) / height);

  let rotX = sin(time * 0.6) * 1.5;
  let rotY = sin(time * 0.4) * 1.7;
  let rotZ = sin(time * 0.3) * 1.9;

  uv = uv * 2.0 - 1.0;
  uv.x *= width / height;

  var o = vec3f(0.0, 0.0, 2.0);
  var d = normalize(vec3f(uv, -1.0));

  d = rotateZ(d, rotZ);
  d = rotateX(d, rotX);
  d = rotateY(d, rotY);

  o = rotateX(o, rotX);
  o = rotateY(o, rotY);

  var col = renderBackground(o, d);

  let t = trace(o, d, 0.0025, 24.0, 64u);

  if(t < INFINITY) {
    col = vec3f(1.0, 0.0, 0.0);
  }

  textureStore(outputTexture, vec2u(globalId.x, globalId.y), vec4f(col, 1.0));
}
