struct Uniforms {
  cameraToWorld: mat4x4f,
  resolution: vec2f,
  time: f32,
  value: f32
}

const PI = 3.14159;
const EPSILON = 0.01;
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

fn fSphere(p: vec3f, c: vec3f, r: f32) -> f32 {
  return length(p - c) - r;
}

fn fBox(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0))) + min(maxComponent(q), 0);
}

fn f(p: vec3f) -> f32 {
  //let d = fSphere(p, vec3f(), 1.25 - uniforms.value);
  //return max(d, fBox(p, vec3f(1)));
  return fSphere(p, vec3f(), 1.0 - uniforms.value);
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

fn calcNormal(p: vec3f) -> vec3f {
  let d = f(p);
  let dx = f(p + vec3f(EPSILON, 0, 0)) - d;
  let dy = f(p + vec3f(0, EPSILON, 0)) - d;
  let dz = f(p + vec3f(0, 0, EPSILON)) - d;
  return normalize(vec3f(dx, dy, dz));
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

  let verticalFovInDeg = 30.0;

  let fragCoord = vec2f(f32(globalId.x), f32(globalId.y));
  let uv = (fragCoord - uniforms.resolution * 0.5) / height;

  let o = vec4f(uniforms.cameraToWorld[3]).xyz;
  let d = (uniforms.cameraToWorld * normalize(vec4f(uv, -0.5 / tan(radians(verticalFovInDeg)), 0.0))).xyz;
 
  var col = renderBackground(o, d);

  let t = trace(o, d, 0.0025, 24.0, 64u);

  let hitPos = o + t * d;
  let normal = calcNormal(hitPos);

  if(t < INFINITY) {
    col = normal;
  }

  /*
  // Frag depth
  let zNear = 0.1; 
  let zFar = 100.0;
  let proj1 = zFar / (zNear - zFar);
  let proj2 = (zFar * zNear) / (zNear - zFar);
  let eyeFwd = (uniforms.cameraToWorld * vec4f(0, 0, -1, 0)).xyz;
  let camSpaceDepth = -t * dot(d, eyeFwd);
  let ndcFragDepth = -proj1 - proj2 / camSpaceDepth;
  col = vec3f(ndcFragDepth);
  */

  // Gamma
  col = pow(col, vec3f(0.4545));

  textureStore(outputTexture, vec2u(globalId.x, globalId.y), vec4f(col, 1.0));
}
