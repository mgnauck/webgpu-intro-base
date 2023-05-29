struct Uniforms {
  cameraToWorld: mat4x4f,
  resolution: vec2f,
  time: f32,
  value: f32
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

const MAX_LIGHT_COUNT = 1;
const PI = 3.14159;
const EPSILON = 0.001;
const INFINITY = 999999.0;

const lights = array<Light, MAX_LIGHT_COUNT>(
  Light(vec3f(10, 2, 10), vec3f(0.7, 0.85, 1.0)),
  //Light(vec3f(-10, -10, 10), vec3f(1.0, 0.85, 0.7))
  );

const sphereMaterial = Material(vec3f(0.005), vec3f(0.01, 0.6, 0.3), vec3f(0.8), 28.0);

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

fn fBox(p: vec3f, e: vec3f, r: f32) -> f32 {
  return length(max(abs(p) - e, vec3f(0))) - r;
}

fn fSphere(p: vec3f, r: f32) -> f32 {
  return length(p) - r;
}

fn f(p: vec3f) -> f32 {
  let dbox = fBox(p + vec3f(0.0, 0.0, 0.0), vec3f(3.0, 0.1, 3.0), 0.01);
  let dsph = fSphere(p + vec3f(0.0, -1.0 + sin(uniforms.time) * 0.5, 0.0), 1.0 + uniforms.value);
  return min(dbox, dsph);
  //return fBox(p, vec3f(0.5), 0.1);
  //return fSphere(p, 0.5);
}

fn traceSimple(o: vec3f, d: vec3f, tmax: f32, maxIterations: u32) -> f32 {
	var t = 0.0;
  var fSign = 1.0;
  if(f(o) < 0.0) {
    fSign = -1.0;
  }
	for(var i=0u; i<maxIterations; i++) {
		let dist = fSign * f(o + t * d);
		if(abs(dist) < 0.001) {
			return t;
		}
		t += dist;
		if(t >= tmax) {
			return INFINITY;
		}
	}
	return INFINITY;
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

fn calcOcclusion(p: vec3f, n: vec3f) -> f32 {
  //let nAdj = normalize(n + vec3f(0, 0.5, 0));
  let samCnt = 5.0;
  let weight = 1.0;
  var sum = 0.0;
  for(var i=1.0; i<samCnt; i+=1.0) {
    let dist = i / samCnt;
    sum += (dist - f(p + n * dist)) / pow(2.0, i - 1.0);
  }
  return clamp(1.0 - weight * sum, 0.0, 1.0);
}

fn calcLightContribution(m: Material, eyePos: vec3f, eyeDir: vec3f, n: vec3f) -> vec3f {
  var col = m.ambient;
  let occlusion = calcOcclusion(eyePos, n);
  for(var i=0u; i<MAX_LIGHT_COUNT; i++) {
    let light = lights[i];
    
    let lv = normalize(light.position - eyePos);
    let diffuseFactor = max(dot(n, lv), 0.0) * occlusion; 
    
    let rv = reflect(lv, n);
    let reflectedLightAngle = dot(rv, eyeDir);
    var specularFactor = 0.0;
    if(diffuseFactor > 0.0 && reflectedLightAngle > 0.0) {
      specularFactor = pow(reflectedLightAngle, m.exponent);
    }
    
    col += (diffuseFactor * m.diffuse + specularFactor * m.specular) * light.color;
  }
  return col;
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

  // TODO provide proj and tan(fov) as uniforms

  let zNear = 0.1;
  let zFar = 100.0;
  let verticalFovInDeg = 60.0;

  // wgpu clip space z is from 0 to 1
  let proj1 = zFar / (zFar - zNear);
  let proj2 = zFar * zNear / (zFar - zNear);

  let fragCoord = vec2f(f32(globalId.x), f32(globalId.y));
  let uv = (fragCoord - uniforms.resolution * 0.5) / height;

  let dirEyeSpace = normalize(vec3f(uv, -0.5 / tan(radians(0.5 * verticalFovInDeg))));
  let dir = (uniforms.cameraToWorld * vec4f(dirEyeSpace, 0.0)).xyz;
  let origin = vec4f(uniforms.cameraToWorld[3]).xyz;
 
  var col = renderBackground(origin, dir);

  var t = trace(origin, dir, 0.01, zFar, 96u);
  //var t = traceSimple(origin, dir, zFar, 64u);
 
  var fSign = 1.0;
  if(f(origin) < 0.0) {
    fSign = -1.0;
  }

  var hitPos = origin + t * dir;
  let normal = fSign * calcNormal(hitPos);

  if(t < INFINITY) {
    col = calcLightContribution(sphereMaterial, hitPos, dir, normal);
  }

  //let fragDepth = proj1 + proj2 / (t * dirEyeSpace).z;
  //col = vec3f(ndcFragDepth);

  col = pow(col, vec3f(0.4545));

  textureStore(outputTexture, vec2u(globalId.x, globalId.y), vec4f(col, 1.0));
}
