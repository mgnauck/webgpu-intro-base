// render voxel grid compute shader

struct Uniforms
{
  cameraToWorld: mat4x4f,
  voxelGridRes: u32,
  time: f32,
  freeValue: vec2f,
}

const WIDTH = 180;
const HEIGHT = 180;

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage> voxelGrid : array<u32>; 

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
 
  var col = vec3f(0.3, 0.3, 0.6);
  if(voxelGrid[WIDTH * globalId.y + globalId.x] == 0) {
    col = vec3f(0.0);
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
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> Output {

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
