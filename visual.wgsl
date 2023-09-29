struct Hit
{
  index: i32,
  pos: vec3f,
  norm: vec3f,
  state: u32,
  dist: f32,
}

const WIDTH = 1024;
const HEIGHT = WIDTH / 1.77;

const gridMul = vec3i(1, 256, 256 * 256);
const gridMulF = vec3f(gridMul);

@group(0) @binding(0) var<storage> uniforms: array<f32>;
@group(0) @binding(1) var<storage> grid: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputGrid: array<u32>;
@group(0) @binding(3) var<storage> rules: array<u32>;

fn getCell(x: i32, y: i32, z: i32) -> u32
{
  // Consider only states 0 and 1. Cells in refactory period do NOT count as active neighbours, i.e. are counted as 0.
  return u32(1 - min(abs(1 - i32(grid[gridMul.z * z + gridMul.y * y + x])), 1));
}

fn getMooreNeighbourCountWrap(pos: vec3i) -> u32
{
  let res = gridMul.y;
  let dec = vec3i((pos.x - 1) % res, (pos.y - 1) % res, (pos.z - 1) % res);
  let inc = vec3i((pos.x + 1) % res, (pos.y + 1) % res, (pos.z + 1) % res);

  return  getCell(pos.x, inc.y, pos.z) +
          getCell(inc.x, inc.y, pos.z) +
          getCell(dec.x, inc.y, pos.z) +
          getCell(pos.x, inc.y, inc.z) +
          getCell(pos.x, inc.y, dec.z) +
          getCell(inc.x, inc.y, inc.z) +
          getCell(inc.x, inc.y, dec.z) +
          getCell(dec.x, inc.y, inc.z) +
          getCell(dec.x, inc.y, dec.z) +
          getCell(inc.x, pos.y, pos.z) +
          getCell(dec.x, pos.y, pos.z) +
          getCell(pos.x, pos.y, inc.z) +
          getCell(pos.x, pos.y, dec.z) +
          getCell(inc.x, pos.y, inc.z) +
          getCell(inc.x, pos.y, dec.z) +
          getCell(dec.x, pos.y, inc.z) +
          getCell(dec.x, pos.y, dec.z) +
          getCell(pos.x, dec.y, pos.z) +
          getCell(inc.x, dec.y, pos.z) +
          getCell(dec.x, dec.y, pos.z) +
          getCell(pos.x, dec.y, inc.z) +
          getCell(pos.x, dec.y, dec.z) +
          getCell(inc.x, dec.y, inc.z) +
          getCell(inc.x, dec.y, dec.z) +
          getCell(dec.x, dec.y, inc.z) +
          getCell(dec.x, dec.y, dec.z);
}

@compute @workgroup_size(4,4,4)
fn cM(@builtin(global_invocation_id) globalId: vec3u)
{
  let pos = vec3i(globalId);
  let index = dot(pos, gridMul);
  let value = grid[index];

  switch(value) {
    case 0: {
      outputGrid[index] = rules[1 + 27 + getMooreNeighbourCountWrap(pos)];
    }
    case 1: {
      // Dying state 1 goes to 2 (or dies directly by being moduloed to 0, in case there are only 2 states)
      outputGrid[index] = (1 + 1 - rules[1 + getMooreNeighbourCountWrap(pos)]) % rules[0];
    }
    default {
      // Refactory period
      outputGrid[index] = min(value + 1, rules[0]) % rules[0]; 
    }
  }
}

fn minComp(v: vec3f) -> f32
{
  return min(v.x, min(v.y, v.z));
}

fn maxComp(v: vec3f) -> f32
{
  return max(v.x, max(v.y, v.z));
}

fn intersectAabb(minExt: vec3f, maxExt: vec3f, ori: vec3f, invDir: vec3f, tmin: ptr<function, f32>, tmax: ptr<function, f32>) -> bool
{
  let t0 = (minExt - ori) * invDir;
  let t1 = (maxExt - ori) * invDir;
  
  *tmin = maxComp(min(t0, t1));
  *tmax = minComp(max(t0, t1));

  return *tmin <= *tmax && *tmax > 0.0;
}

fn traverseGrid(ori: vec3f, invDir: vec3f, tmax: f32, hit: ptr<function, Hit>) -> bool
{
  var stepDir = sign(invDir);
  var t = (vec3f(0.5) + 0.5 * stepDir - fract(ori)) * invDir;
  var mask = vec3f(0);

  (*hit).dist = 0.0;
  (*hit).index = i32(dot(gridMulF, floor(vec3f(gridMulF.y / 2) + ori)));
  
  loop {
    (*hit).state = grid[(*hit).index];
    if((*hit).state > 0) {
      (*hit).norm = mask * -stepDir;
      return true;
    }

    (*hit).dist = minComp(t);
    if((*hit).dist >= tmax) {
      return false;
    }
 
    mask.x = f32(t.x <= t.y && t.x <= t.z);
    mask.y = f32(t.y <= t.x && t.y <= t.z);
    mask.z = f32(t.z <= t.x && t.z <= t.y);

    t += mask * stepDir * invDir;
    (*hit).index += i32(dot(gridMulF, mask * stepDir));
  }
}

fn calcOcclusion(pos: vec3f, index: i32, norm: vec3i) -> f32
{
  let above = index + dot(gridMul, norm);
  let dir = abs(norm);
  let hori = dot(gridMul, dir.yzx);
  let vert = dot(gridMul, dir.zxy);

  let edgeCellStates = vec4f(
    f32(min(1, grid[above + hori])),
    f32(min(1, grid[above - hori])),
    f32(min(1, grid[above + vert])),
    f32(min(1, grid[above - vert])));

  let cornerCellStates = vec4f(
    f32(min(1, grid[above + hori + vert])),
    f32(min(1, grid[above - hori + vert])),
    f32(min(1, grid[above + hori - vert])),
    f32(min(1, grid[above - hori - vert])));

  let uvLocal = fract(pos);
  let uv = vec2f(dot(uvLocal, vec3f(dir.yzx)), dot(uvLocal, vec3f(dir.zxy)));
  let uvInv = vec2f(1) - uv;

  let edgeOcc = edgeCellStates * vec4f(uv.x, uvInv.x, uv.y, uvInv.y);
  let cornerOcc = cornerCellStates * vec4f(uv.x * uv.y, uvInv.x * uv.y, uv.x * uvInv.y, uvInv.x * uvInv.y) * (vec4f(1.0) - edgeCellStates.xzwy) * (vec4f(1.0) - edgeCellStates.zyxw);

  return 1.0 - (edgeOcc.x + edgeOcc.y + edgeOcc.z + edgeOcc.w + cornerOcc.x + cornerOcc.y + cornerOcc.z + cornerOcc.w) * 0.5;
}

fn shade(pos: vec3f, dir: vec3f, tmax: f32, hit: ptr<function, Hit>) -> vec3f
{
  // Wireframe, better add AA
  /*let border = vec3f(0.5 - 0.05);
  let wire = (vec3f(1) - abs((*hit).norm)) * abs(fract(pos) - vec3f(0.5));

  if(any(vec3<bool>(step(border, wire)))) {
    return vec3f(0);
  }*/

  let cnt = f32(rules[0]);
  let val = cnt / min(f32((*hit).state), cnt);
  let sky = 0.4 + (*hit).norm.y * 0.6;
  let col = vec3f(0.005) + (1.0 - 0.15 * (cnt - 5.0)) * (vec3f(0.5) + pos / gridMulF.y) * sky * sky * val * val * 0.3 * exp(-3.5 * (*hit).dist / tmax);
  let occ = calcOcclusion(pos, (*hit).index, vec3i((*hit).norm));

  return col * occ * occ * occ;
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn filmicToneACES(x: vec3f) -> vec3f
{
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return saturate(x * (a * x + vec3f(b)) / (x * (c * x + vec3f(d)) + vec3f(e)));
}

@vertex
fn vM(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f
{
  let pos = array<vec2f, 4>(vec2f(-1, 1), vec2f(-1, -1), vec2f(1), vec2f(1, -1));
  return vec4f(pos[vertexIndex], 0, 1);
}

@fragment
fn fM(@builtin(position) position: vec4f) -> @location(0) vec4f
{
  let dirEyeSpace = normalize(vec3f((position.xy - vec2f(WIDTH, HEIGHT) * 0.5) / f32(HEIGHT), 1 /* FOV */));

  let ori = vec3f(uniforms[0] * cos(uniforms[2]) * cos(uniforms[1]), uniforms[0] * sin(uniforms[2]), uniforms[0] * cos(uniforms[2]) * sin(uniforms[1]));

  let fwd = normalize(-ori);
  let ri = normalize(cross(fwd, vec3f(0, 1, 0)));
  let up = cross(ri, fwd);

  var dir = ri * dirEyeSpace.x - up * dirEyeSpace.y + fwd * dirEyeSpace.z;

  let halfGrid = vec3f(gridMulF.y / 2);
  let invDir = 1.0 / dir;

  var tmin: f32;
  var tmax: f32;
  var col = vec3f(0);
  var hit: Hit;

  if(intersectAabb(-halfGrid, halfGrid, ori, invDir, &tmin, &tmax)) {
    tmin = max(tmin + 0.001, 0.0); // EPSILON 0.001
    tmax = tmax - 0.001 - tmin;
    if(traverseGrid(ori + tmin * dir, invDir, tmax, &hit)) {
      col = shade(ori + (tmin + hit.dist) * dir, dir, tmax, &hit);
    }
  }

  let fadeIn = 1.0 - smoothstep(0.0, 25.0, uniforms[3]);
  let fadeOut = smoothstep(300.0 - 25.0, 300.0, uniforms[3]);
  return vec4f(pow(filmicToneACES(mix(col, vec3f(0), fadeIn + fadeOut)), vec3f(0.4545)), 1.0);
}
