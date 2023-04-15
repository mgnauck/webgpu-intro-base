struct Output {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f
}

@vertex
fn main(@builtin(vertex_index) vertexIndex: u32) -> Output {

  let pos = array<vec2f, 4>(
    vec2f(-1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(1, -1));

  var output: Output;

  let h = vec2f(0.5, 0.5);

  output.position = vec4f(pos[vertexIndex], 0, 1);
  output.texCoord = pos[vertexIndex] * h + h;

  return output;
}
