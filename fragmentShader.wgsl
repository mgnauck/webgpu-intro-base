struct Uniforms
{
  resolution : vec2<f32>,
  time : f32,
}
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

@fragment
fn main( @builtin(position) position : vec4<f32>)  -> @location(0) vec4<f32>
{
  return vec4<f32>( 0.3, abs( sin( uniforms.time * 0.003 ) ), 0.3, 1.0 );
}
