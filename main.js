"use strict";

const FULLSCREEN = false;
const AUDIO = false;
const SHADER_RELOAD = false;

const ASPECT = 1.6;
const CANVAS_WIDTH = 400 * ASPECT;
const CANVAS_HEIGHT = 400;

const AUDIO_BUFFER_WIDTH = 1024;
const AUDIO_BUFFER_HEIGHT = 1024;

const vertexShader = `
@vertex
fn main( @builtin(vertex_index) vertex_index : u32 ) -> @builtin(position) vec4<f32>
{  
  var pos = array<vec2<f32>, 4>( vec2( -1.0, 1.0 ), vec2( -1.0, -1.0 ), vec2( 1.0, 1.0 ), vec2( 1.0, -1.0 ) );
  return vec4<f32>( pos[vertex_index], 0.0, 1.0 );
}
`;

const audioShader = `
struct Uniforms
{
  resolution : vec2<f32>,
  sample_rate : f32,
}
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

@fragment
fn main( @builtin(position) position : vec4<f32> ) -> @location(0) vec2<f32>
{
  let time : f32 = ( uniforms.resolution.x * ( position.y - 0.5 ) + position.x - 0.5 ) / uniforms.sample_rate;
  let val : f32 = sin( time * 440.0 * 1.2 * 3.1415 );
  return vec2<f32>( val, val );
}
`;

const videoShaderFile = "http://localhost:8000/fragmentShader.wgsl";
const videoShader = `
struct Uniforms
{
  resolution : vec2<f32>,
  time : f32,
}
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

@fragment
fn main( @builtin(position) position : vec4<f32>)  -> @location(0) vec4<f32>
{
  return vec4<f32>( 0.6, 0.3, 0.3, 1.0 );
}
`;

let audioContext;
let audioBufferSourceNode;

let device;
let uniformBuffer;
let uniformBindGroupLayout;
let uniformBindGroup;
let renderPassDescriptor;
let pipeline;

let canvas;
let context;
let presentationFormat;

let start;
let reloadData;

function setupCanvasAndContext()
{
  // Canvas
  document.body.innerHTML = "<button>CLICK<canvas style='width:0;cursor:none'>";
  canvas = document.querySelector( "canvas" );
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  // Expecting format "bgra8unorm"
  presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  if( presentationFormat !== "bgra8unorm" )
  {
    throw new Error( `Expected canvas pixel format of 'bgra8unorm' but was '${presentationFormat}'.`)
  }

  // Context
  context = canvas.getContext( "webgpu" );

  context.configure(
  {
    device: device,
    format: presentationFormat,
    alphaMode: "opaque"
  } );
}

function createRenderPassDescriptor( view )
{
  return {
    colorAttachments:
    [
      {
        view,
        clearValue: { r: 0.3, g: 0.3, b: 0.3, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ] };
}

function createPipeline( vertexShaderCode, fragmentShaderCode, presentationFormat, bindGroupLayout )
{
  return device.createRenderPipelineAsync(
    {
      layout: ( bindGroupLayout === undefined ) ? "auto" : device.createPipelineLayout( { bindGroupLayouts: [ bindGroupLayout ] } ),
      vertex:
        {
          module: device.createShaderModule(
            {
              code: vertexShaderCode
            }),
          entryPoint: "main"
        },
      fragment:
        {
          module: device.createShaderModule(
            {
              code: fragmentShaderCode
            }),
          entryPoint: "main",
          targets:
            [
              {
                format: presentationFormat,
              },
            ],
        },
      primitive:
        {
          topology: "triangle-strip",
        },
    } );
}

function writeBufferData( buffer, data )
{
  const bufferData = new Float32Array( data );
  device.queue.writeBuffer( buffer, 0, bufferData.buffer, bufferData.byteOffset, bufferData.byteLength );
}

function encodePassAndSubmitCommandBuffer( renderPassDescriptor, pipeline, bindGroup )
{
  // Command encoder
  const commandEncoder = device.createCommandEncoder();

  // Encode pass
  const passEncoder = commandEncoder.beginRenderPass( renderPassDescriptor );
  passEncoder.setPipeline( pipeline );
  passEncoder.setBindGroup( 0, bindGroup );
  passEncoder.draw( 4 );
  passEncoder.end();

  // Submit command buffer
  device.queue.submit( [ commandEncoder.finish() ] );
}

function render( time )
{
  if( audioContext === undefined && start === undefined )
  {
    start = time;
  }

  renderPassDescriptor.colorAttachments[ 0 ].view = context.getCurrentTexture().createView();
  writeBufferData( uniformBuffer, [ CANVAS_WIDTH, CANVAS_HEIGHT, AUDIO ? audioContext.currentTime * 1000.0 : ( time - start ), 0.0 ] );
  encodePassAndSubmitCommandBuffer( renderPassDescriptor, pipeline, uniformBindGroup );

  requestAnimationFrame( render );
}

function setupShaderReload( url, reloadData, timeout )
{
  setInterval( async function()
    {
      const response = await fetch( url );
      const data = await response.text();

      if( data !== reloadData )
      {
        pipeline = await createPipeline( vertexShader, data, presentationFormat, uniformBindGroupLayout );

        reloadData = data;

        console.log( "Reloaded " + url );
      }

    }, timeout );
}

async function main()
{
  if( !navigator.gpu )
  {
    throw new Error( "WebGPU is not supported on this browser." );
  }

  const gpuAdapter = await navigator.gpu.requestAdapter();
  if( !gpuAdapter )
  {
    throw new Error( "Can not use WebGPU. No GPU adapter available." );
  }

  device = await gpuAdapter.requestDevice();
  if( !device )
  {
    throw new Error( "Failed to request logical device." );
  }

  // Create uniform buffer
  uniformBuffer = device.createBuffer(
    {
      size: 4 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    } );

  // Create bind group layout for uniform buffer visible in fragment shader
  uniformBindGroupLayout = device.createBindGroupLayout(
    {
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer:
          {
            type: "uniform",
          },
        },
      ],
    } );

  // Create bind group for uniform buffer based on above layout
  uniformBindGroup = device.createBindGroup(
    {
      layout: uniformBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource:
          {
            buffer : uniformBuffer,
          },
        },
      ],
    } );

  if( AUDIO )
  {
    audioContext = new AudioContext();

    // Create audio buffer
    let webAudioBuffer = audioContext.createBuffer( 2, AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT, audioContext.sampleRate );
    console.log( "Max audio length: " + ( webAudioBuffer.length / audioContext.sampleRate / 60 ).toFixed( 2 ) + " minutes" );  

    // Create texture target for audio
    let audioTexture = device.createTexture(
      {
        size: [ AUDIO_BUFFER_WIDTH, AUDIO_BUFFER_HEIGHT ],
        format: "rg32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      } );

    // Setup pipeline to render audio to texture
    renderPassDescriptor = createRenderPassDescriptor( audioTexture.createView() );
    pipeline = await createPipeline( vertexShader, audioShader, "rg32float", uniformBindGroupLayout );

    // Write to uniform buffer
    writeBufferData( uniformBuffer, [ AUDIO_BUFFER_WIDTH, AUDIO_BUFFER_HEIGHT, audioContext.sampleRate, 0.0 ] );

    // Render audio
    encodePassAndSubmitCommandBuffer( renderPassDescriptor, pipeline, uniformBindGroup );

    // Create buffer where we can copy our audio texture to
    const audioBuffer = device.createBuffer(
      {
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        size: AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT * 8,
      } );
   
    // Copy texture to audio buffer
    const commandEncoder = device.createCommandEncoder();

    commandEncoder.copyTextureToBuffer(
      {
        texture: audioTexture,
      },
      {
        buffer: audioBuffer,
        bytesPerRow: AUDIO_BUFFER_WIDTH * 2 * 4, // RG32FLOAT
      },
      [ AUDIO_BUFFER_WIDTH, AUDIO_BUFFER_HEIGHT ]
    );
    
    device.queue.submit( [ commandEncoder.finish() ] );
   
    // Map audio buffer for CPU to read
    await audioBuffer.mapAsync( GPUMapMode.READ );
    const audioData = new Float32Array( audioBuffer.getMappedRange() );
    
    // Feed data to web audio
    const channel0 = webAudioBuffer.getChannelData( 0 );
    const channel1 = webAudioBuffer.getChannelData( 1 );
    for( let i = 0; i < AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT; i++ )
    {
      channel0[ i ] = audioData[ ( i << 1 ) + 0 ];
      channel1[ i ] = audioData[ ( i << 1 ) + 1 ];
    }

    // Release GPU buffer
    audioBuffer.unmap();

    // Prepare audio buffer source node and connect it to output device
    audioBufferSourceNode = audioContext.createBufferSource();
    audioBufferSourceNode.buffer = webAudioBuffer;
    audioBufferSourceNode.connect( audioContext.destination );
  }
  else
  {
    renderPassDescriptor = createRenderPassDescriptor( null );
  }

  // Setup canvas and configure WebGPU context
  setupCanvasAndContext();

  // Setup pipeline to render actual graphics
  pipeline = await createPipeline( vertexShader, videoShader, presentationFormat, uniformBindGroupLayout );

  // Event listener for click to full screen (if required) and render start
  document.querySelector( "button" ).addEventListener( "click", e =>
    {
      if( FULLSCREEN )
      {
        canvas.requestFullscreen();
      }
      else
      {
        canvas.style.width = CANVAS_WIDTH;
        canvas.style.height = CANVAS_HEIGHT;
        canvas.style.position = "absolute";
        canvas.style.left = 0;
        canvas.style.top = 0;
      }

      if( AUDIO )
      {
        audioBufferSourceNode.start();
      }
      
      requestAnimationFrame( render );
      
      if( SHADER_RELOAD )
      {
        setupShaderReload( videoShaderFile, reloadData, 1000 );
      }
    } );
}

main();
