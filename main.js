'use strict';

const FULLSCREEN = false;
const AUDIO = true;
const SHADER_RELOAD = true;

const ASPECT = 1.6;
const CANVAS_WIDTH = 400 * ASPECT;
const CANVAS_HEIGHT = 400;

const AUDIO_BUFFER_WIDTH = 1024;
const AUDIO_BUFFER_HEIGHT = 1024;

const audioShader = `
@binding(0) @group(0) var<storage, read_write> outputBuffer: array<vec2<f32>>;
// TODO "uniform" for sample rate

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= ${AUDIO_BUFFER_WIDTH} * ${AUDIO_BUFFER_HEIGHT}) {
    return;
  }

  let time: f32 = f32(global_id.x) / 44100.0;
  let val: f32 = sin(time * 440.0 * 1.2 * 3.1415);
  outputBuffer[global_id.x] = vec2<f32>(val, val);
}
`;

const vertexShader = `
@vertex
fn main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32>
{  
  var pos = array<vec2<f32>, 4>(vec2( -1.0, 1.0 ), vec2( -1.0, -1.0 ), vec2( 1.0, 1.0 ), vec2( 1.0, -1.0 ));
  return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}
`;

const videoShaderFile = 'http://localhost:8000/fragmentShader.wgsl';
const videoShader = `
struct Uniforms {
  resolution: vec2<f32>,
  time: f32,
}
@binding(0) @group(0) var<uniform> uniforms: Uniforms;

@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32>
{
  return vec4<f32>(0.6, 0.3, 0.3, 1.0);
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

let computeBuffer;
let computeBindGroupLayout;
let computeBindGroup;
let computePipeline;

let canvas;
let context;
let presentationFormat;

let start;
let reloadData;

function setupCanvasAndContext() {
  // Canvas
  document.body.innerHTML = '<button>CLICK<canvas style=\'width:0;cursor:none\'>';
  canvas = document.querySelector('canvas');
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  // Expecting format bgra8unorm
  presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  if (presentationFormat !== 'bgra8unorm') {
    throw new Error(`Expected canvas pixel format of 'bgra8unorm' but was '${presentationFormat}'.`)
  }

  // Context
  context = canvas.getContext('webgpu');

  context.configure({device : device, format : presentationFormat, alphaMode : 'opaque'});
}

function createRenderPassDescriptor(view) {
  return {
    colorAttachments : [
      {view, clearValue : {r : 0.3, g : 0.3, b : 0.3, a : 1.0}, loadOp : 'clear', storeOp : 'store'}
    ]
  };
}

function createPipeline(vertexShaderCode, fragmentShaderCode, presentationFormat, bindGroupLayout) {
  return device.createRenderPipelineAsync({
    layout : (bindGroupLayout === undefined)
                 ? 'auto'
                 : device.createPipelineLayout({bindGroupLayouts : [ bindGroupLayout ]}),
    vertex : {module : device.createShaderModule({code : vertexShaderCode}), entryPoint : 'main'},
    fragment : {
      module : device.createShaderModule({code : fragmentShaderCode}),
      entryPoint : 'main',
      targets : [ {format : presentationFormat} ],
    },
    primitive : {
      topology : 'triangle-strip',
    },
  });
}

function createComputePipeline(computeShaderCode, bindGroupLayout) {
  return device.createComputePipelineAsync({
    layout : (bindGroupLayout === undefined)
                 ? 'auto'
                 : device.createPipelineLayout({bindGroupLayouts : [ bindGroupLayout ]}),
    compute : {
      module : device.createShaderModule({code : computeShaderCode}),
      entryPoint : 'main',
    },
  });
}

function writeBufferData(buffer, data) {
  const bufferData = new Float32Array(data);
  device.queue.writeBuffer(buffer, 0, bufferData.buffer, bufferData.byteOffset,
                           bufferData.byteLength);
}

function encodePassAndSubmitCommandBuffer(renderPassDescriptor, pipeline, bindGroup) {
  // Command encoder
  const commandEncoder = device.createCommandEncoder();

  // Encode pass
  const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.draw(4);
  passEncoder.end();

  // Submit command buffer
  device.queue.submit([ commandEncoder.finish() ]);
}

function render(time) {
  if (audioContext === undefined && start === undefined) {
    start = time;
  }

  renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
  writeBufferData(uniformBuffer, [
    CANVAS_WIDTH, CANVAS_HEIGHT, AUDIO ? audioContext.currentTime * 1000.0 : (time - start), 0.0
  ]);
  encodePassAndSubmitCommandBuffer(renderPassDescriptor, pipeline, uniformBindGroup);

  requestAnimationFrame(render);
}

function setupShaderReload(url, reloadData, timeout) {
  setInterval(async function() {
    const response = await fetch(url);
    const data = await response.text();

    if (data !== reloadData) {
      pipeline =
          await createPipeline(vertexShader, data, presentationFormat, uniformBindGroupLayout);

      reloadData = data;

      console.log('Reloaded ' + url);
    }
  }, timeout);
}

async function main() {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported on this browser.');
  }

  const gpuAdapter = await navigator.gpu.requestAdapter();
  if (!gpuAdapter) {
    throw new Error('Can not use WebGPU. No GPU adapter available.');
  }

  device = await gpuAdapter.requestDevice();
  if (!device) {
    throw new Error('Failed to request logical device.');
  }

  if (AUDIO) {

    // WebAudio
    audioContext = new AudioContext();

    // Create web audio buffer
    let webAudioBuffer = audioContext.createBuffer(2, AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT,
                                                   audioContext.sampleRate);
    console.log('Max audio length: ' +
                (webAudioBuffer.length / audioContext.sampleRate / 60).toFixed(2) + ' minutes');

    // Create buffer for the audio compute shader to write to
    computeBuffer = device.createBuffer({
      size : AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT * 2 * 4,
      usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    computeBindGroupLayout = device.createBindGroupLayout({
      entries : [ {
        binding : 0,
        visibility : GPUShaderStage.COMPUTE,
        buffer : {type : 'storage'}, // read-only-storage??
      } ]
    });

    computeBindGroup = device.createBindGroup({
      layout : computeBindGroupLayout,
      entries : [ {
        binding : 0,
        resource : {buffer : computeBuffer},
      } ]
    });

    // Create compute pipeline for audio
    computePipeline = await createComputePipeline(audioShader, computeBindGroupLayout);

    // Command encoder
    let commandEncoder = device.createCommandEncoder();

    // Encode pass
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, computeBindGroup);
    passEncoder.dispatchWorkgroups(AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT / 64);
    passEncoder.end();

    // Submit command buffer
    device.queue.submit([ commandEncoder.finish() ]);

    // Create buffer where we can copy our audio texture to
    const readBuffer = device.createBuffer({
      usage : GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      size : AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT * 2 * 4,
    });

    // Copy texture to audio buffer
    commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(computeBuffer, 0, readBuffer, 0,
                                      AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT * 2 * 4);
    device.queue.submit([ commandEncoder.finish() ]);

    // Map audio buffer for CPU to read
    await readBuffer.mapAsync(GPUMapMode.READ);
    const audioData = new Float32Array(readBuffer.getMappedRange());

    // Feed data to web audio
    const channel0 = webAudioBuffer.getChannelData(0);
    const channel1 = webAudioBuffer.getChannelData(1);
    for (let i = 0; i < AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT; i++) {
      channel0[i] = audioData[(i << 1) + 0];
      channel1[i] = audioData[(i << 1) + 1];
    }

    // Release GPU buffer
    readBuffer.unmap();

    // Prepare audio buffer source node and connect it to output device
    audioBufferSourceNode = audioContext.createBufferSource();
    audioBufferSourceNode.buffer = webAudioBuffer;
    audioBufferSourceNode.connect(audioContext.destination);
  }

  // Setup canvas and configure WebGPU context
  setupCanvasAndContext();

  // Create uniform buffer
  uniformBuffer = device.createBuffer({
    size : 4 * 4,
    usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create bind group layout for uniform buffer visible in fragment shader
  uniformBindGroupLayout = device.createBindGroupLayout({
    entries : [ {
      binding : 0,
      visibility : GPUShaderStage.FRAGMENT,
      buffer : {type : 'uniform'},
    } ]
  });

  // Create bind group for uniform buffer based on above layout
  uniformBindGroup = device.createBindGroup({
    layout : uniformBindGroupLayout,
    entries : [ {
      binding : 0,
      resource : {buffer : uniformBuffer},
    } ],
  });

  // Setup pipeline to render actual graphics
  renderPassDescriptor = createRenderPassDescriptor(undefined);
  pipeline =
      await createPipeline(vertexShader, videoShader, presentationFormat, uniformBindGroupLayout);

  // Event listener for click to full screen (if required) and render start
  document.querySelector('button').addEventListener('click', e => {
    if (FULLSCREEN) {
      canvas.requestFullscreen();
    } else {
      canvas.style.width = CANVAS_WIDTH;
      canvas.style.height = CANVAS_HEIGHT;
      canvas.style.position = 'absolute';
      canvas.style.left = 0;
      canvas.style.top = 0;
    }

    if (AUDIO) {
      audioBufferSourceNode.start();
    }

    requestAnimationFrame(render);

    if (SHADER_RELOAD) {
      setupShaderReload(videoShaderFile, reloadData, 1000);
    }
  });
}

main();
