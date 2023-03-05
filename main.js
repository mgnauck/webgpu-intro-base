'use strict';

const FULLSCREEN = false;
const AUDIO = true;
const SHADER_RELOAD = false;

const ASPECT = 1.6;
const CANVAS_WIDTH = 1920;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;

const AUDIO_BUFFER_WIDTH = 4096;
const AUDIO_BUFFER_HEIGHT = 4096;

const audioShader = `
@group(0) @binding(0) var<storage, read_write> outputBuffer: array<vec2<f32>>;

const BPM: f32 = 160.0;
const PI: f32 = 3.141592654;
const TAU: f32 = 6.283185307;

fn time_to_beat(t: f32) -> f32 {
  return t / 60.0 * BPM;
}

fn beat_to_time(b: f32) -> f32 {
  return b / BPM * 60.0;
}

fn sine(phase: f32) -> f32 {
  return sin(TAU * phase);
}

fn rand(co: vec2<f32>) -> f32 {
  return fract(sin(dot(co, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn noise(phase: f32) -> vec4<f32> {
  let uv: vec2<f32> = phase / vec2<f32>(0.512, 0.487);
  return vec4<f32>(rand(uv));
}

fn kick(time: f32) -> f32 {
  let amp: f32 = exp(-5.0 * time);
  let phase: f32 = 120.0 * time - 15.0 * exp(-60.0 * time);
  return amp * sine(phase);
}

fn hi_hat(time: f32) -> vec2<f32> {
  let amp: f32 = exp(-40.0 * time);
  return amp * noise(time * 110.0).xy;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= ${AUDIO_BUFFER_WIDTH} || global_id.y >= ${AUDIO_BUFFER_HEIGHT}) {
    return;
  }

  let idx: u32 = ${AUDIO_BUFFER_WIDTH} * global_id.y + global_id.x;
  let time: f32 = f32(idx) / 44100.0;

  //let val: f32 = sin(time * 440.0 * 1.2 * 3.1415);
  //outputBuffer[idx] = vec2<f32>(val, val);

  let beat: f32 = time_to_beat(time);

  // Kick
  var res = vec2<f32>(0.6 * kick(beat_to_time(beat % 1.0)));

  // Hihat
  res += 0.3 * hi_hat(beat_to_time((beat + 0.5) % 1.0));

  outputBuffer[idx] = vec2<f32>(clamp(res, vec2<f32>(-1.0), vec2<f32>(1.0)));
  //*/
}
`;

const vertexShader = `
@vertex
fn main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 4>(vec2(-1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(1.0, -1.0));
  return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}
`;

const videoShaderFile = 'http://localhost:8000/fragmentShader.wgsl';
const videoShader = `
struct Uniforms {
  resolution: vec2<f32>,
  time: f32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
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

  context.configure({
    device: device,
    format: presentationFormat,
    alphaMode: 'opaque',
  });
}

function createRenderPassDescriptor(view) {
  return {colorAttachments: [{view, clearValue: {r: 0.3, g: 0.3, b: 0.3, a: 1.0}, loadOp: 'clear', storeOp: 'store'}]};
}

function createPipeline(vertexShaderCode, fragmentShaderCode, presentationFormat, bindGroupLayout) {
  return device.createRenderPipelineAsync({
    layout: (bindGroupLayout === undefined) ? 'auto' :
                                              device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
    vertex: {
      module: device.createShaderModule({code: vertexShaderCode}),
      entryPoint: 'main',
    },
    fragment: {
      module: device.createShaderModule({code: fragmentShaderCode}),
      entryPoint: 'main',
      targets: [{format: presentationFormat}],
    },
    primitive: {
      topology: 'triangle-strip',
    }
  });
}

function createComputePipeline(shaderCode, bindGroupLayout) {
  return device.createComputePipelineAsync({
    layout: (bindGroupLayout === undefined) ? 'auto' :
                                              device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
    compute: {
      module: device.createShaderModule({code: shaderCode}),
      entryPoint: 'main',
    },
  });
}

function writeBufferData(buffer, data) {
  const bufferData = new Float32Array(data);
  device.queue.writeBuffer(buffer, 0, bufferData.buffer, bufferData.byteOffset, bufferData.byteLength);
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
  device.queue.submit([commandEncoder.finish()]);
}

function setupPerformanceTimer(timerName) {
  let begin = performance.now();

  device.queue.onSubmittedWorkDone()
      .then(() => {
        let end = performance.now();
        console.log(`${timerName} (ms): ${(end - begin).toFixed(2)}`);
      })
      .catch((err) => {
        console.log(err);
      });
}

function render(time) {
  if (audioContext === undefined && start === undefined) {
    start = time;
  }

  renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
  writeBufferData(
      uniformBuffer, [CANVAS_WIDTH, CANVAS_HEIGHT, AUDIO ? audioContext.currentTime * 1000.0 : (time - start), 0.0]);
  // setupPerformanceTimer('Render frame');
  encodePassAndSubmitCommandBuffer(renderPassDescriptor, pipeline, uniformBindGroup);

  requestAnimationFrame(render);
}

function setupShaderReload(url, reloadData, timeout) {
  setInterval(async function() {
    const response = await fetch(url);
    const data = await response.text();

    if (data !== reloadData) {
      pipeline = await createPipeline(vertexShader, data, presentationFormat, uniformBindGroupLayout);

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
    audioContext = new AudioContext();

    let audioBuffer = audioContext.createBuffer(2, AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT, audioContext.sampleRate);
    console.log('Max audio length: ' + (audioBuffer.length / audioContext.sampleRate / 60).toFixed(2) + ' minutes');

    // Create buffer for the audio compute shader to write to
    computeBuffer = device.createBuffer({
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      size: AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT * 2 * 4,
    });

    computeBindGroupLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: 'storage'},
      }]
    });

    computeBindGroup = device.createBindGroup({
      layout: computeBindGroupLayout,
      entries: [{
        binding: 0,
        resource: {buffer: computeBuffer},
      }]
    });

    // Create buffer where we can copy to and read it on the CPU
    const readBuffer = device.createBuffer({
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      size: AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT * 2 * 4,
    });

    // Create compute pipeline for audio
    computePipeline = await createComputePipeline(audioShader, computeBindGroupLayout);

    // Command encoder
    let commandEncoder = device.createCommandEncoder();

    // Encode pass
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, computeBindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(AUDIO_BUFFER_WIDTH / 8), Math.ceil(AUDIO_BUFFER_HEIGHT / 8));
    passEncoder.end();

    // Copy computed buffer to read buffer
    commandEncoder.copyBufferToBuffer(
        computeBuffer, 0, readBuffer, 0, AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT * 2 * 4);

    setupPerformanceTimer('Render audio');

    // Submit command buffer
    device.queue.submit([commandEncoder.finish()]);

    // Map buffer for CPU to read
    await readBuffer.mapAsync(GPUMapMode.READ);
    const audioData = new Float32Array(readBuffer.getMappedRange());

    // Feed data to web audio
    const channel0 = audioBuffer.getChannelData(0);
    const channel1 = audioBuffer.getChannelData(1);
    for (let i = 0; i < AUDIO_BUFFER_WIDTH * AUDIO_BUFFER_HEIGHT; i++) {
      channel0[i] = audioData[(i << 1) + 0];
      channel1[i] = audioData[(i << 1) + 1];
    }

    // Release read buffer
    readBuffer.unmap();

    // Prepare audio buffer source node and connect it to output device
    audioBufferSourceNode = audioContext.createBufferSource();
    audioBufferSourceNode.buffer = audioBuffer;
    audioBufferSourceNode.connect(audioContext.destination);
  }

  // Setup canvas and configure WebGPU context
  setupCanvasAndContext();

  // Create uniform buffer
  uniformBuffer = device.createBuffer({
    size: 4 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create bind group layout for uniform buffer visible in fragment shader
  uniformBindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.FRAGMENT,
      buffer: {type: 'uniform'},
    }]
  });

  // Create bind group for uniform buffer based on above layout
  uniformBindGroup = device.createBindGroup({
    layout: uniformBindGroupLayout,
    entries: [{
      binding: 0,
      resource: {buffer: uniformBuffer},
    }],
  });

  // Setup pipeline to render actual graphics
  renderPassDescriptor = createRenderPassDescriptor(undefined);
  pipeline = await createPipeline(vertexShader, videoShader, presentationFormat, uniformBindGroupLayout);

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
