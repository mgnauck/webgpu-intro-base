const FULLSCREEN = false;
const AUDIO = true;

const ASPECT = 1.6;
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;

const AUDIO_WIDTH = 4096;
const AUDIO_HEIGHT = 4096;

const audioShader = `
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

@group(0) @binding(0) var outputTexture: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  
  if (global_id.x >= ${AUDIO_WIDTH} || global_id.y >= ${AUDIO_HEIGHT}) {
    return;
  }

  let time: f32 = f32(${AUDIO_WIDTH} * global_id.y + global_id.x) / 44100.0;
  let beat: f32 = time_to_beat(time);

  // Kick
  var res = vec2<f32>(0.6 * kick(beat_to_time(beat % 1.0)));

  // Hihat
  res += 0.3 * hi_hat(beat_to_time((beat + 0.5) % 1.0));

  textureStore(
    outputTexture,
    vec2<u32>(global_id.x, global_id.y),
    vec4<f32>(clamp(res, vec2<f32>(-1.0), vec2<f32>(1.0)), 0.0, 1.0));
}
`;

const contentShader = `
struct Uniforms {
  resolution: vec2<f32>,
  time: f32,
}

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

  if (global_id.x >= ${CANVAS_WIDTH} || global_id.y >= ${CANVAS_HEIGHT}) {
    return;
  }

  var uv = vec2<f32>(f32(global_id.x) / ${CANVAS_WIDTH}.0, f32(global_id.y) / ${CANVAS_HEIGHT}.0);
  var col = vec3<f32>(0.5 + 0.5 * cos(uniforms.time + uv.xyx + vec3<f32>(0.0, 2.0, 4.0)));

  textureStore(
    outputTexture,
    vec2<u32>(global_id.x, global_id.y),
    vec4<f32>(col, 1.0));
}
`;

const vertexShader = `
struct Output {
  @builtin(position) position: vec4<f32>,
  @location(0) texCoord: vec2<f32>
}

@vertex
fn main(@builtin(vertex_index) vertex_index: u32) -> Output {

  var pos = array<vec2<f32>, 4>(
    vec2(-1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(1.0, -1.0));
  
  var output: Output;

  output.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
  output.texCoord = pos[vertex_index] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);

  return output;
}
`;

const blitShader = `
@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var blitSampler: sampler;

@fragment
fn main(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
  return textureSample(inputTexture, blitSampler, texCoord);
}
`;

let audioContext;
let audioBufferSourceNode;

let device;

let contentTextureView;
let contentUniformBuffer;
let contentBindGroup;
let contentPipeline;

let blitBindGroup;
let blitPassDescriptor;
let blitPipeline;

let canvas;
let context;
let presentationFormat;

let start;

function setupCanvasAndContext() {
  
  document.body.innerHTML = "<button>CLICK<canvas style='width:0;cursor:none'>";
  canvas = document.querySelector("canvas");
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  if (presentationFormat !== "bgra8unorm") {
    throw new Error(`Expected canvas pixel format of 'bgra8unorm' but
       was '${presentationFormat}'.`);
  }

  context = canvas.getContext("webgpu");

  context.configure({device, format: presentationFormat, alphaMode: "opaque"});
}

function writeBufferData(buffer, data) {
 
  const bufferData = new Float32Array(data);
  
  device.queue.writeBuffer(
      buffer, 0, bufferData.buffer, bufferData.byteOffset,
      bufferData.byteLength);
}

async function createComputePipeline(shaderCode, bindGroupLayout) {

  return device.createComputePipelineAsync({
    layout: (bindGroupLayout === undefined) ?
        "auto" :
        device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
    compute: {
      module: device.createShaderModule({code: shaderCode}),
      entryPoint: "main"
    }
  });
}

function encodeComputePassAndSubmit(
  pipeline, bindGroup, workgroupCountX, workgroupCountY, preSubmitOperation) {
  
  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
  passEncoder.end();

  if(preSubmitOperation != undefined) {
    preSubmitOperation(commandEncoder);
  }

  device.queue.submit([commandEncoder.finish()]);
}

function createRenderPassDescriptor(view) {
  return {
    colorAttachments: [{
      view,
      clearValue: {r: 1.0, g: 0.0, b: 0.0, a: 1.0},
      loadOp: "clear",
      storeOp: "store"
    }]
  };
}

async function createRenderPipeline(
    vertexShaderCode, fragmentShaderCode, presentationFormat, bindGroupLayout) {

  return device.createRenderPipelineAsync({
    layout: (bindGroupLayout === undefined) ?
        "auto" :
        device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
    vertex: {
      module: device.createShaderModule({code: vertexShaderCode}),
      entryPoint: "main"
    },
    fragment: {
      module: device.createShaderModule({code: fragmentShaderCode}),
      entryPoint: "main",
      targets: [{format: presentationFormat}]
    },
    primitive: {topology: "triangle-strip"}
  });
}

function encodeBlitPassAndSubmit(passDescriptor, pipeline, bindGroup) {
  
  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginRenderPass(passDescriptor);
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.draw(4);
  passEncoder.end();

  device.queue.submit([commandEncoder.finish()]);
}

function render(time) {

  if (audioContext === undefined && start === undefined) {
    start = time;
  }

  writeBufferData(contentUniformBuffer, [
    CANVAS_WIDTH, CANVAS_HEIGHT,
    AUDIO ? audioContext.currentTime : ((time - start) / 1000.0), 0.0
  ]);

  encodeComputePassAndSubmit(
    contentPipeline, contentBindGroup, 
    Math.ceil(CANVAS_WIDTH / 8), Math.ceil(CANVAS_HEIGHT / 8));

  blitPassDescriptor.colorAttachments[0].view =
      context.getCurrentTexture().createView();
  encodeBlitPassAndSubmit(blitPassDescriptor, blitPipeline, blitBindGroup);

  requestAnimationFrame(render);
}

function setupPerformanceTimer(timerName) {
  
  let begin = performance.now();
  
  device.queue.onSubmittedWorkDone()
      .then(function() {
        let end = performance.now();
        console.log(`${timerName} (ms): ${(end - begin).toFixed(2)}`);
      })
      .catch(function(err) {
        console.log(err);
      });
}

async function renderAudio() {
 
  audioContext = new AudioContext();

  let audioBuffer = audioContext.createBuffer(
      2, AUDIO_WIDTH * AUDIO_HEIGHT, audioContext.sampleRate);
  console.log(
      "Max audio length: " +
      (audioBuffer.length / audioContext.sampleRate / 60).toFixed(2) + " min");

   let audioTexture = device.createTexture({
    format: "rg32float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    size: [AUDIO_WIDTH, AUDIO_HEIGHT]
  });

  let audioBindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: {format: "rg32float"}
    }]
  });

  let audioBindGroup = device.createBindGroup({
    layout: audioBindGroupLayout,
    entries: [{binding: 0, resource: audioTexture.createView()}]
  });

  const readBuffer = device.createBuffer({
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    size: AUDIO_WIDTH * AUDIO_HEIGHT * 2 * 4
  });

  setupPerformanceTimer("Render audio");

  encodeComputePassAndSubmit(
    await createComputePipeline(audioShader, audioBindGroupLayout),
    audioBindGroup,
    Math.ceil(AUDIO_WIDTH / 8), Math.ceil(AUDIO_HEIGHT / 8),
    function(commandEncoder) {
      commandEncoder.copyTextureToBuffer(
        {texture: audioTexture},
        {buffer: readBuffer, bytesPerRow: AUDIO_WIDTH * 2 * 4},
        [AUDIO_WIDTH, AUDIO_HEIGHT]);  
    });

  await readBuffer.mapAsync(GPUMapMode.READ);
  const audioData = new Float32Array(readBuffer.getMappedRange());

  const channel0 = audioBuffer.getChannelData(0);
  const channel1 = audioBuffer.getChannelData(1);

  let i;
  for (i = 0; i < AUDIO_WIDTH * AUDIO_HEIGHT; i += 1) {
    channel0[i] = audioData[(i << 1) + 0];
    channel1[i] = audioData[(i << 1) + 1];
  }

  readBuffer.unmap();

  audioBufferSourceNode = audioContext.createBufferSource();
  audioBufferSourceNode.buffer = audioBuffer;
  audioBufferSourceNode.connect(audioContext.destination);
}

async function prepareContentResources() {

  let contentTexture = device.createTexture({
    format: "rgba8unorm",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    size: [CANVAS_WIDTH, CANVAS_HEIGHT]
  });

  contentTextureView = contentTexture.createView();

  contentUniformBuffer = device.createBuffer(
      {size: 4 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});

  const contentBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {format: "rgba8unorm"}
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: "uniform"}
      }
    ]
  });

  contentBindGroup = device.createBindGroup({
    layout: contentBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: contentTextureView
      },
      {
        binding: 1,
        resource: {buffer: contentUniformBuffer}
      }
    ]
  });
  
  contentPipeline = await createComputePipeline(
    contentShader, contentBindGroupLayout);
}

async function prepareBlitResources() {

  let blitSampler = device.createSampler({
    magFilter: "linear"
  });

  let blitBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {}
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {}
      }
    ]
  });

  blitBindGroup = device.createBindGroup({
    layout: blitBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: contentTextureView
      },
      {
        binding: 1,
        resource: blitSampler
      }
    ]
  });

  blitPassDescriptor = createRenderPassDescriptor(undefined);
  blitPipeline = await createRenderPipeline(
      vertexShader, blitShader, presentationFormat, blitBindGroupLayout);
}

function startRender() {

  if (FULLSCREEN) {
    canvas.requestFullscreen();
  } else {
    canvas.style.width = CANVAS_WIDTH;
    canvas.style.height = CANVAS_HEIGHT;
    canvas.style.position = "absolute";
    canvas.style.left = 0;
    canvas.style.top = 0;
  }

  if (AUDIO) {
    audioBufferSourceNode.start();
  }

  requestAnimationFrame(render);
}

async function main() {

  if (!navigator.gpu) {
    throw new Error("WebGPU is not supported on this browser.");
  }

  const gpuAdapter = await navigator.gpu.requestAdapter();
  if (!gpuAdapter) {
    throw new Error("Can not use WebGPU. No GPU adapter available.");
  }

  device = await gpuAdapter.requestDevice();
  if (!device) {
    throw new Error("Failed to request logical device.");
  }

  if (AUDIO) {
    await renderAudio();
  }
  
  setupCanvasAndContext(); 
 
  await prepareContentResources();
  await prepareBlitResources();
  
  if(AUDIO) {
    document.querySelector("button").addEventListener("click", startRender);
  } else {
    startRender();
  }
}

main();
