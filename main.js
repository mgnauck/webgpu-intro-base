import {mat4, utils, vec3} from "https://wgpu-matrix.org/dist/0.x/wgpu-matrix.module.js";

const FULLSCREEN = false;
const AUDIO = false;

const ASPECT = 1.6;
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;

const AUDIO_WIDTH = 4096;
const AUDIO_HEIGHT = 4096;

const MOVE_VELOCITY = 0.1;
const LOOK_VELOCITY = 0.025;
const WHEEL_VELOCITY = 0.0025;

const BLIT_SHADER = `
struct Output {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> Output {

  let pos = array<vec2f, 4>(
    vec2f(-1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(1, -1));

  var output: Output;

  let h = vec2f(0.5, 0.5);

  output.position = vec4f(pos[vertexIndex], 0, 1);
  output.texCoord = pos[vertexIndex] * h + h;

  return output;
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;

@fragment
fn fragmentMain(@location(0) texCoord: vec2f) -> @location(0) vec4f {
  return textureLoad(
    inputTexture, 
    vec2u(texCoord * vec2f(${CANVAS_WIDTH}, ${CANVAS_HEIGHT})),
    0);
}
`;

// External for now
const AUDIO_SHADER = ``;
const CONTENT_SHADER = ``;

let audioContext;
let audioBufferSourceNode;

let device;

let contentTextureView;
let contentUniformBuffer;
let contentBindGroupLayout;
let contentBindGroup;
let contentPipeline;

let blitBindGroup;
let blitPassDescriptor;
let blitPipeline;

let canvas;
let context;

let viewMatrix;
let eye, dir;
let programmableValue;

let start;

function loadTextFile(url) {
  return fetch(url).then(response => response.text());
}

function setupCanvasAndContext() {
  document.body.innerHTML = "<button>CLICK<canvas style='width:0;cursor:none'>";
  canvas = document.querySelector("canvas");
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  let presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  if (presentationFormat !== "bgra8unorm") {
    throw new Error(`Expected canvas pixel format of bgra8unorm but
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

  if (preSubmitOperation != undefined) {
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

async function createRenderPipeline(shaderCode, bindGroupLayout) {
  let shaderModule = device.createShaderModule({code: shaderCode});
  return device.createRenderPipelineAsync({
    layout: (bindGroupLayout === undefined) ?
        "auto" :
        device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
    vertex: {module: shaderModule, entryPoint: "vertexMain"},
    fragment: {
      module: shaderModule,
      entryPoint: "fragmentMain",
      targets: [{format: "bgra8unorm"}]
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
      await createComputePipeline(
          await loadTextFile("audioShader.wgsl"), audioBindGroupLayout),
      audioBindGroup, Math.ceil(AUDIO_WIDTH / 8), Math.ceil(AUDIO_HEIGHT / 8),
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

  for (let i = 0; i < AUDIO_WIDTH * AUDIO_HEIGHT; i++) {
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
      // 4x4 modelview, canvas w/h, time, programmable value
      {size: (16 + 2 + 1 + 1) * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});

  contentBindGroupLayout = device.createBindGroupLayout({
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
      {binding: 0, resource: contentTextureView},
      {binding: 1, resource: {buffer: contentUniformBuffer}}
    ]
  });

  await setupContentPipeline();
}

async function setupContentPipeline() {
  contentPipeline = await createComputePipeline(
      await loadTextFile("contentShader.wgsl"), contentBindGroupLayout);
}

async function prepareBlitResources() {
  let blitBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {}},
    ]
  });

  blitBindGroup = device.createBindGroup({
    layout: blitBindGroupLayout,
    entries: [
      {binding: 0, resource: contentTextureView},
    ]
  });

  blitPassDescriptor = createRenderPassDescriptor(undefined);
  blitPipeline = await createRenderPipeline(BLIT_SHADER, blitBindGroupLayout);
}

function render(time) {
  if (audioContext === undefined && start === undefined) {
    start = time;
  }

  writeBufferData(
      contentUniformBuffer, new Float32Array([
        ...viewMatrix,
        CANVAS_WIDTH, CANVAS_HEIGHT,
        AUDIO ? audioContext.currentTime : ((time - start) / 1000.0),
        programmableValue
      ]));

  setupPerformanceTimer("Render");

  encodeComputePassAndSubmit(
      contentPipeline, contentBindGroup, Math.ceil(CANVAS_WIDTH / 8),
      Math.ceil(CANVAS_HEIGHT / 8));

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
        let frameTime = end - begin;
        document.title =
            `${(frameTime).toFixed(2)} / ${(1000.0 / frameTime).toFixed(2)}`;
        // console.log(`${timerName} (ms): ${(end - begin).toFixed(2)}`);
      })
      .catch(function(err) {
        console.log(err);
      });
}

function resetView() {
  eye = vec3.create(0, 0, 1);
  dir = vec3.create(0, 0, -1);
  programmableValue = 0.0;
}

function computeViewMatrix() {
  viewMatrix = mat4.lookAt(eye, vec3.add(eye, dir), vec3.create(0, 1, 0));
}

function handleKeyEvent(e) {
  switch (e.key) {
    case "a":
      vec3.add(
          eye, vec3.scale(mat4.getAxis(viewMatrix, 0), -MOVE_VELOCITY), eye);
      break;
    case "d":
      vec3.add(
          eye, vec3.scale(mat4.getAxis(viewMatrix, 0), MOVE_VELOCITY), eye);
      break;
    case "w":
      vec3.add(eye, vec3.scale(dir, MOVE_VELOCITY), eye);
      break;
    case "s":
      vec3.add(eye, vec3.scale(dir, -MOVE_VELOCITY), eye);
      break;
    case "r":
      resetView();
      break;
  };

  computeViewMatrix();
}

function handleMouseMoveEvent(e) {
  let yaw = -e.movementX * LOOK_VELOCITY;  // negation?
  let pitch = -e.movementY * LOOK_VELOCITY;

  // Disallow alignment of forward and global up vector
  const currentPitch = Math.acos(dir[1]);
  const newPitch = currentPitch - pitch;
  const minPitch = utils.degToRad(1.0);
  const maxPitch = utils.degToRad(179.0);

  if (newPitch < minPitch) {
    pitch = currentPitch - minPitch;
  }
  if (newPitch > maxPitch) {
    pitch = currentPitch - maxPitch;
  }

  // Pitch locally, yaw globally to avoid unwanted roll
  vec3.transformMat4(
      dir, mat4.rotation(mat4.getAxis(viewMatrix, 0), pitch), dir);
  vec3.transformMat4(dir, mat4.rotationY(yaw), dir);

  computeViewMatrix();
}

function handleMouseWheelEvent(e) {
  programmableValue -= e.deltaY * WHEEL_VELOCITY;
  console.log("value:" + programmableValue);
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

  resetView();
  computeViewMatrix();

  document.querySelector("button").removeEventListener("click", startRender);

  canvas.addEventListener("click", async () => {
    if (!document.pointerLockElement) {
      await canvas.requestPointerLock({unadjustedMovement: true});
    }
  });

  document.addEventListener("pointerlockchange", () => {
    if (document.pointerLockElement === canvas) {
      document.addEventListener("keydown", handleKeyEvent);
      canvas.addEventListener("mousemove", handleMouseMoveEvent);
      canvas.addEventListener("wheel", handleMouseWheelEvent);
    } else {
      document.removeEventListener("keydown", handleKeyEvent);
      canvas.removeEventListener("mousemove", handleMouseMoveEvent);
      canvas.removeEventListener("wheel", handleMouseWheelEvent);
    }
  });

  if (AUDIO) {
    audioBufferSourceNode.start();
  }

  requestAnimationFrame(render);

  // Periodically reload content shader
  setInterval(setupContentPipeline, 500);
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

  await prepareContentResources();
  await prepareBlitResources();

  setupCanvasAndContext();

  if (AUDIO) {
    document.querySelector("button").addEventListener("click", startRender);
  } else {
    startRender();
  }
}

main();
