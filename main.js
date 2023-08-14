import {mat4, utils, vec3} from "https://wgpu-matrix.org/dist/0.x/wgpu-matrix.module.js";

const FULLSCREEN = false;
const AUDIO = false;

const ASPECT = 1.6;
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;

const AUDIO_WIDTH = 4096;
const AUDIO_HEIGHT = 4096;

const GRID_RES = 128.0;

const MOVE_VELOCITY = 0.5;
const LOOK_VELOCITY = 0.025;
const WHEEL_VELOCITY = 0.0025;

let audioContext;
let audioBufferSourceNode;

let device;

const gridBuffer = [];

let computeStorageBuffer;
let computeBindGroupLayout;
const computeBindGroup = [];
let computePipeline;

let renderUniformBuffer;
let renderBindGroupLayout;
const renderBindGroup = [];
let renderPipeline;
let renderPassDescriptor;

let canvas;
let context;

let viewMatrix;
let eye, dir;
let programmableValue;
let index = 0;

let start, lastUpdate;

function loadTextFile(url)
{
  return fetch(url).then(response => response.text());
}

async function createComputePipeline(shaderCode, bindGroupLayout)
{
  return device.createComputePipelineAsync({
    layout: (bindGroupLayout === undefined) ?
        "auto" :
        device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
    compute: {
      module: device.createShaderModule({code: shaderCode}),
      entryPoint: "m"
    }
  });
}

async function createRenderPipeline(shaderCode, bindGroupLayout)
{
  let shaderModule = device.createShaderModule({code: shaderCode});
  return device.createRenderPipelineAsync({
    layout: (bindGroupLayout === undefined) ?
        "auto" :
        device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
    vertex: {
      module: shaderModule,
      entryPoint: "v"
    },
    fragment: {
      module: shaderModule,
      entryPoint: "f",
      targets: [{format: "bgra8unorm"}]
    },
    primitive: {topology: "triangle-strip"}
  });
}

function encodeComputePassAndSubmit(pipeline, bindGroup, workgroupCountX, workgroupCountY, workgroupCountZ, preSubmitOperation)
{
  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
  passEncoder.end();

  if (preSubmitOperation != undefined) {
    preSubmitOperation(commandEncoder);
  }

  device.queue.submit([commandEncoder.finish()]);
}

function encodeRenderPassAndSubmit(passDescriptor, pipeline, bindGroup)
{
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginRenderPass(passDescriptor);
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.draw(4);
  passEncoder.end();

  device.queue.submit([commandEncoder.finish()]);
}

async function prepareAudio()
{
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
      await createComputePipeline(await loadTextFile("audioShader.wgsl"), audioBindGroupLayout),
      audioBindGroup, Math.ceil(AUDIO_WIDTH / 8), Math.ceil(AUDIO_HEIGHT / 8), 1,
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

async function createGPUResources()
{
  computeBindGroupLayout = device.createBindGroupLayout({
    entries: [ 
      {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
      {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
    ]
  });

  renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}},
      {binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}}
    ]
  }); 
  
  // min + max cell index in xyz (= 2x4 for alignment), gridRes
  computeStorageBuffer = device.createBuffer({
    size: (4 + 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
 
  // 4x4 modelview, grid res, time, 2x programmable value
  renderUniformBuffer = device.createBuffer({
    size: (16 + 1 + 1 + 2) * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});

  for(let i=0; i<2; i++) {
    gridBuffer[i] = device.createBuffer({
      size: GRID_RES * GRID_RES * GRID_RES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  }

  for(let i=0; i<2; i++) {
    computeBindGroup[i] = device.createBindGroup({
      layout: computeBindGroupLayout,
      entries: [
        {binding: 0, resource: {buffer: computeStorageBuffer}},
        {binding: 1, resource: {buffer: gridBuffer[i]}},
        {binding: 2, resource: {buffer: gridBuffer[1 - i]}}
      ]
    });

    renderBindGroup[i] = device.createBindGroup({
      layout: renderBindGroupLayout,
      entries: [
        {binding: 0, resource: {buffer: renderUniformBuffer}},
        {binding: 1, resource: {buffer: gridBuffer[1 - i]}}
      ]
    });
  }

  renderPassDescriptor = {
    colorAttachments: [{
      undefined, // view
      clearValue: {r: 1.0, g: 0.0, b: 0.0, a: 1.0},
      loadOp: "clear",
      storeOp: "store"
    }]
  };
}

async function createPipelines()
{
  let shader = await loadTextFile("contentShader.wgsl");

  computePipeline = await createComputePipeline(shader, computeBindGroupLayout);
  renderPipeline = await createRenderPipeline(shader, renderBindGroupLayout);
}

function render(time)
{
  if (audioContext === undefined && start === undefined) {
    start = time;
    lastUpdate = time;
  }

  if(time - lastUpdate > 2500) {
    let workgroupSize = Math.ceil(GRID_RES / 4);
    encodeComputePassAndSubmit(computePipeline, computeBindGroup[index], workgroupSize, workgroupSize, workgroupSize);
    index = (index + 1) % 2;
    lastUpdate = time;
  }

  device.queue.writeBuffer(renderUniformBuffer, 0, new Float32Array([
      ...viewMatrix,
      GRID_RES,
      AUDIO ? audioContext.currentTime : ((time - start) / 1000.0),
      programmableValue, 1.0
    ]));

  setupPerformanceTimer("Render");

  renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
  encodeRenderPassAndSubmit(renderPassDescriptor, renderPipeline, renderBindGroup[index]);

  requestAnimationFrame(render);
}

function setupPerformanceTimer(timerName)
{
  let begin = performance.now();

  device.queue.onSubmittedWorkDone()
    .then(function() {
      let end = performance.now();
      let frameTime = end - begin;
      document.title = `${(frameTime).toFixed(2)} / ${(1000.0 / frameTime).toFixed(2)}`;
    }).catch(function(err) {
      console.log(err);
    });
}

function resetView()
{
  eye = vec3.create(0, 0, GRID_RES + GRID_RES * 0.1);
  dir = vec3.create(0, 0, -1);
  programmableValue = 0.0;
}

function computeViewMatrix()
{
  viewMatrix = mat4.lookAt(eye, vec3.add(eye, dir), vec3.create(0, 1, 0));
}

function handleKeyEvent(e)
{
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

function handleMouseMoveEvent(e)
{
  let yaw = -e.movementX * LOOK_VELOCITY;
  let pitch = e.movementY * LOOK_VELOCITY;

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
  vec3.transformMat4(dir, mat4.rotation(mat4.getAxis(viewMatrix, 0), pitch), dir);
  vec3.transformMat4(dir, mat4.rotationY(yaw), dir);

  computeViewMatrix();
}

function handleMouseWheelEvent(e)
{
  programmableValue -= e.deltaY * WHEEL_VELOCITY;
  console.log("value:" + programmableValue);
}

function startRender()
{
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

  let grid = new Uint32Array(GRID_RES * GRID_RES * GRID_RES);

  for(let i=0; i<2; i++) {
    for(let j=0; j<grid.length; j++)
      grid[j] = Math.random() > (0.99 * 0.5 * (i + 1)) ? 1 : 0;
    
    device.queue.writeBuffer(gridBuffer[i], 0, grid);
  }

  device.queue.writeBuffer(computeStorageBuffer, 0, new Uint32Array([
      0, 0, 0, 0, GRID_RES - 1, GRID_RES - 1, GRID_RES - 1, 0]));

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

  setInterval(createPipelines, 500); // Reload shader
}

async function main()
{
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
    await prepareAudio();
  }

  await createGPUResources();
  await createPipelines();

  document.body.innerHTML = "<button>CLICK<canvas style='width:0;cursor:none'>";
  canvas = document.querySelector("canvas");
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  let presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  if (presentationFormat !== "bgra8unorm")
    throw new Error(`Expected canvas pixel format of bgra8unorm but was '${presentationFormat}'.`);

  context = canvas.getContext("webgpu");
  context.configure({device, format: presentationFormat, alphaMode: "opaque"});

  if (AUDIO) {
    document.querySelector("button").addEventListener("click", startRender);
  } else {
    startRender();
  }
}

main();
