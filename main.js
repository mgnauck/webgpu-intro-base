const FULLSCREEN = false;
const AUDIO = false;

const ASPECT = 1.6;
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;
const FOV = 50.0;

const AUDIO_WIDTH = 4096;
const AUDIO_HEIGHT = 4096;

const GRID_RES = 128.0;
const SEED_AREA = 5;
const UPDATE_INTERVAL = 150.0;

const MOVE_VELOCITY = 0.5;
const LOOK_VELOCITY = 0.025;
const WHEEL_VELOCITY = 0.0025;

let audioContext;
let audioBufferSourceNode;

let device;

let uniformBuffer;
let gridBuffer = [];
let bindGroup = [];
let pipelineLayout;
let computePipeline;
let renderPipeline;
let renderPassDescriptor;

let canvas;
let context;

let right, up, fwd, eye;
let programmableValue;

let start, lastUpdate;
let simulationSteps = 0;
let pause = false;
let initialGrid = new Uint32Array(9 + GRID_RES * GRID_RES * GRID_RES);

let rand = splitmix32(xmur3("unik"));

// https://github.com/bryc/code/blob/master/jshash/PRNGs.md
function splitmix32(a) {
  return function() {
    a |= 0; a = a + 0x9e3779b9 | 0;
    var t = a ^ a >>> 16; t = Math.imul(t, 0x21f0aaad);
    t = t ^ t >>> 15; t = Math.imul(t, 0x735a2d97);
    return ((t = t ^ t >>> 15) >>> 0) / 4294967296;
  }
}

function xmur3(str)
{
  for(var i=0, h=1779033703 ^ str.length; i<str.length; i++)
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353), h = h << 13 | h >>> 19;
  return function()
  {
    h = Math.imul(h ^ h >>> 16, 2246822507),
    h = Math.imul(h ^ h >>> 13, 3266489909);
    return (h ^= h >>> 16) >>> 0;
  }
}

function loadTextFile(url)
{
  return fetch(url).then(response => response.text());
}

async function createComputePipeline(shaderModule, pipelineLayout)
{
  return device.createComputePipelineAsync({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: "c"
    }
  });
}

async function createRenderPipeline(shaderModule, pipelineLayout)
{
  return device.createRenderPipelineAsync({
    layout: pipelineLayout,
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

function encodeComputePassAndSubmit(commandEncoder, pipeline, bindGroup, countX, countY, countZ)
{
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(countX, countY, countZ);
  passEncoder.end();
}

function encodeRenderPassAndSubmit(commandEncoder, passDescriptor, pipeline, bindGroup)
{
  const passEncoder = commandEncoder.beginRenderPass(passDescriptor);
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.draw(4);
  passEncoder.end();
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

  let bindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: {format: "rg32float"}
    }]
  });

  let bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{binding: 0, resource: audioTexture.createView()}]
  });

  const readBuffer = device.createBuffer({
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    size: AUDIO_WIDTH * AUDIO_HEIGHT * 2 * 4
  });

  setupPerformanceTimer("Audio");

  let shaderModule = device.createShaderModule({code: await loadTextFile("audioShader.wgsl")});
  let pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});

  const commandEncoder = device.createCommandEncoder();
  
  encodeComputePassAndSubmit(commandEncoder, await createComputePipeline(shaderModule, pipelineLayout),
      bindGroup, Math.ceil(AUDIO_WIDTH / 8), Math.ceil(AUDIO_HEIGHT / 8), 1);

  commandEncoder.copyTextureToBuffer(
    {texture: audioTexture}, {buffer: readBuffer, bytesPerRow: AUDIO_WIDTH * 2 * 4}, [AUDIO_WIDTH, AUDIO_HEIGHT]);

  device.queue.submit([commandEncoder.finish()]);

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
  let bindGroupLayout = device.createBindGroupLayout({
    entries: [ 
      {binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
      {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
    ]
  });
 
  // right, up, fwd, eye, fov, time, 2 x programmable value (= padding)
  uniformBuffer = device.createBuffer({
    size: 16 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});

  for(let i=0; i<2; i++) {
    gridBuffer[i] = device.createBuffer({
      size: (9 + GRID_RES * GRID_RES * GRID_RES) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  }

  for(let i=0; i<2; i++) {
    bindGroup[i] = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {binding: 0, resource: {buffer: uniformBuffer}},
        {binding: 1, resource: {buffer: gridBuffer[i]}},
        {binding: 2, resource: {buffer: gridBuffer[1 - i]}}
      ]
    });
  }

  pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});

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
  let shaderCode = await loadTextFile("contentShader.wgsl");
  let shaderModule = device.createShaderModule({code: shaderCode});

  computePipeline = await createComputePipeline(shaderModule, pipelineLayout);
  renderPipeline = await createRenderPipeline(shaderModule, pipelineLayout);
}

function render(time)
{
  if (start === undefined) {
    start = AUDIO ? (audioContext.currentTime * 1000.0) : time;
    lastUpdate = start;
  }

  const currTime = AUDIO ? (audioContext.currentTime * 1000.0) : (time - start);
  const commandEncoder = device.createCommandEncoder();

  if(!pause && (currTime - lastUpdate > UPDATE_INTERVAL)) {
    const count = Math.ceil(GRID_RES / 4);
    // Reset grid min/max of output buffer
    device.queue.writeBuffer(gridBuffer[1 - simulationSteps % 2], 12, initialGrid, 12, 24);
    encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup[simulationSteps % 2], count, count, count);
    simulationSteps++;
    lastUpdate += UPDATE_INTERVAL;
  }

  if(pause) {
    lastUpdate = currTime;
  }

  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    ...right,
    0.5 / Math.tan(0.5 * FOV * Math.PI / 180.0),
    ...up,
    currTime,
    ...fwd,
    programmableValue,
    ...eye,
    1.0 // programmableValue2
  ]));

  setupPerformanceTimer();

  renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
  encodeRenderPassAndSubmit(commandEncoder, renderPassDescriptor, renderPipeline, bindGroup[simulationSteps % 2]);
  
  device.queue.submit([commandEncoder.finish()]);

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

function vec3Add(a, b)
{
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vec3Negate(v)
{
  return [-v[0], -v[1], -v[2]];
}

function vec3Scale(v, s)
{
  return [v[0] * s, v[1] * s, v[2] * s];
}

function vec3Cross(a, b)
{
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function vec3Normalize(v)
{
  let invLen = 1.0 / Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  return [v[0] * invLen, v[1] * invLen, v[2] * invLen];
}

function vec3Transform(v, m)
{
  const x = v[0];
  const y = v[1];
  const z = v[2];

  return [x * m[0] + y * m[4] + z * m[8],
          x * m[1] + y * m[5] + z * m[9],
          x * m[2] + y * m[6] + z * m[10]];
}

function axisRotation(axis, angle)
{
  let x = axis[0];
  let y = axis[1];
  let z = axis[2];
  const l = 1.0 / Math.sqrt(x * x + y * y + z * z);
  x *= l;
  y *= l;
  z *= l;
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const oneMinusCosine = 1 - c;

  return [xx + (1 - xx) * c, x * y * oneMinusCosine + z * s, x * z * oneMinusCosine - y * s, 0,
          x * y * oneMinusCosine - z * s, yy + (1 - yy) * c, y * z * oneMinusCosine + x * s, 0,
          x * z * oneMinusCosine + y * s, y * z * oneMinusCosine - x * s, zz + (1 - zz) * c, 0,
          0, 0, 0, 1]
}

function initGrid()
{ 
  device.queue.writeBuffer(gridBuffer[0], 0, initialGrid);
  device.queue.writeBuffer(gridBuffer[1], 0, initialGrid); 
}

function createGrid()
{
  // Grid multiplier
  initialGrid[0] = 1;
  initialGrid[1] = GRID_RES;
  initialGrid[2] = GRID_RES * GRID_RES;

  let min = GRID_RES / 2 - SEED_AREA;
  initialGrid[3] = min - 1;
  initialGrid[4] = min - 1;
  initialGrid[5] = min - 1;
  
  let max = GRID_RES / 2 + SEED_AREA;
  initialGrid[6] = max + 1;
  initialGrid[7] = max + 1;
  initialGrid[8] = max + 1;

  for(let k=min; k<max; k++)
    for(let j=min; j<max; j++)
      for(let i=min; i<max; i++)
        initialGrid[9 + GRID_RES * GRID_RES * k + GRID_RES * j + i] = rand() > 0.6 ? 1 : 0;
  
  initGrid();
}   

function resetView()
{
  eye = [GRID_RES, GRID_RES, GRID_RES];
  eye = vec3Add(eye, vec3Scale(eye, 0.05));
  fwd = vec3Normalize(vec3Add([GRID_RES/2, GRID_RES/2, GRID_RES/2], vec3Negate(eye)));

  programmableValue = 0.0;
}

function computeView()
{
  right = vec3Normalize(vec3Cross(fwd, [0, 1, 0]));
  up = vec3Cross(right, fwd);
}

function handleKeyEvent(e)
{
  switch (e.key) {
    case "a":
      eye = vec3Add(eye, vec3Scale(right, -MOVE_VELOCITY));
      break;
    case "d":
      eye = vec3Add(eye, vec3Scale(right, MOVE_VELOCITY));
      break;
    case "w":
      eye = vec3Add(eye, vec3Scale(fwd, MOVE_VELOCITY));
      break;
    case "s":
      eye = vec3Add(eye, vec3Scale(fwd, -MOVE_VELOCITY));
      break;
    case "r":
      resetView();
      break;
    case "i":
      initGrid()
      break;
    case "n":
      createGrid();
      break;
    case "l":
      createPipelines();
      console.log("Shader module reloaded");
      break;
    case "p":
      pause = !pause;
      console.log("Simulation paused");
    break;
  };

  computeView();
}

function handleMouseMoveEvent(e)
{
  let yaw = -e.movementX * LOOK_VELOCITY;
  let pitch = -e.movementY * LOOK_VELOCITY;

  const currentPitch = Math.acos(fwd[1]);
  const newPitch = currentPitch - pitch;
  const minPitch = Math.PI / 180.0;
  const maxPitch = 179.0 * Math.PI / 180.0;

  if (newPitch < minPitch) {
    pitch = currentPitch - minPitch;
  }
  if (newPitch > maxPitch) {
    pitch = currentPitch - maxPitch;
  }

  // Pitch locally, yaw globally to avoid unwanted roll
  fwd = vec3Transform(fwd, axisRotation(right, pitch));
  fwd = vec3Transform(fwd, axisRotation([0, 1, 0], yaw));

  computeView();
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
  computeView();
  createGrid();

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

  //setInterval(createPipelines, 500); // Reload shader
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

  if (AUDIO || FULLSCREEN) {
    document.querySelector("button").addEventListener("click", startRender);
  } else {
    startRender();
  }
}

main();
