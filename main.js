const FULLSCREEN = false;
const AUDIO = false;

const ASPECT = 1.6;
const CANVAS_WIDTH = 1024;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;
const FOV = 50.0;

const AUDIO_WIDTH = 4096;
const AUDIO_HEIGHT = 4096;

const MAX_GRID_RES = 128;
const DEFAULT_UPDATE_DELTA = 128;
const SIMULATION_OUTPUT_OFS = 0;

const MOVE_VELOCITY = 0.75;
const LOOK_VELOCITY = 0.025;
const WHEEL_VELOCITY = 0.005;

let audioContext;
let audioBufferSourceNode;

let device;
let uniformBuffer;
let gridBuffer = [];
let rulesBuffer;
let bindGroup = [];
let pipelineLayout;
let computePipeline;
let renderPipeline;
let renderPassDescriptor;

let canvas;
let context;

let right, up, fwd, eye;
let programmableValue;

let grid;
let rules;
let currGridRes = MAX_GRID_RES;

let start;
let paused = true;
let updateDelta = DEFAULT_UPDATE_DELTA;
let lastSimulationUpdate = 0;
let simulationStep = 0;

let rand = splitmix32(Math.floor(Math.random() * 4294967296));

// https://github.com/bryc/code/blob/master/jshash/PRNGs.md
function splitmix32(a) {
  console.log("Seed: " + a);
  return function() {
    a |= 0; a = a + 0x9e3779b9 | 0;
    var t = a ^ a >>> 16; t = Math.imul(t, 0x21f0aaad);
    t = t ^ t >>> 15; t = Math.imul(t, 0x735a2d97);
    return ((t = t ^ t >>> 15) >>> 0) / 4294967296;
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
      {binding: 3, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
    ]
  });
 
  // right, up, fwd, eye, fov, time, simulation step, programmable value/padding
  uniformBuffer = device.createBuffer({
    size: 16 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});

  for(let i=0; i<2; i++) {
    gridBuffer[i] = device.createBuffer({
      size: grid.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  }

  rulesBuffer = device.createBuffer({size: rules.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  for(let i=0; i<2; i++) {
    bindGroup[i] = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {binding: 0, resource: {buffer: uniformBuffer}},
        {binding: 1, resource: {buffer: gridBuffer[i]}},
        {binding: 2, resource: {buffer: gridBuffer[1 - i]}},
        {binding: 3, resource: {buffer: rulesBuffer}},
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
  if(start === undefined)
    start = AUDIO ? (audioContext.currentTime * 1000.0) : time;

  const currTime = AUDIO ? (audioContext.currentTime * 1000.0) : (time - start); 

  if(paused)
    lastSimulationUpdate = currTime;
  
  const commandEncoder = device.createCommandEncoder();
 
  // TODO Distribute one simulation step across different frames (within updateDelta 'budget')
  if(currTime - lastSimulationUpdate > updateDelta) {
    const count = Math.ceil(currGridRes / 4);
    encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup[simulationStep % 2], count, count, count);
    lastSimulationUpdate += updateDelta;
    simulationStep++;
  }

  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    ...right,
    0.5 / Math.tan(0.5 * FOV * Math.PI / 180.0),
    ...up,
    currTime,
    ...fwd,
    simulationStep,
    ...eye,
    programmableValue
  ]));

  setupPerformanceTimer();

  renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
  encodeRenderPassAndSubmit(commandEncoder, renderPassDescriptor, renderPipeline, bindGroup[simulationStep % 2]);
  
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

function initGrid(gridRes, seedAreaHalf)
{
  for(let i=0; i<grid.length; i++)
    grid[i] = 0;

  grid[0] = 1;
  grid[1] = gridRes;
  grid[2] = gridRes * gridRes;

  const center = gridRes / 2;
  const min = center - seedAreaHalf;
  const max = center + seedAreaHalf;

  for(let k=min; k<max; k++)
    for(let j=min; j<max; j++)
      for(let i=min; i<max; i++)
        grid[3 + gridRes * gridRes * k + gridRes * j + i] = rand() > 0.7 ? 1 : 0;

  device.queue.writeBuffer(gridBuffer[0], 0, grid);
  device.queue.writeBuffer(gridBuffer[1], 0, grid);

  console.log(`${(SIMULATION_OUTPUT_OFS + simulationStep)}, initGrid, ${gridRes}, ${seedAreaHalf}`);
}

function initRules(ruleSet)
{
  const RULE_OFS = 2;
  const BIRTH_OFS = 27;

  for(let i=0; i<rules.length; i++)
    rules[i] = 0;

  rules[0] = 0; // automaton kind
  rules[1] = 5; // states

  let id;
  switch(ruleSet) {
    case 1:
      id = "445";
      rules[RULE_OFS + 4] = 1;
      rules[RULE_OFS + BIRTH_OFS + 4] = 1;
      break;
    case 2:
      id = "amoeba";
      for(let i=9; i<27; i++)
        rules[RULE_OFS + i] = 1;
      rules[RULE_OFS + BIRTH_OFS + 5] = 1;
      rules[RULE_OFS + BIRTH_OFS + 6] = 1;
      rules[RULE_OFS + BIRTH_OFS + 7] = 1;
      rules[RULE_OFS + BIRTH_OFS + 12] = 1;
      rules[RULE_OFS + BIRTH_OFS + 13] = 1;
      rules[RULE_OFS + BIRTH_OFS + 15] = 1;
      break;
    case 3:
      id = "pyro5"; 
      rules[1] = 5;
      rules[RULE_OFS + 4] = 1;
      rules[RULE_OFS + 5] = 1;
      rules[RULE_OFS + 6] = 1;
      rules[RULE_OFS + 7] = 1;
      rules[RULE_OFS + BIRTH_OFS + 6] = 1;
      rules[RULE_OFS + BIRTH_OFS + 7] = 1;
      rules[RULE_OFS + BIRTH_OFS + 8] = 1;
      break;
    case 4:
      id = "empty";
      break;
    case 5:
      id = "empty";
      break;
    case 6:
      id = "empty";
      break;
     case 7:
      id = "empty";
      break;
    case 8:
      id = "single";
      rules[1] = 2;
      rules[RULE_OFS + BIRTH_OFS + 2] = 1;
      break;
    case 9:
      id = "clouds";
      rules[1] = 5;
      for(let i=13; i<27; i++)
        rules[RULE_OFS + i] = 1;
      rules[RULE_OFS + BIRTH_OFS + 13] = 1;
      rules[RULE_OFS + BIRTH_OFS + 14] = 1;
      rules[RULE_OFS + BIRTH_OFS + 17] = 1;
      rules[RULE_OFS + BIRTH_OFS + 18] = 1;
      rules[RULE_OFS + BIRTH_OFS + 19] = 1;
      break;
    case 0:
      id = "decay";
      rules[1] = 3;
      for(let i=13; i<27; i++)
        rules[RULE_OFS + i] = 1;
      for(let i=10; i<27; i++)
        rules[RULE_OFS + BIRTH_OFS + i] = 1;
      break;
  }

  device.queue.writeBuffer(rulesBuffer, 0, rules);
  
  console.log(`${(SIMULATION_OUTPUT_OFS + simulationStep)}, initRules, ${ruleSet}, "${id}"`);
}

function resetView()
{
  eye = [currGridRes, currGridRes, currGridRes];
  eye = vec3Add(eye, vec3Scale(eye, 0.05));
  fwd = vec3Normalize(vec3Add([currGridRes/2, currGridRes/2, currGridRes/2], vec3Negate(eye)));

  programmableValue = 0.0;

  computeView();
}

function computeView()
{
  right = vec3Normalize(vec3Cross(fwd, [0, 1, 0]));
  up = vec3Cross(right, fwd);
}

function handleKeyEvent(e)
{
  if(e.key !== " " && !isNaN(e.key))
  {
    initRules(parseInt(e.key));
    return;
  }

  switch (e.key) {
    case "a":
      eye = vec3Add(eye, vec3Scale(right, -MOVE_VELOCITY));
      computeView();
      break;
    case "d":
      eye = vec3Add(eye, vec3Scale(right, MOVE_VELOCITY));
      computeView();
      break;
    case "w":
      eye = vec3Add(eye, vec3Scale(fwd, MOVE_VELOCITY));
      computeView();
      break;
    case "s":
      eye = vec3Add(eye, vec3Scale(fwd, -MOVE_VELOCITY));
      computeView();
      break;
    case "r":
      resetView();
      break;
    case "i":
      initGrid(currGridRes, 15);
      break;
    case "+":
      updateDelta += updateDelta / 4;
      break;
    case "-":
      updateDelta -= updateDelta / 4;
      break;
    case "p":
      createPipelines();
      break;
    case " ":
      paused = !paused;
    break;
  }
}

function handleMouseMoveEvent(e)
{
  let yaw = -e.movementX * LOOK_VELOCITY;
  let pitch = -e.movementY * LOOK_VELOCITY;

  const currentPitch = Math.acos(fwd[1]);
  const newPitch = currentPitch - pitch;
  const minPitch = Math.PI / 180.0;
  const maxPitch = 179.0 * Math.PI / 180.0;

  if(newPitch < minPitch) {
    pitch = currentPitch - minPitch;
  }
  if(newPitch > maxPitch) {
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
}

function startRender()
{
  if(FULLSCREEN) {
    canvas.requestFullscreen();
  } else {
    canvas.style.width = CANVAS_WIDTH;
    canvas.style.height = CANVAS_HEIGHT;
    canvas.style.position = "absolute";
    canvas.style.left = 0;
    canvas.style.top = 0;
  }

  initGrid(currGridRes, 15);
  initRules(1);
  resetView();

  document.querySelector("button").removeEventListener("click", startRender);

  canvas.addEventListener("click", async () => {
    if(!document.pointerLockElement) {
      await canvas.requestPointerLock({unadjustedMovement: true});
    }
  });

  document.addEventListener("pointerlockchange", () => {
    if(document.pointerLockElement === canvas) {
      document.addEventListener("keydown", handleKeyEvent);
      canvas.addEventListener("mousemove", handleMouseMoveEvent);
      canvas.addEventListener("wheel", handleMouseWheelEvent);
    } else {
      document.removeEventListener("keydown", handleKeyEvent);
      canvas.removeEventListener("mousemove", handleMouseMoveEvent);
      canvas.removeEventListener("wheel", handleMouseWheelEvent);
    }
  });

  if(AUDIO) {
    audioBufferSourceNode.start();
  }

  requestAnimationFrame(render);
}

async function main()
{
  if(!navigator.gpu) {
    throw new Error("WebGPU is not supported on this browser.");
  }

  const gpuAdapter = await navigator.gpu.requestAdapter();
  if(!gpuAdapter) {
    throw new Error("Can not use WebGPU. No GPU adapter available.");
  }

  device = await gpuAdapter.requestDevice();
  if(!device) {
    throw new Error("Failed to request logical device.");
  }

  if(AUDIO) {
    await prepareAudio();
  }

  // Initialize to max size even if we only use a portion of it
  grid = new Uint32Array(3 + MAX_GRID_RES * MAX_GRID_RES * MAX_GRID_RES);
  rules = new Uint32Array(2 + 2 * 27);

  await createGPUResources();
  await createPipelines();

  document.body.innerHTML = "<button>CLICK<canvas style='width:0;cursor:none'>";
  canvas = document.querySelector("canvas");
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  let presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  if(presentationFormat !== "bgra8unorm")
    throw new Error(`Expected canvas pixel format of bgra8unorm but was '${presentationFormat}'.`);

  context = canvas.getContext("webgpu");
  context.configure({device, format: presentationFormat, alphaMode: "opaque"});

  if(AUDIO || FULLSCREEN) {
    document.querySelector("button").addEventListener("click", startRender);
  } else {
    startRender();
  }
}

main();
