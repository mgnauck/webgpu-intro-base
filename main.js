const FULLSCREEN = false;
const AUDIO = false;

const RECORDING = false;
const PAUSE_AND_RECORD_AT = -1; // Simulation step where we stop playin and switch into recoding mode
const START_PAUSED = RECORDING | (PAUSE_AND_RECORD_AT == 0);
const OVERVIEW_CAMERA = true;

const ASPECT = 1.6;
const CANVAS_WIDTH = 1024;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;
const FOV = 50.0;

const AUDIO_WIDTH = 4096;
const AUDIO_HEIGHT = 4096;

const MAX_GRID_RES = 128;
const DEFAULT_UPDATE_DELTA = 250;
const RECORDING_OFS = 0;

const MOVE_VELOCITY = 0.75;
const LOOK_VELOCITY = 0.025;
const WHEEL_VELOCITY = 0.005;
const UPDATE_DELTA_CHANGE_FACTOR = 4;

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
let overviewCamera = OVERVIEW_CAMERA;

let seed = Math.floor(Math.random() * 4294967296);
let rand;
let gridRes;
let grid;
let updateDelta;
let rules;

let paused = START_PAUSED;
let recording = RECORDING;
let start;
let lastSimulationUpdate;
let simulationStep;
let activeSimulationStep;

let gridEvents = [
{ step: 0, obj: { gridRes: 128, seed: 321940067, area: 4 } }, // GRID_EVENT
];

let ruleEvents = [
{ step: 0, obj: { ruleSet: 2 } }, // RULE_EVENT: amoeba
{ step: 187, obj: { ruleSet: 3 } }, // RULE_EVENT: pyro
{ step: 233, obj: { ruleSet: 2 } }, // RULE_EVENT: amoeba
{ step: 278, obj: { ruleSet: 9 } }, // RULE_EVENT: clouds
{ step: 296, obj: { ruleSet: 2 } }, // RULE_EVENT: amoeba
{ step: 308, obj: { ruleSet: 4 } }, // RULE_EVENT: grids
{ step: 540, obj: { ruleSet: 1 } }, // RULE_EVENT: 445
{ step: 887, obj: { ruleSet: 7 } }, // RULE_EVENT: empty
{ step: 887, obj: { ruleSet: 8 } }, // RULE_EVENT: single
{ step: 895, obj: { ruleSet: 3 } }, // RULE_EVENT: pyro
{ step: 895, obj: { ruleSet: 4 } }, // RULE_EVENT: grids
{ step: 903, obj: { ruleSet: 2 } }, // RULE_EVENT: amoeba
{ step: 948, obj: { ruleSet: 8 } }, // RULE_EVENT: single
{ step: 960, obj: { ruleSet: 2 } }, // RULE_EVENT: amoeba
{ step: 977, obj: { ruleSet: 9 } }, // RULE_EVENT: clouds
{ step: 1003, obj: { ruleSet: 0 } }, // RULE_EVENT: decay
{ step: 1058, obj: { ruleSet: 6 } }, // RULE_EVENT: empty
];

let simulationSpeedEvents = [
{ step: 0, obj: { delta: 250 } }, // SPEED_EVENT
{ step: 29, obj: { delta: 188 } }, // SPEED_EVENT
{ step: 30, obj: { delta: 141 } }, // SPEED_EVENT
{ step: 31, obj: { delta: 106 } }, // SPEED_EVENT
{ step: 33, obj: { delta: 80 } }, // SPEED_EVENT
{ step: 41, obj: { delta: 60 } }, // SPEED_EVENT
{ step: 44, obj: { delta: 45 } }, // SPEED_EVENT
{ step: 120, obj: { delta: 56 } }, // SPEED_EVENT
{ step: 124, obj: { delta: 70 } }, // SPEED_EVENT
{ step: 130, obj: { delta: 88 } }, // SPEED_EVENT
{ step: 135, obj: { delta: 110 } }, // SPEED_EVENT
{ step: 143, obj: { delta: 138 } }, // SPEED_EVENT
{ step: 145, obj: { delta: 173 } }, // SPEED_EVENT
{ step: 161, obj: { delta: 216 } }, // SPEED_EVENT
{ step: 366, obj: { delta: 162 } }, // SPEED_EVENT
{ step: 376, obj: { delta: 122 } }, // SPEED_EVENT
{ step: 389, obj: { delta: 92 } }, // SPEED_EVENT
{ step: 443, obj: { delta: 69 } }, // SPEED_EVENT
{ step: 499, obj: { delta: 86 } }, // SPEED_EVENT
{ step: 501, obj: { delta: 108 } }, // SPEED_EVENT
{ step: 502, obj: { delta: 135 } }, // SPEED_EVENT
{ step: 503, obj: { delta: 169 } }, // SPEED_EVENT
{ step: 508, obj: { delta: 211 } }, // SPEED_EVENT
{ step: 511, obj: { delta: 264 } }, // SPEED_EVENT
{ step: 570, obj: { delta: 198 } }, // SPEED_EVENT
{ step: 577, obj: { delta: 149 } }, // SPEED_EVENT
{ step: 578, obj: { delta: 112 } }, // SPEED_EVENT
{ step: 580, obj: { delta: 84 } }, // SPEED_EVENT
{ step: 631, obj: { delta: 63 } }, // SPEED_EVENT
{ step: 634, obj: { delta: 47 } }, // SPEED_EVENT
{ step: 735, obj: { delta: 59 } }, // SPEED_EVENT
{ step: 738, obj: { delta: 74 } }, // SPEED_EVENT
{ step: 742, obj: { delta: 93 } }, // SPEED_EVENT
{ step: 745, obj: { delta: 116 } }, // SPEED_EVENT
{ step: 748, obj: { delta: 145 } }, // SPEED_EVENT
{ step: 775, obj: { delta: 181 } }, // SPEED_EVENT
];

let cameraEvents = [];
let fadeEvents = [];

// https://github.com/bryc/code/blob/master/jshash/PRNGs.md
function splitmix32(a) {
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

async function createComputePipeline(shaderModule, pipelineLayout, entryPoint)
{
  return device.createComputePipelineAsync({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: entryPoint
    }
  });
}

async function createRenderPipeline(shaderModule, pipelineLayout, vertexEntryPoint, fragmentEntryPoint)
{
  return device.createRenderPipelineAsync({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: vertexEntryPoint
    },
    fragment: {
      module: shaderModule,
      entryPoint: fragmentEntryPoint,
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

  setPerformanceTimer("Audio");

  let shaderModule = device.createShaderModule({code: await loadTextFile("audioShader.wgsl")});
  let pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});

  const commandEncoder = device.createCommandEncoder();
  
  encodeComputePassAndSubmit(commandEncoder, await createComputePipeline(shaderModule, pipelineLayout, "audioMain"),
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
 
  // Buffer space for right, up, fwd, eye, fov, time, simulation step, programmable value/padding
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

  computePipeline = await createComputePipeline(shaderModule, pipelineLayout, "computeMain");
  renderPipeline = await createRenderPipeline(shaderModule, pipelineLayout, "vertexMain", "fragmentMain");
}

function render(time)
{
  if(start === undefined) {
    start = AUDIO ? (audioContext.currentTime * 1000.0) : time;
    lastSimulationUpdate = 0;
    simulationStep = 0;
    activeSimulationStep = -1;
  }

  const currTime = AUDIO ? (audioContext.currentTime * 1000.0) : (time - start); 

  if(paused)
    lastSimulationUpdate = currTime;

  if(!paused && !recording)
    update(currTime);

  // TEMPTEMPTEMP
  if(overviewCamera) {
    let speed = 0.00025;
    let center = vec3Scale([gridRes, gridRes, gridRes], 0.5);
    let pos = [gridRes * Math.sin(currTime * speed), 0.75 * gridRes * Math.sin(currTime * speed), gridRes * Math.cos(currTime * speed)];
    pos = vec3Add(center, pos);
    setView(pos, vec3Normalize(vec3Add(center, vec3Negate(pos))));
  }
 
  const commandEncoder = device.createCommandEncoder();
 
  // TODO Distribute one simulation step across different frames (within updateDelta 'budget')
  if(currTime - lastSimulationUpdate > updateDelta) {
    const count = Math.ceil(gridRes / 4);
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

  setPerformanceTimer();

  renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
  encodeRenderPassAndSubmit(commandEncoder, renderPassDescriptor, renderPipeline, bindGroup[simulationStep % 2]);
  
  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);
}

function updateEvents(events, updateFunction)
{
  events.forEach(e => {
    if(e.step == simulationStep) {
      updateFunction(e.obj);
      return;
    }
  });
}

function update(currTime)
{
  if(simulationStep == PAUSE_AND_RECORD_AT) {
    recording = true;
    paused = true;
    return;
  }

  if(simulationStep > activeSimulationStep)
  {
    updateEvents(gridEvents, setGrid);
    updateEvents(ruleEvents, setRules);
    updateEvents(simulationSpeedEvents, setUpdateDelta);

    // TODO Implement actual camera handling/updates here
  }

  activeSimulationStep = simulationStep;
}

function setPerformanceTimer(timerName)
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

function setGrid(obj)
{
  for(let i=0; i<grid.length; i++)
    grid[i] = 0;

  if(rand === undefined || obj.seed != seed) {
    seed = obj.seed;
    rand = splitmix32(seed);
  }

  gridRes = Math.min(obj.gridRes, MAX_GRID_RES);

  grid[0] = 1;
  grid[1] = gridRes;
  grid[2] = gridRes * gridRes;

  const center = gridRes / 2;
  const min = center - obj.area;
  const max = center + obj.area;

  for(let k=min; k<max; k++)
    for(let j=min; j<max; j++)
      for(let i=min; i<max; i++)
        grid[3 + gridRes * gridRes * k + gridRes * j + i] = rand() > 0.7 ? 1 : 0;

  device.queue.writeBuffer(gridBuffer[0], 0, grid);
  device.queue.writeBuffer(gridBuffer[1], 0, grid);

  // TEMPTEMP
  resetView();

  // TEMPTEMP only for recording
  if(simulationStep === undefined)
    simulationStep = 0;

  console.log(`{ step: ${(RECORDING_OFS + simulationStep)}, obj: { gridRes: ${gridRes}, seed: ${seed}, area: ${obj.area} } }, // GRID_EVENT`);
}

function setRules(obj)
{
  const RULE_OFS = 2;
  const BIRTH_OFS = 27;

  for(let i=0; i<rules.length; i++)
    rules[i] = 0;

  rules[0] = 0; // automaton kind (not used at the moment)
  rules[1] = 5; // number of states

  let name;
  switch(obj.ruleSet) {
    case 1:
      name = "445";
      rules[RULE_OFS + 4] = 1;
      rules[RULE_OFS + BIRTH_OFS + 4] = 1;
      break;
    case 2:
      name = "amoeba";
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
      name = "pyro"; 
      rules[1] = 10;
      rules[RULE_OFS + 4] = 1;
      rules[RULE_OFS + 5] = 1;
      rules[RULE_OFS + 6] = 1;
      rules[RULE_OFS + 7] = 1;
      rules[RULE_OFS + BIRTH_OFS + 6] = 1;
      rules[RULE_OFS + BIRTH_OFS + 7] = 1;
      rules[RULE_OFS + BIRTH_OFS + 8] = 1;
      break;
    case 4:
      name = "grids";
      rules[1] = 4;
      for(let i=7; i<27; i++)
        rules[RULE_OFS + i] = 1;
      rules[RULE_OFS + BIRTH_OFS + 4] = 1;
      break;
    case 5:
      name = "empty";
      break;
    case 6:
      name = "empty";
      break;
     case 7:
      name = "empty";
      break;
    case 8:
      name = "single";
      rules[1] = 2;
      rules[RULE_OFS + BIRTH_OFS + 2] = 1;
      break;
    case 9:
      name = "clouds";
      for(let i=13; i<27; i++)
        rules[RULE_OFS + i] = 1;
      rules[RULE_OFS + BIRTH_OFS + 13] = 1;
      rules[RULE_OFS + BIRTH_OFS + 14] = 1;
      rules[RULE_OFS + BIRTH_OFS + 17] = 1;
      rules[RULE_OFS + BIRTH_OFS + 18] = 1;
      rules[RULE_OFS + BIRTH_OFS + 19] = 1;
      break;
    case 0:
      name = "decay";
      rules[RULE_OFS + 1] = 1;
      rules[RULE_OFS + 4] = 1;
      rules[RULE_OFS + 8] = 1;
      rules[RULE_OFS + 11] = 1;
      for(let i=13; i<27; i++) {
        rules[RULE_OFS + i] = 1;
        rules[RULE_OFS + BIRTH_OFS + i] = 1;
      }
      break;
  }

  device.queue.writeBuffer(rulesBuffer, 0, rules);

  if(simulationStep === undefined)
    simulationStep = 0;

  console.log(`{ step: ${(RECORDING_OFS + simulationStep)}, obj: { ruleSet: ${obj.ruleSet} } }, // RULE_EVENT: ${name}`);
}

function setUpdateDelta(obj)
{
  updateDelta = obj.delta;

  if(simulationStep === undefined)
    simulationStep = 0;

  console.log(`{ step: ${(RECORDING_OFS + simulationStep)}, obj: { delta: ${updateDelta} } }, // SPEED_EVENT`);
}

function setView(e, f)
{
  eye = e;
  fwd = f;
  right = vec3Normalize(vec3Cross(fwd, [0, 1, 0]));
  up = vec3Cross(right, fwd);
}

function resetView()
{
  setView([gridRes, gridRes, gridRes],
    vec3Normalize(vec3Add(vec3Scale([gridRes, gridRes, gridRes], 0.5), vec3Negate([gridRes, gridRes, gridRes]))));
}

function handleKeyEvent(e)
{
  switch (e.key) {
    case "a":
      setView(vec3Add(eye, vec3Scale(right, -MOVE_VELOCITY)), fwd);
      break;
    case "d":
      setView(vec3Add(eye, vec3Scale(right, MOVE_VELOCITY)), fwd);
      break;
    case "w":
      setView(vec3Add(eye, vec3Scale(fwd, MOVE_VELOCITY)), fwd);
      break;
    case "s":
      setView(vec3Add(eye, vec3Scale(fwd, -MOVE_VELOCITY)), fwd);
      break;
    case "r":
      resetView();
      break;
    case "c":
      overviewCamera = !overviewCamera;
      break;
    case "o":
      recording = !recording;
      break;
    case "p":
      // Reload shader
      createPipelines();
      break;
    case " ":
      paused = !paused;
      break;
  }

  if(recording)
  {
    if(e.key !== " " && !isNaN(e.key))
    {
      setRules({ ruleSet: parseInt(e.key) });
      return;
    }

    switch(e.key) {
      case "i":
        setGrid({ gridRes: MAX_GRID_RES, seed: seed, area: 4 });
        break;
      case "+":
        setUpdateDelta({ delta: Math.round(updateDelta + updateDelta / UPDATE_DELTA_CHANGE_FACTOR) });
        break;
      case "-":
        setUpdateDelta({ delta: Math.round(updateDelta - updateDelta / UPDATE_DELTA_CHANGE_FACTOR) });
        break;
    }
  } else {
    switch(e.key) {
      case "Enter":
        // Reset simulation steps back to start
        start = undefined;
        // TODO Seek audio to begin of track for proper time
        break;
    }
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
  setView(eye, vec3Transform(vec3Transform(fwd, axisRotation(right, pitch)), axisRotation([0, 1, 0], yaw)));
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

  if(recording) {
    setGrid({ gridRes: MAX_GRID_RES, seed: seed, area: 4 });
    setRules({ ruleSet: 2 });
    setUpdateDelta({ delta: DEFAULT_UPDATE_DELTA });
  }
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
