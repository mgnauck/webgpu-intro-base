const FULLSCREEN = true;
const AUDIO = true;

const RECORDING = false;
const RECORDING_OFS = 0;
const START_RECORDING_AT = -1;
const OVERVIEW_CAMERA = true;

const ASPECT = 1.6;
const CANVAS_WIDTH = 1024;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;
const FOV = 50.0;

const AUDIO_WIDTH = 4096;
const AUDIO_HEIGHT = 4096;

const MAX_GRID_RES = 128;
const DEFAULT_UPDATE_DELAY = 250;

const MOVE_VELOCITY = 0.75;
const LOOK_VELOCITY = 0.025;
const WHEEL_VELOCITY = 0.005;

let idle = RECORDING;
let recording = RECORDING;

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
let updateDelay = DEFAULT_UPDATE_DELAY;
let rules;

let startTime;
let lastSimulationUpdateTime = 0;
let simulationPaused = false;
let simulationStep = 0;
let previousSimulationStep = -1;
let simulationIteration = 0;

const GRID_EVENTS = [
{ step: 0, obj: { gridRes: 128, seed: 1464541643, area: 4 } }, // GRID_EVENT
];

const RULE_EVENTS = [
{ step: 0, obj: { ruleSet: 2 } }, // RULE_EVENT (amoeba, states: 5)
{ step: 0, obj: { ruleSet: 7 } }, // RULE_EVENT (shell, states: 2)
{ step: 14, obj: { ruleSet: 5 } }, // RULE_EVENT (coral, states: 4)
{ step: 29, obj: { ruleSet: 6 } }, // RULE_EVENT (crystal, states: 5)
{ step: 34, obj: { ruleSet: 3 } }, // RULE_EVENT (pyro, states: 10)
{ step: 46, obj: { ruleSet: 9 } }, // RULE_EVENT (pulse, states: 10)
{ step: 52, obj: { ruleSet: 2 } }, // RULE_EVENT (amoeba, states: 5)
{ step: 67, obj: { ruleSet: 6 } }, // RULE_EVENT (crystal, states: 5)
{ step: 71, obj: { ruleSet: 7 } }, // RULE_EVENT (shell, states: 2)
{ step: 77, obj: { ruleSet: 0 } }, // RULE_EVENT (clouds, states: 5)
{ step: 112, obj: { ruleSet: 4 } }, // RULE_EVENT (framework, states: 4)
{ step: 158, obj: { ruleSet: 1 } }, // RULE_EVENT (445, states: 5)
{ step: 183, obj: { ruleSet: 9 } }, // RULE_EVENT (pulse, states: 10)
{ step: 194, obj: { ruleSet: 6 } }, // RULE_EVENT (crystal, states: 5)
{ step: 198, obj: { ruleSet: 8 } }, // RULE_EVENT (shell, states: 5)
{ step: 208, obj: { ruleSet: 3 } }, // RULE_EVENT (pyro, states: 10)
{ step: 227, obj: { ruleSet: 2 } }, // RULE_EVENT (amoeba, states: 5)
{ step: 236, obj: { ruleSet: 9 } }, // RULE_EVENT (pulse, states: 10)
{ step: 241, obj: { ruleSet: 5 } }, // RULE_EVENT (coral, states: 4)
{ step: 271, obj: { ruleSet: 3 } }, // RULE_EVENT (pyro, states: 10)
{ step: 284, obj: { ruleSet: 2 } }, // RULE_EVENT (amoeba, states: 5)
{ step: 296, obj: { ruleSet: 1 } }, // RULE_EVENT (445, states: 5)
{ step: 420, obj: { ruleSet: 4 } }, // RULE_EVENT (framework, states: 4)
];

const TIME_EVENTS = [
{ step: 296, obj: { delta: -50 } }, // TIME_EVENT (paused: false, updateDelay: 200
{ step: 297, obj: { delta: -50 } }, // TIME_EVENT (paused: false, updateDelay: 150
{ step: 298, obj: { delta: -50 } }, // TIME_EVENT (paused: false, updateDelay: 100
{ step: 299, obj: { delta: -50 } }, // TIME_EVENT (paused: false, updateDelay: 100
];

const CAMERA_EVENTS = [];

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

  setPerformanceTimer();

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
  if(startTime === undefined) {
    startTime = AUDIO ? (audioContext.currentTime * 1000.0) : time;
    
    lastSimulationUpdateTime = 0;
    simulationPaused = false;
    simulationStep = 0;
    previousSimulationStep = -1;
    simulationIteration = 0;
  }

  const currTime = AUDIO ? (audioContext.currentTime * 1000.0) : (time - startTime);

  if(idle)
    lastSimulationUpdateTime = currTime;

  if(!idle && !recording)
    update();

  const commandEncoder = device.createCommandEncoder();
 
  // TODO Distribute one simulation iteration across different frames (within updateDelay 'budget')
  if(currTime - lastSimulationUpdateTime > updateDelay) {
    if(!simulationPaused) {
      const count = Math.ceil(gridRes / 4);
      encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup[simulationIteration % 2], count, count, count); 
      simulationIteration++;
    }
    lastSimulationUpdateTime += updateDelay;
    simulationStep++;
  }

  updateCamera(currTime);

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
  encodeRenderPassAndSubmit(commandEncoder, renderPassDescriptor, renderPipeline, bindGroup[simulationIteration % 2]);
  
  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);
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

function update()
{
  if(simulationStep == START_RECORDING_AT) {
    recording = true;
    idle = true;
    return;
  }

  if(simulationStep > previousSimulationStep) {
    updateEvents(GRID_EVENTS, setGrid);
    updateEvents(RULE_EVENTS, setRules);
    updateEvents(TIME_EVENTS, setTime);
    previousSimulationStep = simulationStep;
  }
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

function updateCamera(currTime)
{
  if(overviewCamera) {
    let speed = 0.00025;
    let center = vec3Scale([gridRes, gridRes, gridRes], 0.5);
    let pos = [gridRes * Math.sin(currTime * speed), 0.75 * gridRes * Math.sin(currTime * speed), gridRes * Math.cos(currTime * speed)];
    pos = vec3Add(center, pos);
    setView(pos, vec3Normalize(vec3Add(center, vec3Negate(pos))));
  } 
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
      name = "framework";
      rules[1] = 4;
      for(let i=7; i<27; i++)
        rules[RULE_OFS + i] = 1;
      rules[RULE_OFS + BIRTH_OFS + 4] = 1;
      break;
    case 5:
      name = "coral";
      rules[1] = 4;
      rules[RULE_OFS + 5] = 1;
      rules[RULE_OFS + 6] = 1;
      rules[RULE_OFS + 7] = 1;
      rules[RULE_OFS + 8] = 1;
      rules[RULE_OFS + BIRTH_OFS + 6] = 1;
      rules[RULE_OFS + BIRTH_OFS + 7] = 1;
      rules[RULE_OFS + BIRTH_OFS + 9] = 1;
      rules[RULE_OFS + BIRTH_OFS + 12] = 1;
      break;
    case 6:
      name = "crystal";
      rules[RULE_OFS + 1] = 1;
      rules[RULE_OFS + 2] = 1;
      rules[RULE_OFS + BIRTH_OFS + 1] = 1;
      rules[RULE_OFS + BIRTH_OFS + 3] = 1;
      break;
    //case 7:
    //  name = "symmetry";
    //  rules[1] = 10;
    //  rules[RULE_OFS + BIRTH_OFS + 2] = 1;
    // break;
    case 7:
      name = "stable";
      rules[1] = 2;
      for(let i=13; i<27; i++)
        rules[RULE_OFS + i] = 1;
      for(let i=14; i<20; i++)
        rules[RULE_OFS + BIRTH_OFS + i] = 1;
    case 8:
      name = "shell";
      rules[RULE_OFS + 6] = 1;
      rules[RULE_OFS + 7] = 1;
      rules[RULE_OFS + 8] = 1;
      rules[RULE_OFS + 9] = 1;
      rules[RULE_OFS + 11] = 1;
      rules[RULE_OFS + 13] = 1;
      rules[RULE_OFS + 15] = 1;
      rules[RULE_OFS + 16] = 1;
      rules[RULE_OFS + 18] = 1;
      rules[RULE_OFS + BIRTH_OFS + 6] = 1;
      rules[RULE_OFS + BIRTH_OFS + 7] = 1;
      rules[RULE_OFS + BIRTH_OFS + 8] = 1;
      rules[RULE_OFS + BIRTH_OFS + 9] = 1;
      rules[RULE_OFS + BIRTH_OFS + 10] = 1;
      rules[RULE_OFS + BIRTH_OFS + 13] = 1;
      rules[RULE_OFS + BIRTH_OFS + 14] = 1;
      rules[RULE_OFS + BIRTH_OFS + 16] = 1;
      rules[RULE_OFS + BIRTH_OFS + 18] = 1;
      rules[RULE_OFS + BIRTH_OFS + 19] = 1;
      rules[RULE_OFS + BIRTH_OFS + 22] = 1;
      rules[RULE_OFS + BIRTH_OFS + 23] = 1;
      rules[RULE_OFS + BIRTH_OFS + 24] = 1;
      rules[RULE_OFS + BIRTH_OFS + 25] = 1;
     break;
   case 9:
      name = "pulse";
      rules[1] = 10;
      rules[RULE_OFS + 3] = 1;
      rules[RULE_OFS + BIRTH_OFS + 1] = 1;
      rules[RULE_OFS + BIRTH_OFS + 2] = 1;
      rules[RULE_OFS + BIRTH_OFS + 3] = 1;
      break;
     case 0:
      name = "clouds";
      for(let i=13; i<27; i++)
        rules[RULE_OFS + i] = 1;
      rules[RULE_OFS + BIRTH_OFS + 13] = 1;
      rules[RULE_OFS + BIRTH_OFS + 14] = 1;
      rules[RULE_OFS + BIRTH_OFS + 17] = 1;
      rules[RULE_OFS + BIRTH_OFS + 18] = 1;
      rules[RULE_OFS + BIRTH_OFS + 19] = 1;
      break;
  }

  device.queue.writeBuffer(rulesBuffer, 0, rules);

  console.log(`{ step: ${(RECORDING_OFS + simulationStep)}, obj: { ruleSet: ${obj.ruleSet} } }, // RULE_EVENT (${name}, states: ${rules[1]})`);
}

function setTime(obj)
{
  if(obj.delta == 0)
    simulationPaused = !simulationPaused;
  else
    updateDelay += obj.delta;
 
  console.log(`{ step: ${(RECORDING_OFS + simulationStep)}, obj: { delta: ${obj.delta} } }, // TIME_EVENT (paused: ${simulationPaused}, updateDelay: ${updateDelay})`);
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
    case "v":
      resetView();
      break;
    case "o":
      overviewCamera = !overviewCamera;
      break;
    case "Enter":
      recording = !recording;
      console.log("Recording " + (recording ? "enabled" : "disabled"));
      break;
    case "l":
      createPipelines();
      console.log("Shader reloaded");
      break;
    case " ":
      // Intro time and simulation time pause
      idle = !idle;
      console.log("Idle mode " + (idle ? "enabled" : "disabled"));
      break;
  }

  if(recording)
  {
    if(e.key !== " " && !isNaN(e.key))
    {
      setRules({ ruleSet: parseInt(e.key, 10) });
      return;
    }

    switch(e.key) {
      case "i":
        setGrid({ gridRes: MAX_GRID_RES, seed: seed, area: 4 });
        break;
      case "#":
        // Pause simulation (but intro time goes on)
        setTime({ delta: 0 });
        break;
      case "+":
        setTime({ delta: 50 });
        break;
      case "-":
        setTime({ delta: -50 });
        break;
    }
  } else {
    if(e.key == ".")
      startTime = undefined;
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

  update(0);

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
