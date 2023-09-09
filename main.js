const FULLSCREEN = false;
const AUDIO = false;

const RECORDING = false;
const RECORDING_OFS = 0;
const START_RECORDING_AT = -1;
const OVERVIEW_CAMERA = false;

const ASPECT = 1.6;
const CANVAS_WIDTH = 1024;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;
const FOV = 50.0;

const AUDIO_BUFFER_SIZE = Math.pow(256, 3); // 4096*4096

const MAX_GRID_RES = 128;
const DEFAULT_UPDATE_DELAY = 200;

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

const RULES = [
  2023103542460421n, // clouds-5
  34359738629n, // 4/4-5
  97240207056901n, // amoeba-5
  962072678154n, // pyro-10
  36507219973n, // framework-5
  96793530464266n, // spiky-10
  1821066142730n, // builder-10
  96793530462218n, // ripple-10
  2216617588948994n, // stable-2
  30064771210n, // pulse-10
];

const RULES_NAMES = [
  "clouds-5", // key '0'
  "4/4-5", // key '1'
  "amoeba-5", // key '2'
  "pyro-10", // ..
  "framework-5",
  "spiky-10",
  "builder-10",
  "ripple-10",
  "stable-2",
  "pulse-10",
];

const GRID_EVENTS = [
{ step: 0, obj: { gridRes: 128, seed: 1474531643, area: 12 } }, // GRID_EVENT
];

const RULE_EVENTS = [
{ step: 0, obj: { ruleSet: 4 } },
];

const TIME_EVENTS = [
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

  let webAudioBuffer = audioContext.createBuffer(2, AUDIO_BUFFER_SIZE, audioContext.sampleRate);

  console.log("Max audio length: " + (webAudioBuffer.length / audioContext.sampleRate / 60).toFixed(2) + " min");

  let audioBuffer = device.createBuffer({
    // Size * stereo * sizeof(float)
    size: AUDIO_BUFFER_SIZE * 2 * 4, 
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

  let bindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0, 
      visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}
    }]});

  let bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{
      binding: 0,
      resource: {buffer: audioBuffer}
    }]});

  let readBuffer = device.createBuffer({
    size: AUDIO_BUFFER_SIZE * 2 * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});

  let shaderModule = device.createShaderModule({code: await loadTextFile("audioShader.wgsl")});
  let pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});

  let commandEncoder = device.createCommandEncoder();
  
  let count = Math.ceil(256 / 4);
  encodeComputePassAndSubmit(commandEncoder,
    await createComputePipeline(shaderModule, pipelineLayout, "audioMain"),
    bindGroup, count, count, count);

  commandEncoder.copyBufferToBuffer(audioBuffer, 0, readBuffer, 0, AUDIO_BUFFER_SIZE * 2 * 4);

  device.queue.submit([commandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  let audioData = new Float32Array(readBuffer.getMappedRange());

  let channel0 = webAudioBuffer.getChannelData(0);
  let channel1 = webAudioBuffer.getChannelData(1);

  for (let i = 0; i < AUDIO_BUFFER_SIZE; i++) {
    channel0[i] = audioData[(i << 1) + 0];
    channel1[i] = audioData[(i << 1) + 1];
  }

  readBuffer.unmap();

  audioBufferSourceNode = audioContext.createBufferSource();
  audioBufferSourceNode.buffer = webAudioBuffer;
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

  for(let i=0; i<2; i++)
    gridBuffer[i] = device.createBuffer({
      size: grid.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  rulesBuffer = device.createBuffer({
    size: rules.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

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
  if(startTime === undefined)
    startTime = AUDIO ? (audioContext.currentTime * 1000.0) : time;

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
  } else {
    // Keep cam unsteady
    let t = (currTime + simulationStep) * 0.00125;
    let unsteady = vec3Normalize(vec3Add(fwd, vec3Scale([0.4 * Math.cos(1.3 * t + Math.sin(t * 0.3)), Math.pow(0.3 * Math.cos(t * 0.4), 3.0), 0.5 * Math.cos(t * 1.3 + Math.cos(t))], 0.0007)));
    setView(eye, unsteady);
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

  gridRes = Math.min(obj.gridRes, MAX_GRID_RES); // Safety

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
  let rulesBitsBigInt = RULES[obj.ruleSet];

  // State count (bit 0-3)
  rules[0] = Number(rulesBitsBigInt & BigInt(0xf));

  // Alive bits (4-31), birth bits (32-59)
  for(let i=0; i<rules.length - 1; i++)
    rules[1 + i] = Number((rulesBitsBigInt >> BigInt(4 + i)) & BigInt(0x1));

  device.queue.writeBuffer(rulesBuffer, 0, rules);

  console.log(`{ step: ${(RECORDING_OFS + simulationStep)}, obj: { ruleSet: ${obj.ruleSet} } }, // RULE_EVENT (${RULES_NAMES[obj.ruleSet]})`);
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
    if(e.key == ".") {
      startTime = undefined;
      lastSimulationUpdateTime = 0;
      simulationPaused = false;
      simulationStep = 0;
      previousSimulationStep = -1;
      simulationIteration = 0; 
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

  if(!overviewCamera)
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

  // Grid mul + grid
  grid = new Uint32Array(3 + MAX_GRID_RES * MAX_GRID_RES * MAX_GRID_RES);
  
  // State count + alive rules + birth rules
  rules = new Uint32Array(1 + 2 * 27);

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
