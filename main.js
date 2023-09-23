const FULLSCREEN = false;
const AUDIO = true; // AudioContext has the best timer, so better leave it enabled.
const BPM = 120;

const DISABLE_RENDERING = false;
const AUDIO_RELOAD_INTERVAL = 0; // Reload interval in seconds, 0 = disabled
const AUDIO_SHADER_FILE = "audio.wgsl";

const IDLE = false;
const RECORDING = false;
const STOP_REPLAY_AT = 180;

const ASPECT = 1.6;
const CANVAS_WIDTH = 1024; // Careful, this is also hardcoded in the shader!!
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;
const FOV = 50.0;

const AUDIO_BUFFER_SIZE = 4096 * 4096;

const MAX_GRID_RES = 256;
const SIMULATION_UPDATE_STEPS = 4;
const DEFAULT_UPDATE_DELAY = 0.5;

const MOVE_VELOCITY = 0.75;
const LOOK_VELOCITY = 0.025;
const WHEEL_VELOCITY = 0.005;

let idle = IDLE;
let recording = RECORDING;

let audioContext;
let webAudioBuffer;
let audioBuffer;
let audioReadBuffer;
let audioBindgroup;
let audioPipelineLayout;
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

let radius, phi, theta;
let programmableValue;

let seed;
let rand;
let gridRes;
let updateDelay = DEFAULT_UPDATE_DELAY;
let grid;
let rules;

let startTime;
let timeInBeats = 0;
let lastSimulationUpdateTime = 0;
let simulationIteration = 0;
let gridBufferUpdateOffset = 0;
let activeRuleSet = 3;
let activeSimulationEventIndex = -1;
let activeCameraEventIndex = -1;

const RULES = [
  0, // not in use, leave it here, we need it for enable/disable magic
  2023103542460421n, // clouds-5, key 0
  34359738629n, // 4/4-5, key 1
  97240207056901n, // amoeba-5, key 2
  962072678154n, // pyro-10, key 3
  36507219973n, // framework-5, key 4
  96793530464266n, // spiky-10, key 5
  1821066142730n, // builder-10, key 6
  96793530462218n, // ripple-10, key 7
  37688665960915591n, // shells-7, key 8
  30064771210n, // pulse-10, key 9
  4294970885n, // more-builds-5, key )
];

const RULES_NAMES = [
  "U-N-U-S-E-D",
  "clouds-5",
  "4/4-5",
  "amoeba-5",
  "pyro-10",
  "framework-5",
  "spiky-10",
  "builder-10",
  "ripple-10",
  "shells-7",
  "pulse-10",
  "more-builds-5",
];

const SIMULATION_EVENTS = [
{ time: 0, obj: { ruleSet: 3, delta: -0.320, seed: 4079287172, gridRes: MAX_GRID_RES, area: 24 } },
{ time: 40, obj: { ruleSet: 4, delta: 0.320 } },
{ time: 60, obj: { ruleSet: 3, delta: 0.05 } },
{ time: 80, obj: { ruleSet: 1, delta: 0.125 } },
{ time: 120, obj: { ruleSet: 8, delta: -0.130 } },
{ time: 180, obj: { ruleSet: -8 } },
];

const CAMERA_EVENTS = [
{ time: 0, obj: [ 42, 1.5708, 0.0000 ] },
{ time: 40, obj: [ 320, -3.7292, 0.7250 ] },
{ time: 60, obj: [ 240, -4.4042, -0.7000 ] },
{ time: 80, obj: [ 200, -5.7792, 0.8000 ] },
{ time: 120, obj: [ 170, -2.7960, -0.7000 ] },
{ time: 180, obj: [ 220, -1.3600, 0.5000] },
];

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

async function createAudioResources()
{
  audioContext = new AudioContext();
  console.log("Audio context sample rate: " + audioContext.sampleRate);

  webAudioBuffer = audioContext.createBuffer(2, AUDIO_BUFFER_SIZE, audioContext.sampleRate);
  console.log("Max audio length: " + (webAudioBuffer.length / audioContext.sampleRate / 60).toFixed(2) + " min");

  audioBuffer = device.createBuffer({
    // Buffer size * stereo * 4 bytes (float/uint32)
    size: AUDIO_BUFFER_SIZE * 2 * 4, 
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

  let audioUniformBuffer = device.createBuffer({
    // (buffer dimension + sample rate) * 4 bytes (uint32)
    size: 2 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});

  let audioBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}
    ]});

  audioBindGroup = device.createBindGroup({
    layout: audioBindGroupLayout,
    entries: [
      {binding: 0, resource: {buffer: audioUniformBuffer}},
      {binding: 1, resource: {buffer: audioBuffer}}
    ]});

  audioReadBuffer = device.createBuffer({
    size: AUDIO_BUFFER_SIZE * 2 * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});

  audioPipelineLayout = device.createPipelineLayout({bindGroupLayouts: [audioBindGroupLayout]});

  device.queue.writeBuffer(audioUniformBuffer, 0, new Uint32Array([Math.ceil(Math.cbrt(AUDIO_BUFFER_SIZE)), audioContext.sampleRate]));
}

async function renderAudio()
{
  const renderStartTime = performance.now();

  let shaderCode = await loadTextFile(AUDIO_SHADER_FILE);
  let shaderModule = device.createShaderModule({code: shaderCode});

  let pipeline = await createComputePipeline(shaderModule, audioPipelineLayout, "audioMain");

  let commandEncoder = device.createCommandEncoder();

  let count = Math.ceil(Math.cbrt(AUDIO_BUFFER_SIZE) / 4);
  encodeComputePassAndSubmit(commandEncoder, pipeline, audioBindGroup, count, count, count);

  commandEncoder.copyBufferToBuffer(audioBuffer, 0, audioReadBuffer, 0, AUDIO_BUFFER_SIZE * 2 * 4);

  device.queue.submit([commandEncoder.finish()]);

  await audioReadBuffer.mapAsync(GPUMapMode.READ);
  let audioData = new Float32Array(audioReadBuffer.getMappedRange());

  let channel0 = webAudioBuffer.getChannelData(0);
  let channel1 = webAudioBuffer.getChannelData(1);

  for(let i=0; i<AUDIO_BUFFER_SIZE; i++) {
    channel0[i] = audioData[(i << 1) + 0];
    channel1[i] = audioData[(i << 1) + 1];
  }

  audioReadBuffer.unmap();

  console.log("Rendered audio, duration: " + (performance.now() - renderStartTime).toFixed(2) + " ms");
}

async function suspendAudio()
{
  if(audioContext.state === "running") {
    await audioContext.suspend();
    audioBufferSourceNode.stop();
  }
}

async function playAudio()
{
  audioBufferSourceNode = audioContext.createBufferSource();
  audioBufferSourceNode.buffer = webAudioBuffer;
  audioBufferSourceNode.connect(audioContext.destination);
 
  if(audioContext.state === "suspended")
    await audioContext.resume();
 
  audioBufferSourceNode.start(0, timeInBeats / BPM * 60);
}

async function reloadAudio()
{
  if(!idle)
    await suspendAudio();
  
  await renderAudio();
  
  if(!idle)
    await playAudio();

  console.log("Audio shader reloaded/re-rendered audio");

  if(AUDIO_RELOAD_INTERVAL > 0)
    setTimeout(reloadAudio, AUDIO_RELOAD_INTERVAL);
}

async function createRenderResources()
{
  let bindGroupLayout = device.createBindGroupLayout({
    entries: [ 
      {binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
      {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
      {binding: 3, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
    ]
  });
 
  // Right, up, dir, eye, fov, time, simulation step, programmable value/padding
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
  let shaderCode = await loadTextFile("visual.wgsl");
  let shaderModule = device.createShaderModule({code: shaderCode});

  computePipeline = await createComputePipeline(shaderModule, pipelineLayout, "computeMain");
  renderPipeline = await createRenderPipeline(shaderModule, pipelineLayout, "vertexMain", "fragmentMain");
}

let last;

function render(time)
{  
  //setPerformanceTimer();
  if(last !== undefined) {
    let frameTime = (performance.now() - last);
    document.title = `${(frameTime).toFixed(2)} / ${(1000.0 / frameTime).toFixed(2)}`;
  }
  last = performance.now();

  // Get current time
  time = (AUDIO ? audioContext.currentTime : time / 1000.0);
  if(startTime === undefined) {
    startTime = time;
    console.log("Set startTime: " + startTime);
  }

  // Intro uses time in beats everywhere
  timeInBeats = (time - startTime) * BPM / 60;

  if(!idle && !recording && timeInBeats >= STOP_REPLAY_AT) {
    handleKeyEvent(new KeyboardEvent("keydown", {key: " "}));
    handleKeyEvent(new KeyboardEvent("keydown", {key: "Enter"}));
  }

  const commandEncoder = device.createCommandEncoder();
  
  if(!idle && !recording) {
    updateSimulation();
    updateCamera();
  }
 
  if(!idle && activeRuleSet >= 0) {
    if(gridBufferUpdateOffset < gridRes) {
      const count = Math.ceil(gridRes / 4);
      device.queue.writeBuffer(gridBuffer[simulationIteration % 2], 12, new Uint32Array([gridBufferUpdateOffset]));
      encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup[simulationIteration % 2], count, count, count / SIMULATION_UPDATE_STEPS); 
      gridBufferUpdateOffset += count / SIMULATION_UPDATE_STEPS * 4;
    }
    if(timeInBeats - lastSimulationUpdateTime > updateDelay) {
      if(gridBufferUpdateOffset >= gridRes) {
        simulationIteration++;
        gridBufferUpdateOffset = 0;
        lastSimulationUpdateTime = ((AUDIO ? audioContext.currentTime : time / 1000.0) - startTime) * BPM / 60;
      } else {
        console.log("WARNING: Simulation update not finished within time budget.");
      }
    }
  } else {
    gridBufferUpdateOffset = 0;
    lastSimulationUpdateTime = timeInBeats;
  }

  let view = calcView();
  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    ...view.right,
    0.5 / Math.tan(0.5 * FOV * Math.PI / 180.0),
    ...view.up,
    timeInBeats,
    ...view.dir,
    Math.abs(activeRuleSet) - 1,
    ...view.eye,
    1.0 // free value
  ]));

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

function updateSimulation()
{
  if(activeSimulationEventIndex + 1 < SIMULATION_EVENTS.length && timeInBeats >= SIMULATION_EVENTS[activeSimulationEventIndex + 1].time) {
    let eventObj = SIMULATION_EVENTS[++activeSimulationEventIndex].obj;
    setGrid(eventObj);
    setRules(eventObj);
    setTime(eventObj);
  }
}

function updateCamera()
{
  if(activeCameraEventIndex + 1 < CAMERA_EVENTS.length && timeInBeats >= CAMERA_EVENTS[activeCameraEventIndex + 1].time)
    recordCameraEvent(CAMERA_EVENTS[++activeCameraEventIndex].obj);

  if(activeCameraEventIndex >= 0) {
    let curr = CAMERA_EVENTS[activeCameraEventIndex];
    let vals = curr.obj;
    if(activeCameraEventIndex + 1 < CAMERA_EVENTS.length) {
      let next = CAMERA_EVENTS[activeCameraEventIndex + 1];
      vals = vec3Add(vals, vec3Scale(vec3Add(next.obj, vec3Negate(vals)), (timeInBeats - curr.time) / (next.time - curr.time)));
    }
    radius = vals[0];
    phi = vals[1];
    theta = vals[2];
    // TODO Apply unsteady cam again
  }
}

function calcView()
{
  let e = [radius * Math.cos(theta) * Math.cos(phi), radius * Math.sin(theta), radius * Math.cos(theta) * Math.sin(phi)];
  let d = vec3Normalize(vec3Negate(e));
  let r = vec3Normalize(vec3Cross(d, [0, 1, 0]));  
  return { eye: e, dir: d, right: r, up: vec3Cross(r, d) };
}

function resetView()
{
  radius = gridRes * 0.5;
  phi = Math.PI * 0.5;
  theta = 0;
}

function setGrid(obj)
{
  if(obj.gridRes === undefined)
    return;

  for(let i=0; i<grid.length; i++)
    grid[i] = 0;

  if(seed === undefined || obj.seed != seed) {
    seed = obj.seed === undefined ? Math.floor(Math.random() * 4294967296) : obj.seed;
    rand = splitmix32(seed);
  }

  gridRes = Math.min(obj.gridRes, MAX_GRID_RES); // Safety

  grid[0] = 1;
  grid[1] = gridRes;
  grid[2] = gridRes * gridRes;

  // grid[3] is reserved as offset for buffer update position

  const center = gridRes * 0.5; 
  const d = obj.area * 0.5;

  for(let k=center - d; k<center + d; k++)
    for(let j=center - d; j<center + d; j++)
      for(let i=center - d; i<center + d; i++)
        grid[4 + gridRes * gridRes * k + gridRes * j + i] = rand() > 0.6 ? 1 : 0;

  device.queue.writeBuffer(gridBuffer[0], 0, grid);
  device.queue.writeBuffer(gridBuffer[1], 0, grid);

  recordGridEvent(obj);
}

function setRules(obj)
{
  if(obj.ruleSet !== undefined) {
    activeRuleSet = obj.ruleSet; // Can be active (positive) or inactive (negative), zero is excluded by definition
    let rulesBitsBigInt = RULES[Math.abs(activeRuleSet)];
    // State count (bit 0-3)
    rules[0] = Number(rulesBitsBigInt & BigInt(0xf));
    // Alive bits (4-31), birth bits (32-59)
    for(let i=0; i<rules.length - 1; i++)
      rules[1 + i] = Number((rulesBitsBigInt >> BigInt(4 + i)) & BigInt(0x1));
    device.queue.writeBuffer(rulesBuffer, 0, rules);
    recordRulesEvent(obj);
  }
}

function setTime(obj)
{
  if(obj.delta !== undefined) {
    updateDelay += obj.delta;
    recordTimeEvent(obj);
  }
}

function recordGridEvent(obj)
{
  console.log(`{ time: ${timeInBeats.toFixed(2)}, obj: { gridRes: ${gridRes}, seed: ${seed}, area: ${obj.area} } },`);
}

function recordRulesEvent(obj)
{
  console.log(`{ time: ${timeInBeats.toFixed(2)}, obj: { ruleSet: ${obj.ruleSet} } }, // ${RULES_NAMES[Math.abs(activeRuleSet)]}`);
}

function recordTimeEvent(obj)
{
  console.log(`{ time: ${timeInBeats.toFixed(2)}, obj: { delta: ${obj.delta.toFixed(2)} } }, // updateDelay: ${updateDelay.toFixed(3)}`);
}

function recordCameraEvent(obj)
{
  console.log(`{ time: ${timeInBeats.toFixed(2)}, obj: [ ${obj[0].toFixed(1)}, ${obj[1].toFixed(4)}, ${obj[2].toFixed(4)} ] },`);
}

async function handleKeyEvent(e)
{    
  // Rule sets 1-10 (keys 0-9)
  if(e.key != " " && !isNaN(e.key)) {
    setRules({ ruleSet: parseInt(e.key, 10) + 1 });
    return;
  }

  // Rule set 11 (key =)
  if(e.key == "=") {
    setRules({ ruleSet: 11 });
    return;
  }

  switch (e.key) {
    case "v":
      resetView();
      break;
    case "l":
      createPipelines();
      console.log("Visual shader reloaded");
      break;
    case "Enter":
      recording = !recording;
      console.log("Recording mode " + (recording ? "enabled" : "disabled"));
      break;
    case " ":
      // Global idle: Intro time and simulation time are paused
      idle = !idle;
      if(idle) {
        if(AUDIO)
          await suspendAudio();
      } else {
        if(AUDIO)
          await playAudio();
      }
      console.log("Idle mode " + (idle ? "enabled" : "disabled") + ", time: " + timeInBeats);
      break;
    case "i":
      setGrid({ gridRes: MAX_GRID_RES, seed: seed, area: 4 });
      break;
    case "+":
      setTime({ delta: 0.25 });
      break;
    case "-":
      if(updateDelay > 0)
        setTime({ delta: -0.25 });
      break;
    case "#":
      setRules({ ruleSet: -activeRuleSet }); // Enable or disable activity of current rule set (pos/neg)
      break;
    case ".":
      recordCameraEvent([radius, phi, theta]); 
      break;
    case ">":
      // Re-render and restart audio from last audio position
      if(AUDIO)
        reloadAudio();
      break;
    case "<":
      console.log("Reset everything to start");
      if(AUDIO) {
        if(!idle)
          await suspendAudio();
        await renderAudio();
      }
      seed = undefined;
      updateDelay = DEFAULT_UPDATE_DELAY;
      startTime = undefined;
      timeInBeats = 0;
      lastSimulationUpdateTime = 0;
      simulationIteration = 0;
      gridBufferUpdateOffset = 0;
      activeRuleSet = 3;
      activeSimulationEventIndex = -1;
      activeCameraEventIndex = -1;
      if(AUDIO && !idle)
        await playAudio();
      break;
  }
}

function handleCameraControlEvent(e)
{
  switch(e.key)
  {
    case "s":
      radius += MOVE_VELOCITY;
      break;
    case "w":
      radius = Math.max(radius - MOVE_VELOCITY, 0.01);
      break;
  }
}

function handleMouseMoveEvent(e)
{
  phi = (phi + e.movementX * LOOK_VELOCITY) % (2.0 * Math.PI);
  theta = Math.min(Math.max(theta + e.movementY * LOOK_VELOCITY, -1.5), 1.5);
}

function handleMouseWheelEvent(e)
{
  programmableValue -= e.deltaY * WHEEL_VELOCITY;
}

async function startRender()
{
  if(FULLSCREEN)
    canvas.requestFullscreen();
  else {
    canvas.style.width = CANVAS_WIDTH;
    canvas.style.height = CANVAS_HEIGHT;
    canvas.style.position = "absolute";
    canvas.style.left = 0;
    canvas.style.top = 0;
  }

  updateSimulation();
  resetView(); // In case we have no camera, i.e. when recording
  updateCamera();

  document.querySelector("button").removeEventListener("click", startRender);

  document.addEventListener("keydown", handleKeyEvent);

  canvas.addEventListener("click", async () => {
    if(!document.pointerLockElement)
      await canvas.requestPointerLock({unadjustedMovement: true});
  });

  document.addEventListener("pointerlockchange", () => {
    if(document.pointerLockElement === canvas) {
      document.addEventListener("keydown", handleCameraControlEvent);
      canvas.addEventListener("mousemove", handleMouseMoveEvent);
      canvas.addEventListener("wheel", handleMouseWheelEvent);
    } else {
      document.removeEventListener("keydown", handleCameraControlEvent);
      canvas.removeEventListener("mousemove", handleMouseMoveEvent);
      canvas.removeEventListener("wheel", handleMouseWheelEvent);
    }
  });

  if(AUDIO && !idle) {
    await playAudio();

    if(AUDIO_RELOAD_INTERVAL > 0)
      setTimeout(reloadAudio, AUDIO_RELOAD_INTERVAL);
  }

  if(!DISABLE_RENDERING)
    requestAnimationFrame(render);
}

async function main()
{
  if(!navigator.gpu)
    throw new Error("WebGPU is not supported on this browser.");

  const gpuAdapter = await navigator.gpu.requestAdapter();
  if(!gpuAdapter)
    throw new Error("Can not use WebGPU. No GPU adapter available.");

  console.log(gpuAdapter.limits);

  device = await gpuAdapter.requestDevice();
  if(!device)
    throw new Error("Failed to request logical device.");

  if(AUDIO) {
    await createAudioResources();
    await renderAudio();
  }

  // Grid mul + buffer update z-offset + grid
  grid = new Uint32Array(4 + MAX_GRID_RES * MAX_GRID_RES * MAX_GRID_RES);
  
  // State count + alive rules + birth rules
  rules = new Uint32Array(1 + 2 * 27);

  /*
  // Title buffer
  const offCanvas = new OffscreenCanvas(512, 256);
  const offContext = offCanvas.getContext("2d");
  offContext.font = "128px serif";
  offContext.fillStyle = "black";
  offContext.fillRect(0, 0, 512, 256);
  offContext.fillStyle = "white";
  offContext.rect(90, 127, 320, 2);
  offContext.fill();
  offContext.fillText("unik", 134, 116);
  offContext.translate(0, 256);
  offContext.scale(1, -1);
  offContext.fillText("elusive", 70, 114);
  */

  /*offCanvas.convertToBlob().then((blob) => {
    var blobUrl = URL.createObjectURL(blob);
    window.location.replace(blobUrl);
  });*/

  //const imageData = offContext.getImageData(0, 0, 256, 256);
  // imageData.data --> Uint8ClampedArray

  await createRenderResources();
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

  if(AUDIO || FULLSCREEN)
    document.querySelector("button").addEventListener("click", startRender);
  else
    startRender();
}

main();
