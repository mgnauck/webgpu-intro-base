const FULLSCREEN = false;
const AUDIO = true; // AudioContext has the best timer, so better leave it enabled.
const BPM = 120;

const DISABLE_RENDERING = false;
const AUDIO_RELOAD_INTERVAL = 0; // Reload interval in seconds, 0 = disabled
const AUDIO_SHADER_FILE = "audio.wgsl";

const IDLE = false;
const RECORDING = false;
const STOP_REPLAY_AT = 300;

const ASPECT = 1.77;
const CANVAS_WIDTH = 1024; // Careful, this is also hardcoded in the shader!!
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;

const AUDIO_BUFFER_SIZE = 4096 * 4096;

const MAX_GRID_RES = 256;

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

let view = []; // radius, phi, theta
let programmableValue;

let rand;
let gridRes;
let updateDelay;
let grid;

let startTime;
let timeInBeats = 0;
let lastSimulationUpdateTime = 0;
let simulationIteration = 0;
let activeRuleSet = 3;
let activeScene = -1;

const RULES = new Uint32Array([
   5,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0, // clouds-5 / 1
   5,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 4/4-5 / 2
   5,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0, // amoeba-5 / 3
  10,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // pyro-10 / 4
   5,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // framework-5 / 5
  10,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0, // spiky-10 / 6
  10,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0, // ripple-10 / 7
  10,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // builder-10 / 8, unused
  7,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,1,0,1,0,0,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,1,0,1,0,0,0,0,1,0,0, // shells-7 / 9, unused
  10,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // pulse-10 / 10, unused
  5,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // more-builds-5 / 11, unused
]);

const RULES_NAMES = [
  "U-N-U-S-E-D",
  "clouds-5",
  "4/4-5",
  "amoeba-5",
  "pyro-10",
  "framework-5",
  "spiky-10",
  "ripple-10",
  "builder-10", // unused
  "shells-7", // unused
  "pulse-10", // unused
  "more-builds-5", // unused
];

// Rule set indices are +1 in main compared to player!!
const SCENES = [
  { t: 0, r: 3, d: 0.2, p: 40 }, // amoeba
  { t: 40, r: 4, d: 0.5, p: 320 }, // pyro
  { t: 60, r: 3, d: 0.6, p: 220 }, // amoeba
  { t: 80, r: 1, d: 1.0, p: 180  }, // clouds
  { t: 110, r: 7, d: 0.75, p: 160 }, // ripple
  { t: 150, r: 4, d: 0.875, p: 180 }, // pyro (trim down)
  { t: 155, r: 5, d: 0.25, p: 170 }, // framework
  { t: 190, r: 6, d: 0.25, p: 160 }, // spiky
  { t: 220, r: 2, d: 0.375, p: 140 }, // 445
  { t: 300, r: 2, d: 0.375, p: 180 }
];

// https://github.com/bryc/code/blob/master/jshash/PRNGs.md
function xorshift32(a)
{
  return function()
  {
    a ^= a << 13;
    a ^= a >>> 17;
    a ^= a << 5;
    return (a >>> 0) / 4294967296;
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
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  let audioBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
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

  device.queue.writeBuffer(audioUniformBuffer, 0, new Uint32Array([audioContext.sampleRate]));
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
      {binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
      {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
      {binding: 3, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
    ]
  });
 
  uniformBuffer = device.createBuffer({
    size: 4 * 4, // radius, phi, theta, time 
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  for(let i=0; i<2; i++)
    gridBuffer[i] = device.createBuffer({
      size: grid.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  rulesBuffer = device.createBuffer({
    size: RULES.length * 4,
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

  computePipeline = await createComputePipeline(shaderModule, pipelineLayout, "cM");
  renderPipeline = await createRenderPipeline(shaderModule, pipelineLayout, "vM", "fM");
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
    update();
  }

  if(!idle && activeRuleSet > 0) {
    if(timeInBeats - lastSimulationUpdateTime > updateDelay) {
      const count = Math.ceil(gridRes / 4);
      encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup[simulationIteration % 2], count, count, count); 
      simulationIteration++;
      lastSimulationUpdateTime = ((AUDIO ? audioContext.currentTime : time / 1000.0) - startTime) * BPM / 60;
    }
  } else
    lastSimulationUpdateTime = timeInBeats;

  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([...view, timeInBeats]));

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
  if(activeScene + 1 < SCENES.length) {
    if(timeInBeats >= SCENES[activeScene + 1].t) {
      let s = SCENES[++activeScene];
    
      if(s.r !== undefined)
        setRules(s.r);
    
      if(s.d !== undefined)
        setTime(s.d);

      view[0] = s.p;
      recordCameraEvent();
    }

    let curr = SCENES[activeScene];
    let next = SCENES[activeScene + 1];
    let t = (timeInBeats - curr.t) / (next.t - curr.t);
    view[0] = curr.p + (next.p - curr.p) * t;
    view[1] = ((activeScene % 2) ? 1 : -1) * t * 2 * Math.PI;
    view[2] = (0.9 + 0.3 * Math.sin(timeInBeats * 0.2)) * Math.sin(timeInBeats * 0.05);
  }
}

function resetView()
{
  view = [gridRes * 0.5, Math.PI * 0.5, 0];
}

function setGrid(area, prob, seed)
{
  for(let i=0; i<grid.length; i++)
    grid[i] = 0;

  gridRes = MAX_GRID_RES;

  const center = gridRes * 0.5;
  const d = Math.ceil(area / 2);

  rand = xorshift32(seed);

  for(let k=center - d; k<center + d; k++)
    for(let j=center - d; j<center + d; j++)
      for(let i=center - d; i<center + d; i++)
        grid[(gridRes ** 2) * k + gridRes * j + i] =  rand() > prob ? 1 : 0;

  device.queue.writeBuffer(gridBuffer[0], 0, grid);
  device.queue.writeBuffer(gridBuffer[1], 0, grid);
}

function setRules(r)
{
  activeRuleSet = r; // Can be active (positive) or inactive (negative), zero is excluded by definition
  
  device.queue.writeBuffer(rulesBuffer, 0, RULES, (Math.abs(activeRuleSet) - 1) * 55, 55);
 
  recordRulesEvent(r);
}

function setTime(d)
{
  updateDelay = d;
  
  recordTimeEvent(d);
}

function recordRulesEvent(r)
{
  console.log(`{ t: ${timeInBeats.toFixed(2)}, r: ${r} }, // ${RULES_NAMES[Math.abs(r)]}`);
}

function recordTimeEvent(d)
{
  console.log(`{ t: ${timeInBeats.toFixed(2)}, d: ${d.toFixed(2)} },`);
}

function recordCameraEvent()
{
  console.log(`{ t: ${timeInBeats.toFixed(2)}, p: ${view[0].toFixed(1)} },`);
}

async function handleKeyEvent(e)
{    
  // Rule sets 1-10 (keys 0-9)
  if(e.key != " " && !isNaN(e.key)) {
    setRules(parseInt(e.key, 10) + 1);
    return;
  }

  // Rule set 11 (key =)
  if(e.key == "=") {
    setRules(11);
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
    case "i":
      let seed = rand() * 4294967296;
      console.log("Initialized new grid with seed " + seed);
      setGrid(22, 0.7, seed);
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
    case "+":
      setTime(updateDelay + 0.25);
      break;
    case "-":
      if(updateDelay > 0)
        setTime(updateDelay - 0.25);
      break;
    case "#":
      setRules(-activeRuleSet); // Enable or disable activity of current rule set (pos/neg)
      break;
    case ".":
      recordCameraEvent();
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
      startTime = undefined;
      timeInBeats = 0;
      lastSimulationUpdateTime = 0;
      simulationIteration = 0;
      activeRuleSet = 3;
      activeScene = -1;
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
      view[0] += MOVE_VELOCITY;
      break;
    case "w":
      view[0] = Math.max(view[0] - MOVE_VELOCITY, 0.01);
      break;
  }
}

function handleMouseMoveEvent(e)
{
  view[1] = (view[1] + e.movementX * LOOK_VELOCITY) % (2.0 * Math.PI);
  view[2] = Math.min(Math.max(view[2] + e.movementY * LOOK_VELOCITY, -1.5), 1.5);
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

  // Grid mul + grid
  grid = new Uint32Array(MAX_GRID_RES * MAX_GRID_RES * MAX_GRID_RES);
  
  await createRenderResources();
  await createPipelines();
  setGrid(24, 0.6, 4079287172);

  if(idle || recording)
    resetView();
  update();
 
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
