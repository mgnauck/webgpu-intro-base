const FULLSCREEN = false;
const AUDIO = true;
const BPM = 120;

const DISABLE_RENDERING = false;
const AUDIO_RELOAD_INTERVAL = 0; // Reload interval in seconds, 0 = disabled
const AUDIO_SHADER_FILE = "audio.wgsl";

const IDLE = true;
const RECORDING = true;

const ASPECT = 1.6;
const CANVAS_WIDTH = 1024;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;
const FOV = 50.0;

const AUDIO_BUFFER_SIZE = 4096 * 4096;

const MAX_GRID_RES = 128;
const SIMULATION_UPDATE_STEPS = 4;
const DEFAULT_UPDATE_DELAY = 1;
const GLOBAL_CAM_SPEED = 5;
const DEFAULT_CAM_SPEED = 1;

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

let right, up, dir, eye;
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
let activeRuleSet = -1;
let activeSimulationEventIndex = -1;
let activeCameraEventIndex = -1;

let cameraReference;

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
  "UNUSED",
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
  { time: 0, obj: { ruleSet: 3, gridRes: 128, seed: 1846359466, area: 40 } },
  { time: 20, obj: { ruleSet: -1 } },
  { time: 30, obj: { ruleSet: 2, delta: -0.5 } },
  { time: 50, obj: { delta: 1 } },
];

const CAMERA_EVENTS = [
  { time: 0, obj: { eye: [133.37, 134.02, 133.47], dirPhi: -2.2577, dirTheta: 2.1886 } }, // CAMERA_EVENT
  { time: 5, obj: { eye: [-21.47, 118.41, -7.01], dirPhi: -0.5561, dirTheta: 1.0594, moveEyePhi: -0.6081, moveEyeTheta: 0.9807 } }, // CAMERA_EVENT
  { time: 20, obj: { eye: [122.11, 27.74, 129.76], dirPhi: 2.6738, dirTheta: 2.3647, moveEyePhi: 0.0000, moveEyeTheta: 3.0183 } }, // CAMERA_EVENT
  { time: 50, obj: { eye: [133.37, 134.02, 133.47], dirPhi: -2.2577, dirTheta: 2.1886 } }, // CAMERA_EVENT
];

// 46x10
const LOGO = [
1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
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

  const shaderInfo = await shaderModule.getCompilationInfo();
  shaderInfo.messages.forEach((m) => console.log(`${m.type} (${m.lineNum}): ${m.message}`)); 

  let pipeline = await createComputePipeline(shaderModule, audioPipelineLayout, "audioMain");

  let commandEncoder = device.createCommandEncoder();

  let count = Math.ceil(Math.cbrt(AUDIO_BUFFER_SIZE) / 4);
  encodeComputePassAndSubmit(commandEncoder, pipeline, audioBindGroup, count, count, count);

  commandEncoder.copyBufferToBuffer(audioBuffer, 0, audioReadBuffer, 0, AUDIO_BUFFER_SIZE * 2 * 4);

  device.queue.submit([commandEncoder.finish()]);

  await audioReadBuffer.mapAsync(GPUMapMode.READ);
  let audioData = new Float32Array(audioReadBuffer.getMappedRange());

  let channel0 = webAudioBuffer.getChannelData(0); // right
  let channel1 = webAudioBuffer.getChannelData(1); // left

  for(let i=0; i<AUDIO_BUFFER_SIZE; i++) {
    channel0[i] = audioData[(i << 1) + 0];
    channel1[i] = audioData[(i << 1) + 1];
  }

  audioReadBuffer.unmap();

  const renderDuration = performance.now() - renderStartTime;

  console.log("Rendered audio, duration: " + renderDuration.toFixed(2) + " ms");
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
 
  audioBufferSourceNode.start(0, timeInBeats / BPM / 60);
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
  time = (AUDIO ? audioContext.currentTime : time);
  if(startTime === undefined) {
    startTime = time;
    console.log("Set startTime: " + startTime);
  }

  // Intro uses time in beats everywhere
  timeInBeats = (time - startTime) * BPM / 60;

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
        lastSimulationUpdateTime = timeInBeats;
      } else {
        console.log("Simulation update not finished within time budget.");
      }
    }
  } else {
    gridBufferUpdateOffset = 0;
    lastSimulationUpdateTime = timeInBeats;
  }

  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    ...right,
    0.5 / Math.tan(0.5 * FOV * Math.PI / 180.0),
    ...up,
    timeInBeats,
    ...dir,
    Math.abs(activeRuleSet) - 1,
    ...eye,
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
    let cam = CAMERA_EVENTS[activeCameraEventIndex];
    let deltaTime = (timeInBeats - cam.time) / (activeCameraEventIndex + 1 < CAMERA_EVENTS.length ? (CAMERA_EVENTS[activeCameraEventIndex + 1].time - cam.time) : 30);
    setView(vec3Add(cam.obj.eye, vec3Scale(cam.obj.moveEye, deltaTime * cam.obj.speed * GLOBAL_CAM_SPEED)), calcUnsteadyDir(cam.obj.dir, cam.obj.unsteady));
  }
}

function calcUnsteadyDir(dir, amp)
{
  let unsteady = vec3Normalize(vec3Add(dir, vec3Scale(
    [ 0.4 * Math.cos(1.3 * timeInBeats + Math.sin(timeInBeats * 0.3)),
      Math.pow(0.3 * Math.cos(timeInBeats * 0.4), 3.0),
      0.5 * Math.cos(timeInBeats * 1.3 + Math.cos(timeInBeats))], 0.001 * amp)));
  return unsteady;
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

  const center = gridRes / 2;
  
  /*
  const min = center - obj.area;
  const max = center + obj.area;

  for(let k=min; k<max; k++)
    for(let j=min; j<max; j++)
      for(let i=min; i<max; i++)
        grid[4 + gridRes * gridRes * k + gridRes * j + i] = rand() > 0.7 ? 1 : 0;
  */
 
  // Logo
  for(let k=0; k<12; k++)
    for(let j=0; j<10; j++)
      for(let i=0; i<46; i++)
        grid[4 + gridRes * gridRes * (center - 6 + k) + gridRes * (center - 5 + j) + (center - 23 + i)] = LOGO[46 * (9 - j) + i];

  device.queue.writeBuffer(gridBuffer[0], 0, grid);
  device.queue.writeBuffer(gridBuffer[1], 0, grid);

  recordGridEvent(obj);
}

function setRules(obj)
{
  if(obj.ruleSet !== undefined) {
    activeRuleSet = (obj.ruleSet == -1) ? -activeRuleSet : obj.ruleSet; // -1 toggles current rules set between active/inactive
    if(activeRuleSet > 0) {
      let rulesBitsBigInt = RULES[activeRuleSet];
      // State count (bit 0-3)
      rules[0] = Number(rulesBitsBigInt & BigInt(0xf));
      // Alive bits (4-31), birth bits (32-59)
      for(let i=0; i<rules.length - 1; i++)
        rules[1 + i] = Number((rulesBitsBigInt >> BigInt(4 + i)) & BigInt(0x1));
      device.queue.writeBuffer(rulesBuffer, 0, rules);
    }
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

function setView(e, f)
{
  eye = e;
  dir = f;
  right = vec3Normalize(vec3Cross(dir, [0, 1, 0]));
  up = vec3Cross(right, dir);
}

function recordGridEvent(obj)
{
  console.log(`{ time: ${timeInBeats.toFixed(2)}, obj: { gridRes: ${gridRes}, seed: ${seed}, area: ${obj.area} } }, // GRID_EVENT`);
}

function recordRulesEvent(obj)
{
  console.log(`{ time: ${timeInBeats.toFixed(2)}, obj: { ruleSet: ${obj.ruleSet} } }, // RULE_EVENT (${RULES_NAMES[Math.abs(activeRuleSet)]})`);
}

function recordTimeEvent(obj)
{
  console.log(`{ time: ${timeInBeats.toFixed(2)}, obj: { delta: ${obj.delta.toFixed(2)} } }, // TIME_EVENT (updateDelay: ${updateDelay})`);
}

function recordCameraEvent(obj)
{
  let optional = "";

  if(obj.moveEye !== undefined)
    optional += `, moveEyePhi: ${Math.atan2(obj.moveEye[1], obj.moveEye[0]).toFixed(4)}, moveEyeTheta: ${Math.acos(obj.moveEye[2]).toFixed(4)}`;

  if(obj.speed !== undefined)
    optional += `, speed: ${obj.speed}`;

  console.log(`{ time: ${timeInBeats.toFixed(2)}, obj: { eye: [${obj.eye[0].toFixed(2)}, ${obj.eye[1].toFixed(2)}, ${obj.eye[2].toFixed(2)}], ` +
    `dirPhi: ${Math.atan2(obj.dir[1], obj.dir[0]).toFixed(4)}, dirTheta: ${Math.acos(obj.dir[2]).toFixed(4)}${optional} } }, // CAMERA_EVENT`);
}

function completeCameras()
{
  for(let i=0; i<CAMERA_EVENTS.length; i++) {
    let obj = CAMERA_EVENTS[i].obj;
    CAMERA_EVENTS[i].obj.dir = vec3FromSpherical(obj.dirTheta, obj.dirPhi)
    CAMERA_EVENTS[i].obj.moveEye = obj.moveEyeTheta !== undefined ? vec3FromSpherical(obj.moveEyeTheta, obj.moveEyePhi) : [0, 0, 0];
    CAMERA_EVENTS[i].obj.speed = obj.speed !== undefined ? obj.speed : 1;
    CAMERA_EVENTS[i].obj.unsteady = obj.unsteady !== undefined ? obj.unsteady : 1;
  }
}

function resetView()
{
  setView([gridRes, gridRes, gridRes], vec3Normalize(vec3Add(vec3Scale([gridRes, gridRes, gridRes], 0.5), vec3Negate([gridRes, gridRes, gridRes]))));
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

function vec3FromSpherical(theta, phi)
{
  return [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)];
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

function handleCameraControlEvent(e)
{
  switch (e.key) {
    case "a":
      setView(vec3Add(eye, vec3Scale(right, -MOVE_VELOCITY)), dir);
      break;
    case "d":
      setView(vec3Add(eye, vec3Scale(right, MOVE_VELOCITY)), dir);
      break;
    case "w":
      setView(vec3Add(eye, vec3Scale(dir, MOVE_VELOCITY)), dir);
      break;
    case "s":
      setView(vec3Add(eye, vec3Scale(dir, -MOVE_VELOCITY)), dir);
      break;
  } 
}

async function handleKeyEvent(e)
{    
  // Rule sets 1-10 (keys 0-9)
  if(e.key != " " && !isNaN(e.key))
  {
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
      // Global idle, intro time and simulation time are paused
      idle = !idle;
      if(idle) {
        if(AUDIO)
          await suspendAudio();
      } else {
        if(AUDIO)
          await playAudio();
      }
      console.log("Idle mode " + (idle ? "enabled" : "disabled"));
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
      setRules({ ruleSet: -1 }); // Toggle rule set, i.e. pause/run simulation
      break;
    case ".":
      // Records a STATIC cam
      recordCameraEvent({ eye: eye, dir: dir }); 
      break;
    case "m":
      // Set start pose of moving cam
      cameraReference = { eye: eye, dir: dir };
      console.log("Camera reference pinned");
      break;
    case ",":
      if(cameraReference !== undefined) {
        // Record start and end pose of moving cam
        cameraReference.moveEye = vec3Normalize(vec3Add(eye, vec3Negate(cameraReference.eye)));
        recordCameraEvent(cameraReference);
        cameraReference = undefined;
      }
      break;
    case ":":
      seed = undefined;
      console.log("Reset random seed");
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
      activeRuleSet = -1;
      activeSimulationEventIndex = -1;
      activeCameraEventIndex = -1;
      if(AUDIO && !idle)
        await playAudio();
      break;
  }
}

function handleMouseMoveEvent(e)
{
  let yaw = -e.movementX * LOOK_VELOCITY;
  let pitch = -e.movementY * LOOK_VELOCITY;

  const currentPitch = Math.acos(dir[1]);
  const newPitch = currentPitch - pitch;
  const minPitch = Math.PI / 180.0;
  const maxPitch = 179.0 * Math.PI / 180.0;

  if(newPitch < minPitch)
    pitch = currentPitch - minPitch;

  if(newPitch > maxPitch)
    pitch = currentPitch - maxPitch;

  setView(eye, vec3Transform(vec3Transform(dir, axisRotation(right, pitch)), axisRotation([0, 1, 0], yaw)));
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

  completeCameras();
  resetView();
  updateSimulation();
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

  device.lost.then((info) => {
    console.error(`WebGPU device was lost: ${info.message}`);
    device = null;
    if (info.reason !== "destroyed") { // Only reason as per spec is 'destroyed' at the moment
      // Re-initialize
      //main();
    }
  });

  if(AUDIO) {
    await createAudioResources();
    await renderAudio();
  }

  // Grid mul + buffer update z-offset + grid
  grid = new Uint32Array(4 + MAX_GRID_RES * MAX_GRID_RES * MAX_GRID_RES);
  
  // State count + alive rules + birth rules
  rules = new Uint32Array(1 + 2 * 27);

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
