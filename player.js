const FULLSCREEN = false;
const BPM = 120;

const CANVAS_WIDTH = 1024; // Careful, this is also hardcoded in the shader!!
const CANVAS_HEIGHT = CANVAS_WIDTH / 1.6;

const BUFFER_DIM = 256; // Used for audio buffer and grid buffer

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

let view = [];
let grid = new Uint32Array(3 + (BUFFER_DIM ** 3));
let rules = new Uint32Array(1 + 2 * 27);
let updateDelay = 0.5;

let startTime;
let timeInBeats = 0;
let lastSimulationUpdateTime = 0;
let simulationIteration = 0;
let activeRuleSet;
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

const SIMULATION_EVENTS = [
{ t: 0, r: 3, d: -0.320 },
{ t: 40, r: 4, d: 0.320 },
{ t: 60, r: 3, d: 0.05 },
{ t: 80, r: 1, d: 0.125 },
{ t: 120, r: 8, d: -0.130 },
{ t: 180, r: -8 },
];

const CAMERA_EVENTS = [
{ t: 0, v: [ 42, 1.5708, 0.0000 ] },
{ t: 40, v: [ 320, -3.7292, 0.7250 ] },
{ t: 60, v: [ 240, -4.4042, -0.7000 ] },
{ t: 80, v: [ 200, -5.7792, 0.8000 ] },
{ t: 120, v: [ 170, -2.7960, -0.7000 ] },
{ t: 180, v: [ 220, -1.3600, 0.5000] },
];

const AUDIO_SHADER = `
REPLACE_ME_AUDIO
`;

const VISUAL_SHADER = `
REPLACE_ME_VISUAL
`;

// https://github.com/bryc/code/blob/master/jshash/PRNGs.md
function splitmix32(a) {
  return function() {
    a |= 0; a = a + 0x9e3779b9 | 0;
    var t = a ^ a >>> 16; t = Math.imul(t, 0x21f0aaad);
    t = t ^ t >>> 15; t = Math.imul(t, 0x735a2d97);
    return ((t = t ^ t >>> 15) >>> 0) / 4294967296;
  }
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

function encodeComputePassAndSubmit(commandEncoder, pipeline, bindGroup, count)
{
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(count, count, count);
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
  webAudioBuffer = audioContext.createBuffer(2, BUFFER_DIM ** 3, audioContext.sampleRate);

  let audioBuffer = device.createBuffer({
    size: (BUFFER_DIM ** 3) * 2 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

  let audioUniformBuffer = device.createBuffer({
    size: 2 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});

  let audioBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}
    ]});

  let audioBindGroup = device.createBindGroup({
    layout: audioBindGroupLayout,
    entries: [
      {binding: 0, resource: {buffer: audioUniformBuffer}},
      {binding: 1, resource: {buffer: audioBuffer}}
    ]});

  let audioReadBuffer = device.createBuffer({
    size: (BUFFER_DIM ** 3) * 2 * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});

  let audioPipelineLayout = device.createPipelineLayout({bindGroupLayouts: [audioBindGroupLayout]});

  device.queue.writeBuffer(audioUniformBuffer, 0, new Uint32Array([BUFFER_DIM, audioContext.sampleRate]));

  let pipeline = await createComputePipeline(device.createShaderModule({code: AUDIO_SHADER}), audioPipelineLayout, "audioMain");

  let commandEncoder = device.createCommandEncoder();

  encodeComputePassAndSubmit(commandEncoder, pipeline, audioBindGroup, BUFFER_DIM / 4);

  commandEncoder.copyBufferToBuffer(audioBuffer, 0, audioReadBuffer, 0, (BUFFER_DIM ** 3) * 2 * 4);

  device.queue.submit([commandEncoder.finish()]);

  await audioReadBuffer.mapAsync(GPUMapMode.READ);
  let audioData = new Float32Array(audioReadBuffer.getMappedRange());

  let channel0 = webAudioBuffer.getChannelData(0);
  let channel1 = webAudioBuffer.getChannelData(1);

  for(let i=0; i<BUFFER_DIM ** 3; i++) {
    channel0[i] = audioData[(i << 1) + 0];
    channel1[i] = audioData[(i << 1) + 1];
  }

  audioReadBuffer.unmap();

  audioBufferSourceNode = audioContext.createBufferSource();
  audioBufferSourceNode.buffer = webAudioBuffer;
  audioBufferSourceNode.connect(audioContext.destination); 
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
 
  uniformBuffer = device.createBuffer({
    size: 4 * 4, // radius, phi, theta, time
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

  let shaderModule = device.createShaderModule({code: VISUAL_SHADER});
  computePipeline = await createComputePipeline(shaderModule, pipelineLayout, "computeMain");
  renderPipeline = await createRenderPipeline(shaderModule, pipelineLayout, "vertexMain", "fragmentMain");
}

let last;

function render(time)
{  
  if(last !== undefined) {
    let frameTime = (performance.now() - last);
    document.title = `${(frameTime).toFixed(2)} / ${(1000.0 / frameTime).toFixed(2)}`;
  }
  last = performance.now();

  if(startTime === undefined) {
    audioBufferSourceNode.start(0, 0);
    startTime = audioContext.currentTime;
  }

  timeInBeats = (audioContext.currentTime - startTime) * BPM / 60;

  const commandEncoder = device.createCommandEncoder();
  
  updateSimulation();
  updateCamera();
 
  if(activeRuleSet > 0) {
    if(timeInBeats - lastSimulationUpdateTime > updateDelay) {
      encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup[simulationIteration % 2], BUFFER_DIM / 4); 
      simulationIteration++;
      lastSimulationUpdateTime = (audioContext.currentTime - startTime) * BPM / 60;
    }
  } else
    lastSimulationUpdateTime = timeInBeats;

  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([...view, timeInBeats]));

  renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
  encodeRenderPassAndSubmit(commandEncoder, renderPassDescriptor, renderPipeline, bindGroup[simulationIteration % 2]);
  
  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);
}

function updateSimulation()
{
  if(activeSimulationEventIndex + 1 < SIMULATION_EVENTS.length && timeInBeats >= SIMULATION_EVENTS[activeSimulationEventIndex + 1].t) {
    let e = SIMULATION_EVENTS[++activeSimulationEventIndex];

    // Rules
    if(e.r !== undefined) {
      activeRuleSet = e.r; // Can be active (positive) or inactive (negative), zero is excluded by definition
      let rulesBitsBigInt = RULES[Math.abs(activeRuleSet)];
      // State count (bit 0-3)
      rules[0] = Number(rulesBitsBigInt & BigInt(0xf));
      // Alive bits (4-31), birth bits (32-59)
      for(let i=0; i<rules.length - 1; i++)
        rules[1 + i] = Number((rulesBitsBigInt >> BigInt(4 + i)) & BigInt(0x1));
      device.queue.writeBuffer(rulesBuffer, 0, rules);
    }

    // Time
    updateDelay = (e.d === undefined) ? updateDelay : updateDelay + e.d;
  }
}

function updateCamera()
{
  if(activeCameraEventIndex + 1 < CAMERA_EVENTS.length && timeInBeats >= CAMERA_EVENTS[activeCameraEventIndex + 1].t)
    ++activeCameraEventIndex;

  if(activeCameraEventIndex >= 0 && activeCameraEventIndex + 1 < CAMERA_EVENTS.length) {
    let curr = CAMERA_EVENTS[activeCameraEventIndex];
    let next = CAMERA_EVENTS[activeCameraEventIndex + 1];
    let t = (timeInBeats - curr.t) / (next.t - curr.t);
    for(let i=0; i<3; i++)
      view[i] = curr.v[i] + (next.v[i] - curr.v[i]) * t;
  }
}

function setGrid(area)
{
  for(let i=0; i<grid.length; i++)
    grid[i] = 0;

  grid[0] = 1;
  grid[1] = BUFFER_DIM;
  grid[2] = BUFFER_DIM ** 2;

  const center = BUFFER_DIM * 0.5;
  const d = area * 0.5;

  let rand = splitmix32(4079287172);

  // TODO Make initial grid somewhat more interesting
  for(let k=center - d; k<center + d; k++)
    for(let j=center - d; j<center + d; j++)
      for(let i=center - d; i<center + d; i++)
        grid[3 + (BUFFER_DIM ** 2) * k + BUFFER_DIM * j + i] = rand() > 0.6 ? 1 : 0;

  device.queue.writeBuffer(gridBuffer[0], 0, grid);
  device.queue.writeBuffer(gridBuffer[1], 0, grid);
}

function startRender()
{
  document.querySelector("button").removeEventListener("click", startRender);

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
  updateCamera();
  requestAnimationFrame(render);
}

async function main()
{
  if(!navigator.gpu)
    alert("No WebGPU");

  const gpuAdapter = await navigator.gpu.requestAdapter();
  device = await gpuAdapter.requestDevice();

  await createAudioResources();
  await createRenderResources();
  setGrid(24);

  document.body.innerHTML = "<button>CLICK<canvas style='width:0;cursor:none'>";
  canvas = document.querySelector("canvas");
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  context = canvas.getContext("webgpu");
  context.configure({device, format: "bgra8unorm", alphaMode: "opaque"});

  document.querySelector("button").addEventListener("click", startRender);
}

main();
