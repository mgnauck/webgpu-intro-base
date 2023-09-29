const FULLSCREEN = false;

const CANVAS_WIDTH = 1024; // Careful, this is also hardcoded in the shader!!
const CANVAS_HEIGHT = CANVAS_WIDTH / 1.77;

let audioContext;
let audioBufferSourceNode;

let device;
let uniformBuffer;
let rulesBuffer;
let bindGroup = [];
let computePipeline;
let renderPipeline;
let renderPassDescriptor;

let canvas;
let context;

let startTime;
let lastSimulationUpdateTime = 0;
let simulationIteration = 0;
let activeScene = -1;

const RULES = new Uint32Array([
   5,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0, // clouds-5 / 0
   5,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 4/4-5 / 1
   5,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0, // amoeba-5 / 2
  10,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // pyro-10 / 3
   5,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // framework-5 / 4
  10,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0, // spiky-10 / 5
  10,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0, // ripple-10 / 6
]);

// Rule set indices are -1 in player compared to main!!
const SCENES = [
  { t: 0, r: 2, d: 0.2, p: 40 }, // amoeba
  { t: 40, r: 3, d: 0.5, p: 320 }, // pyro
  { t: 60, r: 2, d: 0.6, p: 220 }, // amoeba
  { t: 80, r: 0, d: 1.0, p: 180  }, // clouds
  { t: 110, r: 6, d: 0.75, p: 160 }, // ripple
  { t: 150, r: 3, d: 0.875, p: 180 }, // pyro (trim down)
  { t: 155, r: 4, d: 0.25, p: 170 }, // framework
  { t: 190, r: 5, d: 0.25, p: 160 }, // spiky
  { t: 220, r: 1, d: 0.375, p: 140 }, // 445
  { t: 300, r: 1, d: 0.375, p: 180 }
];

const AUDIO_SHADER = `
REPLACE_ME_AUDIO
`;

const VISUAL_SHADER = `
REPLACE_ME_VISUAL
`;

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

async function createComputePipeline(shaderModule, pipelineLayout)
{
  return device.createComputePipelineAsync({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: "cM"
    }
  });
}

async function createRenderPipeline(shaderModule, pipelineLayout)
{
  return device.createRenderPipelineAsync({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vM"
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fM",
      targets: [{format: "bgra8unorm"}]
    },
    primitive: {topology: "triangle-strip"}
  });
}

function encodeComputePassAndSubmit(commandEncoder, pipeline, bindGroup)
{
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(64, 64, 64);
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
  let webAudioBuffer = audioContext.createBuffer(2, 256 ** 3, audioContext.sampleRate);

  let audioBuffer = device.createBuffer({
    size: (256 ** 3) * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

  // Will be reused by visual shader
  uniformBuffer = device.createBuffer({
    size: 4 * 4, // We actually only need two floats for audio uniforms
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  let audioBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}
    ]});

  let audioBindGroup = device.createBindGroup({
    layout: audioBindGroupLayout,
    entries: [
      {binding: 0, resource: {buffer: uniformBuffer}},
      {binding: 1, resource: {buffer: audioBuffer}}
    ]});

  let audioReadBuffer = device.createBuffer({
    size: (256 ** 3) * 8,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});

  let audioPipelineLayout = device.createPipelineLayout({bindGroupLayouts: [audioBindGroupLayout]});

  device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([audioContext.sampleRate]));

  let pipeline = await createComputePipeline(device.createShaderModule({code: AUDIO_SHADER}), audioPipelineLayout);

  let commandEncoder = device.createCommandEncoder();

  encodeComputePassAndSubmit(commandEncoder, pipeline, audioBindGroup);

  commandEncoder.copyBufferToBuffer(audioBuffer, 0, audioReadBuffer, 0, (256 ** 3) * 8);

  device.queue.submit([commandEncoder.finish()]);

  await audioReadBuffer.mapAsync(GPUMapMode.READ);
  let audioData = new Float32Array(audioReadBuffer.getMappedRange());

  let channel0 = webAudioBuffer.getChannelData(0);
  let channel1 = webAudioBuffer.getChannelData(1);

  for(let i=0; i<256 ** 3; i++) {
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
      {binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
      {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
      {binding: 3, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
    ]
  });

  let gridBuffer = [];
  for(let i=0; i<2; i++)
    gridBuffer[i] = device.createBuffer({
      size: (256 ** 3) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  rulesBuffer = device.createBuffer({
    size: 385 * 4, //RULES.length * 4,
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

  let pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});

  renderPassDescriptor = {
    colorAttachments: [{
      undefined, // view
      //clearValue: {r: 1.0, g: 0.0, b: 0.0, a: 1.0},
      loadOp: "clear",
      storeOp: "store"
    }]
  };

  let shaderModule = device.createShaderModule({code: VISUAL_SHADER});
  computePipeline = await createComputePipeline(shaderModule, pipelineLayout);
  renderPipeline = await createRenderPipeline(shaderModule, pipelineLayout);

  // Set grid
  let rand = xorshift32(4079287172);
  const area = 24;
  const pos = 128 - area / 2;
  let grid = new Uint32Array(area);
  for(let k=0; k<area; k++) {
    for(let j=0; j<area; j++) { 
      for(let i=0; i<area; i++)
        grid[i] = rand() > 0.6 ? 1 : 0;
      let ofs = (256 ** 2) * (pos + k) + 256 * (pos + j) + pos;
      device.queue.writeBuffer(gridBuffer[0], ofs * 4, grid);
      device.queue.writeBuffer(gridBuffer[1], ofs * 4, grid);
    }
  }
}

let last;

function render(time)
{  
  if(last !== undefined) {
    let frameTime = (performance.now() - last);
    document.title = `${(frameTime).toFixed(2)} / ${(1000.0 / frameTime).toFixed(2)}`;
  }
  last = performance.now();

  // Initialize time and start audio
  if(startTime === undefined) {
    audioBufferSourceNode.start(0, 0);
    startTime = audioContext.currentTime;
  }

  // Current time
  let timeInBeats = (audioContext.currentTime - startTime) * 2; // BPM / 60
  if(timeInBeats >= 300) //SCENES.at(-1).t
    return;

  // Scene update
  if(timeInBeats >= SCENES[activeScene + 1].t)
    device.queue.writeBuffer(rulesBuffer, 0, RULES, SCENES[ ++activeScene ].r * 55, 55);

  // Current scene time
  let curr = SCENES[activeScene];
  let next = SCENES[activeScene + 1];
  let t = (timeInBeats - curr.t) / (next.t - curr.t);

  const commandEncoder = device.createCommandEncoder();
  
  // Simulation
  if(timeInBeats - lastSimulationUpdateTime > SCENES[activeScene].d) {
    encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup[simulationIteration % 2]); 
    simulationIteration++;
    lastSimulationUpdateTime = (audioContext.currentTime - startTime) * 2; // BPM / 60
  }

  // Camera
  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    curr.p + (next.p - curr.p) * t, // radius
    ((activeScene % 2) ? 1 : -1) * t * 2 * Math.PI, // phi
    (0.8 + 0.3 * Math.sin(timeInBeats * 0.2)) * Math.sin(timeInBeats * 0.05), // theta
    timeInBeats]));

  // Render
  renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
  encodeRenderPassAndSubmit(commandEncoder, renderPassDescriptor, renderPipeline, bindGroup[simulationIteration % 2]);
  
  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);
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

  document.body.innerHTML = "<button>CLICK<canvas style='width:0;cursor:none'>";
  canvas = document.querySelector("canvas");
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  context = canvas.getContext("webgpu");
  context.configure({device, format: "bgra8unorm", alphaMode: "opaque"});

  document.querySelector("button").addEventListener("click", startRender);
}

main();
