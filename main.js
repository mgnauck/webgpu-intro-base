const FULLSCREEN = false;
const AUDIO = false;

const ASPECT = 1.6;
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;

const AUDIO_WIDTH = 4096;
const AUDIO_HEIGHT = 4096;

const GRID_RES = 128.0;

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

function encodeComputePassAndSubmit(pipeline, bindGroup, workgroupCountX, workgroupCountY, workgroupCountZ, preSubmitOperation)
{
  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
  passEncoder.end();

  if (preSubmitOperation != undefined) {
    preSubmitOperation(commandEncoder);
  }

  device.queue.submit([commandEncoder.finish()]);
}

function encodeRenderPassAndSubmit(passDescriptor, pipeline, bindGroup)
{
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginRenderPass(passDescriptor);
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.draw(4);
  passEncoder.end();

  device.queue.submit([commandEncoder.finish()]);
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

  setupPerformanceTimer("Render audio");

  let shaderModule = device.createShaderModule({code: await loadTextFile("audioShader.wgsl")});
  let pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});

  encodeComputePassAndSubmit(
      await createComputePipeline(shaderModule, pipelineLayout),
      bindGroup, Math.ceil(AUDIO_WIDTH / 8), Math.ceil(AUDIO_HEIGHT / 8), 1,
      function(commandEncoder) {
        commandEncoder.copyTextureToBuffer(
            {texture: audioTexture},
            {buffer: readBuffer, bytesPerRow: AUDIO_WIDTH * 2 * 4},
            [AUDIO_WIDTH, AUDIO_HEIGHT]);
      });

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
 
  // 4*vec3f, grid res, time, programmable value
  uniformBuffer = device.createBuffer({
    size: 16 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});

  for(let i=0; i<2; i++) {
    gridBuffer[i] = device.createBuffer({
      size: GRID_RES * GRID_RES * GRID_RES * 4,
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
  if (audioContext === undefined && start === undefined) {
    start = time;
    lastUpdate = time;
  }

  if(time - lastUpdate > 2500) {
    let workgroupSize = Math.ceil(GRID_RES / 4);
    encodeComputePassAndSubmit(computePipeline, bindGroup[simulationSteps % 2], workgroupSize, workgroupSize, workgroupSize);
    simulationSteps++;
    lastUpdate = time;
  }

  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    ...right,
    GRID_RES,
    ...up,
    50.0, // fov
    ...fwd,
    AUDIO ? audioContext.currentTime : ((time - start) / 1000.0),
    ...eye,
    programmableValue
  ]));

  setupPerformanceTimer("Render");

  renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
  encodeRenderPassAndSubmit(renderPassDescriptor, renderPipeline, bindGroup[simulationSteps % 2]);

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

function resetView()
{
  eye = [0, 0, GRID_RES + 5.0];
  fwd = [0, 0, -1];

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

  let grid = new Uint32Array(GRID_RES * GRID_RES * GRID_RES);
  for(let j=0; j<grid.length; j++)
    grid[j] = Math.random() > 0.9 ? 1 : 0;
    
  device.queue.writeBuffer(gridBuffer[0], 0, grid);
  device.queue.writeBuffer(gridBuffer[1], 0, grid);

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

  setInterval(createPipelines, 500); // Reload shader
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

  if (AUDIO) {
    document.querySelector("button").addEventListener("click", startRender);
  } else {
    startRender();
  }
}

main();
