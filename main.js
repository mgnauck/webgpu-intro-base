"use strict";

const FULLSCREEN = false;
const ASPECT = 1.6;
const CANVAS_WIDTH = 400 * ASPECT;
const CANVAS_HEIGHT = 400;

const vertexShader = `
@vertex
fn main( @builtin(vertex_index) vertex_index : u32 ) -> @builtin(position) vec4<f32>
{  
	var pos = array<vec2<f32>, 4>( vec2(-1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(1.0, -1.0) );
	return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}
`;

const audioShader = `
struct Uniforms
{
	resolution : vec2<f32>,
	sample_rate : f32,
}
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

const BPM : f32 = 160.0;
const PI : f32 = 3.141592654;
const TAU : f32 = 6.283185307;

fn time_to_beat( t : f32 ) -> f32
{
	return t / 60.0 * BPM;
}

fn beat_to_time( b : f32 ) -> f32
{
	return b / BPM * 60.0;
}

fn sine( phase : f32 ) -> f32
{
	return sin( TAU * phase );
}

fn rand( co : vec2<f32> ) -> f32
{
	return fract( sin( dot( co, vec2<f32>( 12.9898, 78.233 ) ) ) * 43758.5453 );
}

fn noise( phase : f32 ) -> vec4<f32>
{
 	let uv : vec2<f32> = phase / vec2<f32>( 0.512, 0.487 );
    return vec4<f32>( rand( uv ) );	
}

fn kick( time : f32 ) -> f32
{
	let amp : f32 = exp( -5.0 * time );
	let phase : f32 = 60.0 * time - 15.0 * exp( -60.0 * time );
	return amp * sine( phase );
}

fn hi_hat( time : f32 ) -> vec2<f32>
{
	let amp : f32 = exp( -40.0 * time );
	return amp * noise( time * 110.0 ).xy;
}

@fragment
fn main( @builtin(position) position : vec4<f32> ) -> @location(0) vec4<f32>
{
	let time : f32 = ( position.x - 0.5 + uniforms.resolution.x * ( position.y - 0.5 ) ) / uniforms.sample_rate;
	let beat : f32 = time_to_beat( time );
	var res : vec2<f32> = vec2<f32>( 0.0 );

	// Kick
	res += 0.6 * kick( beat_to_time( beat % 1.0 ) );

	// Hihat
	res += 0.3 * hi_hat( beat_to_time( ( beat + 0.5 ) % 1.0 ) );

	return vec4<f32>( clamp( res, vec2<f32>( 0.0 ), vec2<f32>( 1.0 ) ), 0.0, 1.0 );
}
`;

const videoShaderFile = "http://localhost:8000/fragmentShader.c";
const videoShader = `
	TODO
`;

let device;
let canvas, context, presentationFormat;
let pipeline;
let renderPassDescriptor;
let uniformBindGroup, uniformBuffer;

let reloadData;
let start;

function setupCanvasAndContext()
{
	// Canvas
	document.body.innerHTML = "<button>CLICK<canvas style='width:0;cursor:none'>";
	canvas = document.querySelector( "canvas" );
	canvas.width = CANVAS_WIDTH;
	canvas.height = CANVAS_HEIGHT;

	// Contex
	context = canvas.getContext( "webgpu" );

	// Expecting format "bgra8unorm"
	presentationFormat = navigator.gpu.getPreferredCanvasFormat();

	// Link gpu and canvas
	context.configure( {
		device: device,
		format: presentationFormat,
		alphaMode: "opaque"
	} );
}

function setupPipeline( vertexShaderCode, fragmentShaderCode, presentationFormat )
{
	// Define render pipeline
	pipeline = device.createRenderPipeline(
		{
			layout: "auto",
			vertex:
				{
					module: device.createShaderModule(
						{
							code: vertexShaderCode
						}),
					entryPoint: "main"
				},
			fragment:
				{
					module: device.createShaderModule(
						{
							code: fragmentShaderCode
						}),
					entryPoint: "main",
					targets:
						[
							{
								format: presentationFormat,
							},
						],
				},
			primitive:
				{
					topology: "triangle-strip",
				},
		});
}

function setupRenderPassDescriptor()
{
	// Render pass descriptor
	renderPassDescriptor =
	{
		colorAttachments: [
			{
				view: undefined, // Assign during frame update
				//clearValue: { r: 0.3, g: 0.3, b: 0.6, a: 1.0 },
				loadOp: "clear",
				storeOp: "store",
			}]
	};
}

function setupUniformBindGroup()
{
	uniformBuffer = device.createBuffer(
		{
    		size: 4 * 4, // 4 floats
    		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  		} );

	uniformBindGroup = device.createBindGroup(
		{
    		layout: pipeline.getBindGroupLayout(0),
    		entries: [
	      		{
					binding: 0,
					resource:
					{
						buffer: uniformBuffer,
					},
				},
			],
		} );
}

function writeUniformData( data )
{
	let uniformData = new Float32Array( data );
	device.queue.writeBuffer( uniformBuffer, 0, uniformData.buffer, uniformData.byteOffset, uniformData.byteLength );
}

function setupShaderReload( url, reloadData, timeout )
{
	setInterval( async function()
		{
    		const response = await fetch( url );
    		const data = await response.text();

			if( data !== reloadData )
			{
				setupPipeline( vertexShader, data, presentationFormat );
				setupUniformBindGroup();

				reloadData = data;

				console.log("Reloaded");
			}

		}, timeout );
}

function render( time )
{
	if( start === undefined )
	{
		start = time;

		setupShaderReload( videoShaderFile, reloadData, 1000 );
	}

	const elapsed = time - start;

	// Update uniform buffer
	writeUniformData( [ CANVAS_WIDTH, CANVAS_HEIGHT, elapsed, 0.0 ] );

	// Texture view of current swap chain texture from context
	renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();

	// Command encoder
	const commandEncoder = device.createCommandEncoder();

	// Encode pass
	const passEncoder = commandEncoder.beginRenderPass( renderPassDescriptor );
	passEncoder.setPipeline( pipeline );
	passEncoder.setBindGroup( 0, uniformBindGroup );
	passEncoder.draw( 4 );
	passEncoder.end();

	// Submit command buffer
	device.queue.submit( [commandEncoder.finish()] );

	// Request next frame
	requestAnimationFrame( render );
}	

async function main()
{
	// Check if window.navigator.gpu is available, so we can use WebGPU
	if ( !navigator.gpu )
	{
		throw new Error( "WebGPU is not supported on this browser" );
	}

	// Default gpu adapter
	const gpuAdapter = await navigator.gpu.requestAdapter();
	if ( !gpuAdapter )
	{
		throw new Error( "Can not use WebGPU. No GPU adapter available." );
	}

	// Default logical gpu device
	device = await gpuAdapter.requestDevice();
	// TODO check device lost

	setupCanvasAndContext();
	setupPipeline( vertexShader, audioShader, presentationFormat );
	setupRenderPassDescriptor();
	setupUniformBindGroup();
	writeUniformData( [ CANVAS_WIDTH, CANVAS_HEIGHT, 44100.0, 0.0 ] );

	if( FULLSCREEN )
	{
		// Event listener for click to fullscreen (if required) and render start
		document.querySelector( "button" ).addEventListener( "click", e =>
			{
				canvas.requestFullscreen();
				requestAnimationFrame( render );
			} );
	}
	else
	{	
		canvas.style.width = CANVAS_WIDTH;
		canvas.style.height = CANVAS_HEIGHT;
		canvas.style.position = "absolute";
		canvas.style.left = 0;
		canvas.style.top = 0;

		requestAnimationFrame( render );
	}
}

main();