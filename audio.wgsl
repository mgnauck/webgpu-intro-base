struct AudioParameters
{
  bufferDim: u32, // Cubic root of audio buffer size to match grid dimension of compute shader invocations
  sampleRate: u32 // Sample rate as per WebAudio context
}

const BPM = 120.0;
const PI = 3.141592654;
const TAU = 6.283185307;
const TIME_PER_BEAT = 60.0 / BPM / 4.0;
const TIME_PER_PATTERN = 60.0 / BPM * 4.0;
const PATTERN_COUNT = 6;
const PATTERN_ROW_COUNT = 16;
const KICK = 0;
const HIHAT = 1;
const BASS = 2;
const DURCH = 3;

struct Note 
{
  note: i32,
  instr: i32,
  amp: f32
}

struct Row 
{
  note1: Note,
  note2: Note,
  note3: Note,
  note4: Note
}

const NoteOff = Note(-1, -1, 0.0);
const RowOff = Row(NoteOff, NoteOff, NoteOff, NoteOff);

// Suboptimal random (ripped from somewhere)
fn rand(co: vec2f) -> f32
{
  return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn noise(phase: f32) -> vec4f
{
  let uv = phase / vec2f(0.512, 0.487);
  return vec4f(rand(uv));
}

// TODO: translate into function
// https://graphtoy.com/?f1(x,t)=max(0,min(x/1,max(0.5,1-(x-1)*(1-0.5)/1)*min(1,1-max(0,(x-1-1-1)/1))))&v1=true&f2(x,t)=&v2=false&f3(x,t)=&v3=false&f4(x,t)=&v4=false&f5(x,t)=&v5=false&f6(x,t)=&v6=false&grid=1&coords=2.0657836566653387,-0.2538551861155832,3.823569812524288
fn adsr(time: f32, att: f32, dec: f32, sus: f32, susL: f32, rel : f32) -> f32
{
  return 
    max(  0.0,
          min( 
            smoothstep(0.0, 1.0, time / att),
            max( 
              susL, 
              smoothstep(0.0, 1.0, 1.0 - (time - att) * (1.0 - susL) / dec)) *
              min(1.0, 1.0 - max(0.0, (time - att - dec - sus) / rel)
            )
          )
    );
}

//////////////////////// INSTRUMENTS 

fn bass(time: f32, freq: f32) -> f32
{
  let phase = freq * time; 
  let env = exp(-3.0 * time);
  let bass = atan2( sin(TAU * phase), 1.0-0.25);

  return bass * env;
}

fn hihat(time: f32, freq: f32) -> f32
{
  let dist = 0.75;
  let out = noise(time * freq).x;
  let env = exp(-90.0 * time);
  let hihat = atan2(out, 1.0-dist);
  return hihat * env; 
}

// inspired by:
// https://www.shadertoy.com/view/7lpatternIndexczz
fn kick(time: f32, freq: f32) -> f32
{
  let dist = 0.65;
  let phase = freq * time - 8.0 * exp(-20.0 * time) - 3.0 * exp(-800.0 * time);
  let env = exp(-4.0 * time);
  let kick = atan2(sin(TAU * phase), 1.0 - dist);

  return kick * env;
}

fn clap(time: f32, freq : f32) -> f32
{
  // TODO
  let clap = 0.0;
  let env = 0.0;

  return clap * env;
}

fn simple(time: f32, freq: f32) -> f32
{
  let lfo = 0.25 * sin(TAU * time * 0.01);
  let phase = time * (freq + lfo);
  let phase2 = time * freq;
  let env = exp(-2.0 * time);
  let o1 = sin(TAU * phase);
  let o2 = 2 * (fract(phase) - 0.5);

  return o1 * o2 * env;
}

fn simple2(time: f32, freq: f32) -> f32
{
  let c = 0.2;
  let r = 0.3;
  
  var v0 = 0.0;
  var v1 = 0.0;
  const cnt = 12;
  for(var i=0;i<cnt;i++)
  {
    let last = f32(cnt-i)*(1.0/f32(params.sampleRate));
    let t = time - last;
    let inp = simple(t, freq);
    v0 = (1.0-r*c)*v0  -  (c)*v1  + (c)*inp;
    v1 = (1.0-r*c)*v1  +  (c)*v0;
  }

  return v1;
}

// convert a note to it's frequency representation
fn noteToFreq(note: f32) -> f32
{
  return 440.0 * pow(2.0, (note - 69.0) / 12.0);
}

const PATTERN_1 = array<Row, PATTERN_ROW_COUNT>
( 
  Row(Note(33,KICK,0.5), Note(33,HIHAT,0.25), Note(33,DURCH,0.15), NoteOff),
  RowOff, 
  Row(NoteOff, Note(33,HIHAT,0.15), Note(33+12,DURCH,0.20), Note(33,BASS,0.20)),
  RowOff,
  Row(NoteOff, Note(33,HIHAT,0.25), Note(33+24,DURCH,0.25), Note(33,BASS,0.20)),
  RowOff, 
  Row(Note(33,KICK,0.3), Note(33,HIHAT,0.15), NoteOff, Note(33,BASS,0.10)),
  RowOff,
  Row(NoteOff, Note(33,HIHAT,0.25), NoteOff, NoteOff),
  RowOff, 
  Row(NoteOff, Note(33,HIHAT,0.15), Note(33+24,DURCH,0.1), Note(33,BASS,0.20)),
  RowOff,
  Row(Note(33,KICK,0.4), Note(33,HIHAT,0.25), NoteOff, Note(33,BASS,0.20)),
  RowOff, 
  Row(NoteOff, Note(33,HIHAT,0.15), Note(33+24,DURCH,0.02), Note(33,BASS,0.20)),
  Row(NoteOff, Note(33,HIHAT,0.05), NoteOff, NoteOff),
);

const PATTERN_2 = array<Row, PATTERN_ROW_COUNT>
( 
  Row(Note(33,KICK,0.5), Note(33,HIHAT,0.25), Note(33,BASS,0.20), NoteOff),
  RowOff, 
  Row(NoteOff, Note(33,HIHAT,0.15), NoteOff, NoteOff),
  RowOff,
  Row(NoteOff, Note(33,HIHAT,0.25), NoteOff, NoteOff),
  RowOff, 
  Row(Note(33,KICK,0.4), Note(33,HIHAT,0.15), Note(33,BASS,0.20), NoteOff),
  RowOff,
  Row(NoteOff, Note(33,HIHAT,0.25), NoteOff, NoteOff),
  RowOff, 
  Row(NoteOff, Note(33,HIHAT,0.15), Note(33+36-7,DURCH,0.2), NoteOff),
  Row(NoteOff, NoteOff, Note(33+36-7-7,DURCH,0.1), NoteOff),
  Row(Note(33,KICK,0.4), Note(33,HIHAT,0.25), Note(33+24,DURCH,0.2), Note(33+12,BASS,0.20)),
  RowOff, 
  Row(NoteOff, Note(33,HIHAT,0.15), Note(33,DURCH,0.1), NoteOff),
  Row(NoteOff, Note(33,HIHAT,0.05), NoteOff, NoteOff),
);

const PATTERNS = array<array<Row, PATTERN_ROW_COUNT>, PATTERN_COUNT> 
( 
  PATTERN_1,
  PATTERN_2,
  PATTERN_1,
  PATTERN_2,
  PATTERN_1,
  PATTERN_2,
);

fn sine(time: f32, freq: f32) -> f32
{
  return sin(time * TAU * freq);
}

fn pling(time: f32, freq: f32) -> f32
{
  return sin(time * TAU * freq) * exp(-1.0*time);
}

fn sqr(time: f32, freq: f32) -> f32
{
  return (1.0 - 0.5) * 2.0 * atan(sin(time*freq) / 0.5) / PI;
}

fn sample1(time: f32, freq: f32) -> f32
{
  let lfo = sin(time * TAU * 0.1) * 1.0;
  let lfo2 = 0.5; // 1.0 + 0.5 * sin(time * TAU * 1) * 0.1;

  let voices = 13.0;
  var out = 0.0;
  for(var v=1.0;v<=voices;v+=1.0)
  {
    let detune = sin((time + v) * TAU * 0.25) * 0.25;
    out += 1.0/voices * sine(time, (freq+detune+lfo)*v);
  }

  out = atan2(out, 1.0-lfo2);
  let env = exp(-3.0*time);

  return out * env;
}

fn sample2(time: f32, freq: f32) -> f32
{
  return sample1(time, freq);
}

@group(0) @binding(0) var<uniform> params: AudioParameters;
@group(0) @binding(1) var<storage, read_write> buffer: array<vec2f>;

@compute @workgroup_size(4, 4, 4)
fn audioMain(@builtin(global_invocation_id) globalId: vec3u)
{
  // Make sure workgroups align properly with buffer size, i.e. do not run beyond buffer dimension
  if(globalId.x >= params.bufferDim || globalId.y >= params.bufferDim || globalId.z >= params.bufferDim) {
    return;
  }

  // Calculate current sample from given buffer id
  let sample = dot(globalId, vec3u(1, params.bufferDim, params.bufferDim * params.bufferDim));
  let time = f32(sample) / f32(params.sampleRate);
  let musicTime = time % (TIME_PER_PATTERN * PATTERN_COUNT);

  // Samples are calculated in mono and then written to left/right
  var output = vec2(0.0);

  for(var patternIndex=0; patternIndex<PATTERN_COUNT; patternIndex++)
  {
    let pattern = PATTERNS[patternIndex];

    for(var rowIndex=0; rowIndex<PATTERN_ROW_COUNT; rowIndex++)
    {
      let patternTime = f32(patternIndex) * TIME_PER_PATTERN + f32(rowIndex) * TIME_PER_BEAT;
      let noteTime = musicTime - patternTime;
      let row = pattern[rowIndex];
      let notes = array<Note, 4>(row.note1, row.note2, row.note3, row.note4);

      for(var noteIndex=0; noteIndex<4; noteIndex++)
      {
        let note = notes[noteIndex];

        if(note.note > -1 && note.instr > -1 && noteTime > 0.0)
        {
          let noteFreq = noteToFreq(f32(note.note));
          var instrOutput = 0.0;

          if(note.instr == KICK)
          { 
            instrOutput = kick(noteTime, noteFreq);
          }
          else if(note.instr == HIHAT)
          {
            instrOutput = hihat(noteTime, noteFreq);
          }
          else if(note.instr == BASS)
          {
            instrOutput = bass(noteTime, noteFreq);
          }
          else if(note.instr == DURCH)
          {
            instrOutput = sample1(noteTime, noteFreq);
          }

          output += instrOutput * note.amp;
        }
      }
    }
  }

  // Write 2 floats between -1 and 1 to output buffer (stereo)
  buffer[sample] = clamp(output, vec2f(-1), vec2f(1));
}
