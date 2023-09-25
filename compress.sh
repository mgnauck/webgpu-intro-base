#!/bin/bash

# Assumes wgslminify and js-payload-compress available in current dir

infile=$1
infile_ext="${infile##*.}"
infile_name="${infile%.*}"

output_dir=output

shader_excludes=$2

rm -rf $output_dir
mkdir $output_dir
cp $infile $output_dir

./wgslminify -e $shader_excludes audio.wgsl > $output_dir/audio_minified.wgsl
./wgslminify -e $shader_excludes visual.wgsl > $output_dir/visual_minified.wgsl

cd $output_dir

sed -f ../compress.sed $infile > ${infile_name}_with_shaders.${infile_ext}

terser ${infile_name}_with_shaders.${infile_ext} --mangle --rename --toplevel --c toplevel,passes=5,unsafe=true,pure_getters=true > ${infile_name}_minified.${infile_ext}

../js-payload-compress ${infile_name}_minified.${infile_ext} ${infile_name}_compressed.html

# Example:
# ./compress.sh player.js audioMain,computeMain,vertexMain,fragmentMain
