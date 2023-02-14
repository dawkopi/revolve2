#!/bin/bash
# creates a "virtual screen" for purposes of rendering videos
# this file based on https://github.com/Farama-Foundation/Gymnasium/blob/9d2b8310ada39b86bb83cbe0cb55f5b61084e27f/bin/docker_entrypoint

set -ex

sudo apt install -y xvfb

# Set up display; otherwise rendering will fail
Xvfb -screen 0 1920x1080x24 &
export DISPLAY=:0

# Wait for the file to come up
display=0
file="/tmp/.X11-unix/X$display"
for i in $(seq 1 10); do
    if [ -e "$file" ]; then
        break
    fi

    echo "Waiting for $file to be created (try $i/10)"
    sleep "$i"
done
if ! [ -e "$file" ]; then
    echo "Timing out: $file was not created"
    exit 1
fi