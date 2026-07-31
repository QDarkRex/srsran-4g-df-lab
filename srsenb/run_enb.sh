#!/bin/bash

# Build directory
BUILD_DIR="../build"

echo "Building srsenb..."
if make -C "$BUILD_DIR" -j$(nproc); then
    echo "Build successful! Starting srsenb loop..."
else
    echo "Build failed. Starting loop with existing binary..."
fi

# Run loop
while true; do
    echo "Starting srsenb..."
    sudo "$BUILD_DIR/srsenb/src/srsenb" enb.conf
    echo "srsenb exited with status $?. Restarting in 1 second (Press Ctrl+C to stop)..."
    sleep 1
done
