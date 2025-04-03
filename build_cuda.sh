#!/bin/bash
# Build script for the TTS CUDA kernels

echo "Compiling optimized TTS CUDA kernels..."

# Set CUDA path if not already set
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        echo "Set CUDA_HOME to /usr/local/cuda"
    else
        echo "Error: CUDA_HOME not set and couldn't find CUDA in /usr/local/cuda"
        echo "Please set CUDA_HOME environment variable to your CUDA installation directory"
        exit 1
    fi
fi

# Check if CUDA is installed
if [ ! -d "$CUDA_HOME" ]; then
    echo "Error: CUDA not found at $CUDA_HOME"
    exit 1
fi

echo "Using CUDA from: $CUDA_HOME"

# Detect GPU architecture
GPU_ARCH=""
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    echo "Detected GPU: $GPU_NAME"
    
    # Map GPU names to CUDA architectures
    if [[ "$GPU_NAME" == *"A100"* ]]; then
        GPU_ARCH=80
    elif [[ "$GPU_NAME" == *"RTX 40"* ]]; then
        GPU_ARCH=89
    elif [[ "$GPU_NAME" == *"RTX 30"* ]]; then
        GPU_ARCH=86
    elif [[ "$GPU_NAME" == *"RTX 20"* ]]; then
        GPU_ARCH=75
    elif [[ "$GPU_NAME" == *"V100"* ]]; then
        GPU_ARCH=70
    elif [[ "$GPU_NAME" == *"P100"* ]]; then
        GPU_ARCH=60
    elif [[ "$GPU_NAME" == *"T4"* ]]; then
        GPU_ARCH=75
    else
        # Default architecture for unknown GPUs
        GPU_ARCH=70
        echo "Warning: Unknown GPU architecture, defaulting to compute capability 70"
    fi
    
    echo "Using CUDA architecture: compute_$GPU_ARCH"
else
    echo "Warning: nvidia-smi not found, cannot detect GPU architecture"
    echo "Defaulting to compute capability 70"
    GPU_ARCH=70
fi

# Build the kernels
make GPU_ARCH=$GPU_ARCH clean
make GPU_ARCH=$GPU_ARCH -j$(nproc)

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "You can now run the optimized TTS engine with: python test_optimized_tts.py"
else
    echo "Build failed. Please check the error messages above."
    exit 1
fi

# Set library path if needed
echo "To use the CUDA kernels, make sure the library path is set:"
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)" 
