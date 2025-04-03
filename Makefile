CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc
CXX = g++

# Detect GPU architecture if not specified
ifndef GPU_ARCH
GPU_ARCH = $(shell $(NVCC) --help | grep compute_70 > /dev/null && echo 70 || echo 60)
endif

# Common flags
CXXFLAGS = -O3 -Wall -fPIC -std=c++14
NVCCFLAGS = -O3 -std=c++14 --use_fast_math --expt-relaxed_constexpr
LDFLAGS = -shared

# Include directories
INCLUDES = -I$(CUDA_HOME)/include -I.

# Define target architectures for CUDA compilation
NVCC_ARCH_FLAGS = -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

# Enable different optimization levels
release: NVCCFLAGS += -O3
debug: NVCCFLAGS += -O0 -G -g

# Library name
LIB_NAME = libseseme_tts_kernels.so

# Source files
SOURCES = seseme_tts_kernels.cu

# Default target
all: release

# Compile for release
release: $(LIB_NAME)

# Compile for debug
debug: $(LIB_NAME)

# Main library target
$(LIB_NAME): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $(NVCC_ARCH_FLAGS) $(INCLUDES) -Xcompiler "$(CXXFLAGS)" $(LDFLAGS) -o $@ $^ -lcublas -lcudart

# Auto-tuning for optimal kernel parameters
tune: $(LIB_NAME)
	./tune.py

# Clean build artifacts
clean:
	rm -f $(LIB_NAME)

# Install library
install: $(LIB_NAME)
	mkdir -p $(PREFIX)/lib
	cp $(LIB_NAME) $(PREFIX)/lib/

.PHONY: all release debug clean install tune 