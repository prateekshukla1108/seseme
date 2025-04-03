CC = g++
NVCC = nvcc
CUDA_PATH ?= /usr/local/cuda

CFLAGS = -O3 -std=c++14 -Wall -Wextra -ffast-math -pthread
NVCCFLAGS = -O3 -std=c++14 --use_fast_math --default-stream per-thread -Xcompiler -fPIC

INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcurand -lsndfile

# Check if we have AVX2 support
ifeq ($(shell grep -c avx2 /proc/cpuinfo),0)
  # No AVX2, try using SSE4.2
  ifeq ($(shell grep -c sse4_2 /proc/cpuinfo),0)
    # No SSE4.2, use SSE2 which is widely available
    CFLAGS += -msse2
  else
    CFLAGS += -msse4.2
  endif
else
  # Use AVX2 for better performance
  CFLAGS += -mavx2
endif

# Add FMA if available
ifeq ($(shell grep -c fma /proc/cpuinfo),0)
  # No FMA
else
  CFLAGS += -mfma
endif

# Detect CUDA capability
CUDA_COMPUTE_CAPABILITY ?= 61
ifeq ($(shell $(CUDA_PATH)/bin/nvcc --help | grep -c "sm_$(CUDA_COMPUTE_CAPABILITY)"),0)
  $(warning CUDA compute capability $(CUDA_COMPUTE_CAPABILITY) may not be supported, trying common values)
  NVCCFLAGS += -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80
else
  NVCCFLAGS += -gencode arch=compute_$(CUDA_COMPUTE_CAPABILITY),code=sm_$(CUDA_COMPUTE_CAPABILITY)
endif

SOURCES = tts_engine.cpp
CUDA_SOURCES = tts_kernels.cu
OBJECTS = $(SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
EXECUTABLE = tts_engine
CUDA_LIB = libtts_kernels.so

all: $(EXECUTABLE)

$(CUDA_LIB): $(CUDA_SOURCES)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -shared -o $@ $<

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(CUDA_OBJECTS): $(CUDA_SOURCES)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(EXECUTABLE): $(OBJECTS) $(CUDA_LIB)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS) -L. -ltts_kernels

.PHONY: clean run help

clean:
	rm -f $(OBJECTS) $(CUDA_OBJECTS) $(EXECUTABLE) $(CUDA_LIB)

run: $(EXECUTABLE)
	LD_LIBRARY_PATH=.:$(CUDA_PATH)/lib64 ./$(EXECUTABLE)

help:
	@echo "Makefile for optimized TTS engine"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the TTS engine (default)"
	@echo "  clean     - Remove all build artifacts"
	@echo "  run       - Build and run the TTS engine"
	@echo "  help      - Display this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_PATH - Path to CUDA installation (default: /usr/local/cuda)"
	@echo "  CUDA_COMPUTE_CAPABILITY - CUDA compute capability (default: 61)"
	@echo ""
	@echo "Example usage:"
	@echo "  make CUDA_COMPUTE_CAPABILITY=75 - Build for compute capability 7.5"
	@echo "  make run - Build and run with default settings" 