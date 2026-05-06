NVCC ?= nvcc

# CUDA 11.3 on CARC supports sm_60 (P100), sm_70 (V100), sm_80 (A100), sm_86 (A40).
# We also embed PTX (compute_86) for some forward-compatibility on newer GPUs.
GENCODE ?= \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_86,code=compute_86

NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE)
LDLIBS ?= -lcurand -lstdc++fs

BUILD_DIR := build
TARGET := $(BUILD_DIR)/jump_diffusion_cuda
SRC := jump_diffusion_synth.cu

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET) $(LDLIBS)

clean:
	rm -rf $(BUILD_DIR)

