# Build the SMASH Phase 4 GPU prototype.
#   make            # build
#   make run        # build and run with default sizes
# Requires nvcc (CUDA toolkit). -arch=native targets the local GPU.
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=native -Xcompiler -fopenmp

smash_gpu_prototype: smash_gpu_prototype.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: run clean
run: smash_gpu_prototype
	OMP_NUM_THREADS=$(shell nproc) ./smash_gpu_prototype 64 24

clean:
	rm -f smash_gpu_prototype
