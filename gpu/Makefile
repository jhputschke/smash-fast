# Build the SMASH Phase 4 GPU prototype.
#   make            # build
#   make run        # build and run with default sizes
# Requires nvcc (CUDA toolkit). -arch=native targets the local GPU.
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=native -Xcompiler -fopenmp

all: smash_gpu_prototype unified_memory_bench

smash_gpu_prototype: smash_gpu_prototype.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# GB10 unified-/coherent-memory benchmark (explicit vs managed vs ATS vs resident).
unified_memory_bench: unified_memory_bench.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: all run clean
run: smash_gpu_prototype unified_memory_bench
	OMP_NUM_THREADS=$(shell nproc) ./smash_gpu_prototype 64 24
	@echo
	OMP_NUM_THREADS=$(shell nproc) ./unified_memory_bench 64 24

clean:
	rm -f smash_gpu_prototype unified_memory_bench
