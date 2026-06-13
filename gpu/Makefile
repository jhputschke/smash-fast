# Build the SMASH Phase 4 GPU prototype.
#   make            # build (CUDA)
#   make run        # build and run with default sizes (CUDA)
#   make run-mlx    # run the MLX (Apple Silicon) port -- no build step needed
# CUDA targets require nvcc (CUDA toolkit); -arch=native targets the local GPU.
# MLX targets require the fno_env_mlx conda env (Python 3.13 + mlx + numpy).
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=native -Xcompiler -fopenmp
PYTHON ?= /opt/homebrew/Caskroom/miniconda/base/envs/fno_env_mlx/bin/python

all: smash_gpu_prototype unified_memory_bench

smash_gpu_prototype: smash_gpu_prototype.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# GB10 unified-/coherent-memory benchmark (explicit vs managed vs ATS vs resident).
unified_memory_bench: unified_memory_bench.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: all run run-mlx clean
run: smash_gpu_prototype unified_memory_bench
	OMP_NUM_THREADS=$(shell nproc) ./smash_gpu_prototype 64 24
	@echo
	OMP_NUM_THREADS=$(shell nproc) ./unified_memory_bench 64 24

# MLX (Apple Silicon) port -- mirrors the CUDA prototypes on the Metal GPU.
# smash_mlx_prototype.py        : vectorized MLX array ops (idiomatic)
# smash_mlx_metal_prototype.py  : explicit Metal kernels (kernel-for-kernel CUDA port)
# unified_memory_bench_mlx.py   : unified-memory benchmark
run-mlx:
	$(PYTHON) smash_mlx_prototype.py 64 24
	@echo
	$(PYTHON) smash_mlx_metal_prototype.py 64 24
	@echo
	$(PYTHON) unified_memory_bench_mlx.py 64 24

clean:
	rm -f smash_gpu_prototype unified_memory_bench
