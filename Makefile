# Data-parallel SGD on CUDA with hand-written allreduce
# make            -> build everything (needs nvcc for the CUDA targets)
# make seq        -> sequential C baseline only (gcc)
# ARCH=sm_75 is the T4 (Kaggle/Colab). Use sm_80 for A100, sm_86 for RTX 30xx.

CC     ?= gcc
NVCC   ?= nvcc
ARCH   ?= sm_75
CFLAGS  = -O2 -Wall -Wextra -std=gnu11
NVFLAGS = -O2 -arch=$(ARCH) -std=c++14 -Xcompiler -Wall
BIN     = bin

COMMON_C  = src/common/mnist.c src/common/params.c
COMMON_O  = $(BIN)/mnist.o $(BIN)/params.o
AR_SRC    = src/allreduce/common.cu src/allreduce/registry.cu src/allreduce/host_ref.cu \
            src/allreduce/tree.cu src/allreduce/ring.cu src/allreduce/rd.cu

.PHONY: all seq cuda clean data simcheck
all: seq cuda
seq: $(BIN)/mlp_seq
cuda: $(BIN)/train_dp $(BIN)/bench_allreduce

$(BIN):
	mkdir -p $(BIN)

$(BIN)/mlp_seq: src/seq/mlp_seq.c $(COMMON_C) src/common/*.h | $(BIN)
	$(CC) $(CFLAGS) -o $@ src/seq/mlp_seq.c $(COMMON_C) -lm

# src/common/*.c are compiled once with gcc (C) and linked into the CUDA programs.
$(BIN)/%.o: src/common/%.c src/common/*.h | $(BIN)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BIN)/train_dp: src/cuda/train_dp.cu src/cuda/kernels.cu src/cuda/kernels.cuh $(AR_SRC) src/allreduce/*.h $(COMMON_O) src/common/*.h | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ src/cuda/train_dp.cu src/cuda/kernels.cu $(AR_SRC) $(COMMON_O)

$(BIN)/bench_allreduce: bench/bench_allreduce.cu $(AR_SRC) src/allreduce/*.h src/common/*.h | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ bench/bench_allreduce.cu $(AR_SRC)

# Race/correctness check of tree, ring, rd on the CPU with a mocked CUDA runtime (g++ only)
simcheck: | $(BIN)
	g++ -O1 -std=c++17 -w -Itests/mock -Isrc/allreduce -x c++ tests/sim_allreduce.cpp \
	    src/allreduce/tree.cu src/allreduce/ring.cu src/allreduce/rd.cu -o $(BIN)/sim_allreduce
	./$(BIN)/sim_allreduce

data:
	bash scripts/get_mnist.sh data

clean:
	rm -rf $(BIN)
