# GRAB-ANNS

This repository contains the implementation of **GRAB-ANNS: GPU Acceleration for Graph-Based Hybrid Vector Index and Search**.

GRAB-ANNS is a GPU-native graph-based framework for hybrid vector search. The codebase includes index construction, search, and evaluation components used for our experiments.

## Requirements

The project is designed for Linux environments with NVIDIA GPUs.

### Dependencies

- CMake
- C++17 compiler
- CUDA Toolkit
- FAISS with GPU support
- Conda (recommended)

We recommend installing dependencies with Conda:

```bash
conda create -n grab-anns
conda activate grab-anns
conda install -c conda-forge faiss-gpu
```

Please make sure that your CUDA toolkit, NVIDIA driver, and FAISS version are compatible.

## Build

Clone the repository and build the project:

```bash
mkdir build
cd build
cmake ..
make -j
```

## Run

All examples and tests are in `./tests/` path. All tests uses data in format `xxx.fvecs`. 