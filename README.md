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

Clone the repository with the pinned baseline implementations and build the
project:

```bash
git clone --recursive <repository-url>
cd GRAB-ANNS
mkdir build
cd build
cmake ..
make -j
```

## Run

Examples and component tests are under `tests/`. Paper experiment drivers for
GRAB-ANNS, ACORN, SeRF, HNSW-Filter, and Milvus are under [`eval/`](eval/).
Datasets use the `fvecs` format; configure their locations with
`eval/datasets.env.example`.
