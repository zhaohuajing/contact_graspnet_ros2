#!/bin/bash

# Get TF compile and link flags
TF_CFLAGS=$(python3.10 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python3.10 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# Optional: explicitly set your CUDA path if it's nonstandard
CUDA_HOME=/usr/local/cuda-12

# Compile .cu file with -std=c++17
/usr/local/cuda/bin/nvcc tf_sampling_g.cu -c -o tf_sampling_g.cu.o \
    -std=c++17 \
    -x cu -Xcompiler -fPIC \
    -I/usr/include \
    -I${CUDA_HOME}/include \
    ${TF_CFLAGS}

# Link with g++ using the same standard
g++ -std=c++17 -shared -o tf_sampling_so.so tf_sampling.cpp tf_sampling_g.cu.o \
    -fPIC ${TF_CFLAGS} ${TF_LFLAGS} \
    -L${CUDA_HOME}/lib64 -lcudart
