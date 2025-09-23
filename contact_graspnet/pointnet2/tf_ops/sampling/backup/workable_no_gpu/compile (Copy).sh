#!/bin/bash

TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDA_HOME=/usr/local/cuda  # Change if your CUDA is somewhere else

nvcc tf_sampling_g.cu -c -o tf_sampling_g.cu.o \
    -I$TF_INC -I$TF_INC/external/nsync/public \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -x cu -Xcompiler -fPIC -std=c++14

g++ -std=c++14 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so \
    -shared -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
    -L$TF_LIB -ltensorflow_framework \
    -L$CUDA_HOME/lib64 -lcudart
