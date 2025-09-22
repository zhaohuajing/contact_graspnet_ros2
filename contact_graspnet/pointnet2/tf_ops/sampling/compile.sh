# #!/bin/bash

# # TensorFlow paths
# TF_BASE="/home/csrobot/.local/lib/python3.12/site-packages/tensorflow"
# TF_INC="$TF_BASE/include"
# TF_LIB="$TF_BASE"

# # CUDA paths (adjust if your CUDA is not at /usr/local/cuda)
# CUDA_HOME="/usr/local/cuda"
# # CUDA_HOME="/usr/local/cuda-12.9"
# CUDA_INC="$CUDA_HOME/include"
# CUDA_LIB="$CUDA_HOME/lib64"

# echo "TF_INC: $TF_INC"
# echo "TF_LIB: $TF_LIB"
# echo "CUDA_INC: $CUDA_INC"
# echo "CUDA_LIB: $CUDA_LIB"

# rm -f *.o *.so

# nvcc -std=c++14 -c -o tf_sampling_g.cu.o tf_sampling_g.cu \
#     -I $TF_INC -I $TF_INC/external/nsync/public -I $CUDA_INC \
#     -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# g++ -std=c++14 -shared -o tf_sampling_so.so tf_sampling.cpp tf_sampling_g.cu.o \
#     -I $TF_INC -I $TF_INC/external/nsync/public -I $CUDA_INC \
#     -L $TF_LIB -L $CUDA_LIB -ltensorflow_framework.2 -lcudart -fPIC -O2


#!/bin/bash
# Compile TensorFlow custom op for Contact-GraspNet (sampling)

# Explicit paths
TF_BASE=/home/csrobot/.local/lib/python3.12/site-packages/tensorflow
TF_INC=$TF_BASE/include
TF_LIB=$TF_BASE
CUDA_HOME=/usr/local/cuda
CUDA_INC=$CUDA_HOME/include
CUDA_LIB=$CUDA_HOME/lib64

echo "TF_INC: $TF_INC"
echo "TF_LIB: $TF_LIB"
echo "CUDA_INC: $CUDA_INC"
echo "CUDA_LIB: $CUDA_LIB"

# Clean old .so
rm -f tf_sampling_so.so

# Compile the .cpp (not .cu) with nvcc
nvcc tf_sampling.cpp -o tf_sampling_so.so \
    -std=c++17 -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
    -I $TF_INC -I $TF_INC/external/nsync/public \
    -I $CUDA_INC -L $CUDA_LIB -L $TF_LIB \
    -ltensorflow_framework


