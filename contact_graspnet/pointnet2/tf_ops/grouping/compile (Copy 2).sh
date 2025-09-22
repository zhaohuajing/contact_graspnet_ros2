#!/usr/bin/env bash
set -euo pipefail

# ---- Paths (adjust if your setup differs) ----
TF_BASE="/home/csrobot/.local/lib/python3.12/site-packages/tensorflow"
CUDA_HOME="/usr/local/cuda"

# ---- TensorFlow includes / link flags (hard-coded; no Python in-script) ----
TF_CFLAGS="-I${TF_BASE}/include -I${TF_BASE}/include/external/nsync/public"
TF_LFLAGS="-L${TF_BASE} -ltensorflow_framework"

# ---- Compilers ----
CXX="g++"
NVCC="${CUDA_HOME}/bin/nvcc"

# ---- Common defines (match working sampling style) ----
# Note: Manylinux TF builds still use old C++ ABI => 0
COMMON_DEFS="-D_GLIBCXX_USE_CXX11_ABI=0 -DEIGEN_USE_GPU -DGOOGLE_CUDA=1 -DNDEBUG"

# Try to sidestep fp8 template issues seen in ml_dtypes headers
# (Include both spellings to be safe across versions)
FP8_DEFS="-DML_DTYPES_DISABLE_FLOAT8 -DMLDTYPES_DISABLE_FLOAT8"

# ---- Host & NVCC flags ----
CXXSTD="-std=c++17"
CXXFLAGS="${CXXSTD} ${COMMON_DEFS} ${FP8_DEFS} -fPIC -O2 -fno-exceptions -fno-rtti"
NVCCFLAGS="${CXXSTD} ${COMMON_DEFS} ${FP8_DEFS} \
  -Xcompiler -fPIC \
  --expt-relaxed-constexpr \
  --extended-lambda \
  -O2 \
  --threads=1"

# ---- Includes ----
INC_TF="${TF_CFLAGS}"
INC_CUDA="-I${CUDA_HOME}/include"

# ---- Clean previous partials (optional) ----
rm -f tf_grouping.cu.o tf_grouping.o tf_grouping_so.so

echo "[1/3] Compiling CUDA: tf_grouping_g.cu -> tf_grouping.cu.o"
"${NVCC}" ${NVCCFLAGS} -c tf_grouping_g.cu -o tf_grouping.cu.o ${INC_TF} ${INC_CUDA}

echo "[2/3] Compiling C++:  tf_grouping.cpp -> tf_grouping.o"
"${CXX}" ${CXXFLAGS} -c tf_grouping.cpp -o tf_grouping.o ${INC_TF} ${INC_CUDA}

echo "[3/3] Linking -> tf_grouping_so.so"
# "${CXX}" -shared tf_grouping.o tf_grouping.cu.o -o tf_grouping_so.so \
#   ${TF_LFLAGS} -L"${CUDA_HOME}/lib64" -lcudart
"${CXX}" -shared tf_grouping.o tf_grouping.cu.o -o tf_grouping_so.so \
  ${TF_LFLAGS} -L"${CUDA_HOME}/lib64" -lcudart -ltensorflow_cc


echo "Build complete: tf_grouping_so.so"

# Quick smoke test (optional; comment out to avoid any runtime side-effects)
# python3 - <<'PY'
# import tensorflow as tf
# tf.load_op_library('./tf_grouping_so.so')
# print('tf_grouping_so.so loaded OK')
# PY
