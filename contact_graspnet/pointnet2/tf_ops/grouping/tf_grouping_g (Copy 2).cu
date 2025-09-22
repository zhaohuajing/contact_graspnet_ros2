// tf_grouping_g.cu
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

// ---------------------------------------------------------
// CUDA kernels (no TF headers here)
//   points:   [B, C, N]
//   idx:      [B, npoints] (indices into N)
//   out:      [B, C, npoints]
// ---------------------------------------------------------

__global__ void GroupPointKernel(
    int B, int C, int N, int npoints,
    const float* __restrict__ points,   // [B*C*N]
    const int*   __restrict__ idx,      // [B*npoints]
    float*       __restrict__ out) {    // [B*C*npoints]
  const int b  = blockIdx.x;                               // batch
  const int pi = blockIdx.y * blockDim.x + threadIdx.x;    // point id in [0, npoints)

  if (b >= B || pi >= npoints) return;

  const int gather_i = idx[b * npoints + pi];
  if (gather_i < 0 || gather_i >= N) return;

  // copy along channels
  const int bcN_base   = b * C * N;
  const int bcNP_base  = b * C * npoints;
  for (int c = 0; c < C; ++c) {
    out[bcNP_base + c * npoints + pi] =
        points[bcN_base + c * N + gather_i];
  }
}

__global__ void GroupPointGradKernel(
    int B, int C, int N, int npoints,
    const float* __restrict__ grad_out,   // [B*C*npoints]
    const int*   __restrict__ idx,        // [B*npoints]
    float*       __restrict__ grad_points)// [B*C*N]
{
  const int b  = blockIdx.x;
  const int pi = blockIdx.y * blockDim.x + threadIdx.x;

  if (b >= B || pi >= npoints) return;

  const int gather_i = idx[b * npoints + pi];
  if (gather_i < 0 || gather_i >= N) return;

  const int bcN_base   = b * C * N;
  const int bcNP_base  = b * C * npoints;
  for (int c = 0; c < C; ++c) {
    // atomic add since many pi can map to same gather_i
    atomicAdd(
      &grad_points[bcN_base + c * N + gather_i],
       grad_out   [bcNP_base + c * npoints + pi]
    );
  }
}

// ---------------------------------------------------------
// C-ABI launchers (called from tf_grouping.cpp)
// ---------------------------------------------------------
extern "C" void GroupPointLauncher(
    int B, int C, int N, int npoints,
    const float* points, const int* idx,
    float* out, cudaStream_t stream)
{
  // 1D threads along points
  int tx = (npoints >= 256) ? 256 : (npoints > 0 ? npoints : 1);
  dim3 threads(tx, 1, 1);
  dim3 blocks(B, (npoints + tx - 1) / tx, 1);
  GroupPointKernel<<<blocks, threads, 0, stream>>>(
      B, C, N, npoints, points, idx, out);
}

extern "C" void GroupPointGradLauncher(
    int B, int C, int N, int npoints,
    const float* grad_out, const int* idx,
    float* grad_points, cudaStream_t stream)
{
  int tx = (npoints >= 256) ? 256 : (npoints > 0 ? npoints : 1);
  dim3 threads(tx, 1, 1);
  dim3 blocks(B, (npoints + tx - 1) / tx, 1);
  GroupPointGradKernel<<<blocks, threads, 0, stream>>>(
      B, C, N, npoints, grad_out, idx, grad_points);
}
