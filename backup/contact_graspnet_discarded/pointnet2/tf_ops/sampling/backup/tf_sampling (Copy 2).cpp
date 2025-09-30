// tf_sampling.cpp (TF 2.20+ compatible)
// Registers PointNet++ sampling/interpolation ops with absl::Status shape fns.
// CPU kernels below are stubs that report UNIMPLEMENTED (GPU expected).

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "absl/status/status.h"

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::DEVICE_GPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
// DO NOT 'using' REGISTER_KERNEL_BUILDER (it's a macro)
namespace shape_inference = ::tensorflow::shape_inference;

// -------------------------------
// FarthestPointSample
// -------------------------------
REGISTER_OP("FarthestPointSample")
    .Input("points: float32")        // [B, N, 3] or [N, 3]
    .Input("npoint: int32")          // scalar
    .Output("idx: int32")            // [B, npoint] or [npoint]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle pts;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &pts));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, nullptr));  // npoint scalar

      auto npoint_dim = c->UnknownDim();
      if (c->RankKnown(pts) && c->Rank(pts) == 3) {
        auto B = c->Dim(pts, 0);
        c->set_output(0, c->MakeShape({B, npoint_dim}));
      } else {
        c->set_output(0, c->MakeShape({npoint_dim}));
      }
      return absl::OkStatus();
    })
    .Doc(R"doc(
Selects farthest-point samples.

points: float32, [B, N, 3] or [N, 3]
npoint: int32 scalar
idx: int32, [B, npoint] or [npoint]
)doc");

// -------------------------------
// GatherPoint / GatherPointGrad
// -------------------------------
REGISTER_OP("GatherPoint")
    .Input("points: float32")   // [B, C, N] or [C, N]
    .Input("idx: int32")        // [B, M] or [M]
    .Output("out: float32")     // [B, C, M] or [C, M]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle pts, idx;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &pts));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &idx));

      if (c->RankKnown(pts) && c->Rank(pts) == 3 &&
          c->RankKnown(idx) && c->Rank(idx) == 2) {
        auto B = c->Dim(pts, 0);
        auto C = c->Dim(pts, 1);
        auto M = c->Dim(idx, 1);
        c->set_output(0, c->MakeShape({B, C, M}));
      } else if (c->RankKnown(pts) && c->Rank(pts) == 2 &&
                 c->RankKnown(idx) && c->Rank(idx) == 1) {
        auto C = c->Dim(pts, 0);
        auto M = c->Dim(idx, 0);
        c->set_output(0, c->MakeShape({C, M}));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return absl::OkStatus();
    });

REGISTER_OP("GatherPointGrad")
    .Input("points: float32")   // [B, C, N] or [C, N] (for shape)
    .Input("idx: int32")        // [B, M] or [M]
    .Input("grad_out: float32") // [B, C, M] or [C, M]
    .Output("grad_points: float32") // [B, C, N] or [C, N]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0)); // same shape as points
      return absl::OkStatus();
    });

// -------------------------------
// ThreeNN
// -------------------------------
REGISTER_OP("ThreeNN")
    .Input("unknown: float32")   // [B, n, 3] or [n, 3]
    .Input("known: float32")     // [B, m, 3] or [m, 3]
    .Output("dist2: float32")    // [B, n, 3] or [n, 3]
    .Output("idx: int32")        // [B, n, 3] or [n, 3]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unk;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &unk));
      if (c->RankKnown(unk)) {
        if (c->Rank(unk) == 3) {
          auto B = c->Dim(unk, 0);
          auto n = c->Dim(unk, 1);
          c->set_output(0, c->MakeShape({B, n, 3}));
          c->set_output(1, c->MakeShape({B, n, 3}));
        } else {
          auto n = c->Dim(unk, 0);
          c->set_output(0, c->MakeShape({n, 3}));
          c->set_output(1, c->MakeShape({n, 3}));
        }
      } else {
        c->set_output(0, c->UnknownShape());
        c->set_output(1, c->UnknownShape());
      }
      return absl::OkStatus();
    });

// -------------------------------
// ThreeInterpolate / ThreeInterpolateGrad
// -------------------------------
REGISTER_OP("ThreeInterpolate")
    .Input("points: float32")    // [B, C, m] or [C, m]
    .Input("idx: int32")         // [B, n, 3] or [n, 3]
    .Input("weight: float32")    // [B, n, 3] or [n, 3]
    .Output("out: float32")      // [B, C, n] or [C, n]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle pts, idx, w;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &pts));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &idx));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 2, &w));

      if (c->RankKnown(pts) && c->Rank(pts) == 3 &&
          c->RankKnown(idx) && c->Rank(idx) == 3) {
        auto B = c->Dim(pts, 0);
        auto C = c->Dim(pts, 1);
        auto n = c->Dim(idx, 1);
        c->set_output(0, c->MakeShape({B, C, n}));
      } else if (c->RankKnown(pts) && c->Rank(pts) == 2 &&
                 c->RankKnown(idx) && c->Rank(idx) == 2) {
        auto C = c->Dim(pts, 0);
        auto n = c->Dim(idx, 0);
        c->set_output(0, c->MakeShape({C, n}));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return absl::OkStatus();
    });

REGISTER_OP("ThreeInterpolateGrad")
    .Input("grad_out: float32")  // [B, C, n] or [C, n]
    .Input("idx: int32")         // [B, n, 3] or [n, 3]
    .Input("weight: float32")    // [B, n, 3] or [n, 3]
    .Input("m: int32")           // scalar (size m)
    .Output("grad_points: float32") // [B, C, m] or [C, m]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle gout;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &gout));
      if (c->RankKnown(gout) && c->Rank(gout) == 3) {
        auto B = c->Dim(gout, 0);
        auto C = c->Dim(gout, 1);
        c->set_output(0, c->MakeShape({B, C, c->UnknownDim()}));
      } else if (c->RankKnown(gout) && c->Rank(gout) == 2) {
        auto C = c->Dim(gout, 0);
        c->set_output(0, c->MakeShape({C, c->UnknownDim()}));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return absl::OkStatus();
    });

// =====================================================================
// CPU STUB KERNELS
// =====================================================================

template <typename T>
class UnimplementedCpuKernel : public OpKernel {
 public:
  explicit UnimplementedCpuKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    ctx->CtxFailure(
        ::tensorflow::errors::Unimplemented(
            "This op is only implemented for GPU in Contact-GraspNet "
            "(PointNet++ custom ops). Ensure the CUDA kernels are built and "
            "loaded (e.g., *_g.cu compiled into the same .so)."));
  }
};

#define REGISTER_CPU_STUB(OPNAME) \
  REGISTER_KERNEL_BUILDER(        \
      Name(OPNAME).Device(DEVICE_CPU), UnimplementedCpuKernel<float>)

// Register CPU stubs for all ops
REGISTER_CPU_STUB("FarthestPointSample");
REGISTER_CPU_STUB("GatherPoint");
REGISTER_CPU_STUB("GatherPointGrad");
REGISTER_CPU_STUB("ThreeNN");
REGISTER_CPU_STUB("ThreeInterpolate");
REGISTER_CPU_STUB("ThreeInterpolateGrad");

#undef REGISTER_CPU_STUB
