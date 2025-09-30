#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"

using namespace tensorflow;

// ------------------------------------------------------------
// Op Declarations
// ------------------------------------------------------------
REGISTER_OP("GroupPoint")
    .Input("points: float32")      // [B, N, C]
    .Input("idx: int32")           // [B, M, S]
    .Output("out: float32")        // [B, M, S, C]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle points_shape;
        shape_inference::ShapeHandle idx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &points_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &idx_shape));

        shape_inference::DimensionHandle batch_dim = c->Dim(points_shape, 0);
        shape_inference::DimensionHandle m_dim = c->Dim(idx_shape, 1);
        shape_inference::DimensionHandle s_dim = c->Dim(idx_shape, 2);
        shape_inference::DimensionHandle c_dim = c->Dim(points_shape, 2);

        c->set_output(0, c->MakeShape({batch_dim, m_dim, s_dim, c_dim}));
        return Status();
    });

REGISTER_OP("GroupPointGrad")
    .Input("points: float32")     // [B, N, C]
    .Input("idx: int32")          // [B, M, S]
    .Input("grad_out: float32")   // [B, M, S, C]
    .Output("grad_points: float32") // [B, N, C]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    });

// ------------------------------------------------------------
// Forward Declarations for CUDA kernels
// ------------------------------------------------------------
void GroupPointLauncher(int b, int n, int c, int m, int s,
                        const float* points, const int* idx, float* out,
                        cudaStream_t stream);

void GroupPointGradLauncher(int b, int n, int c, int m, int s,
                            const float* grad_out, const int* idx,
                            float* grad_points, cudaStream_t stream);

// ------------------------------------------------------------
// CPU stubs (throw errors)
// ------------------------------------------------------------
template <typename T>
class GroupPointOpCPU : public OpKernel {
public:
    explicit GroupPointOpCPU(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        ctx->CtxFailure(__FILE__, __LINE__, errors::InvalidArgument("GroupPoint CPU not implemented."));
    }
};

template <typename T>
class GroupPointGradOpCPU : public OpKernel {
public:
    explicit GroupPointGradOpCPU(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        ctx->CtxFailure(__FILE__, __LINE__, errors::InvalidArgument("GroupPointGrad CPU not implemented."));
    }
};

// ------------------------------------------------------------
// GPU kernels
// ------------------------------------------------------------
template <typename T>
class GroupPointOpGPU : public OpKernel {
public:
    explicit GroupPointOpGPU(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor& points = ctx->input(0);  // [B, N, C]
        const Tensor& idx = ctx->input(1);     // [B, M, S]

        OP_REQUIRES(ctx, points.dims() == 3, errors::InvalidArgument("points must be [B,N,C]"));
        OP_REQUIRES(ctx, idx.dims() == 3, errors::InvalidArgument("idx must be [B,M,S]"));

        int b = points.dim_size(0);
        int n = points.dim_size(1);
        int c = points.dim_size(2);
        int m = idx.dim_size(1);
        int s = idx.dim_size(2);

        Tensor* out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({b, m, s, c}), &out));

        auto stream = ctx->eigen_device<Eigen::GpuDevice>().stream();
        GroupPointLauncher(b, n, c, m, s,
                           points.flat<float>().data(),
                           idx.flat<int>().data(),
                           out->flat<float>().data(),
                           stream);
    }
};

template <typename T>
class GroupPointGradOpGPU : public OpKernel {
public:
    explicit GroupPointGradOpGPU(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor& points = ctx->input(0);   // [B,N,C]
        const Tensor& idx = ctx->input(1);      // [B,M,S]
        const Tensor& grad_out = ctx->input(2); // [B,M,S,C]

        int b = points.dim_size(0);
        int n = points.dim_size(1);
        int c = points.dim_size(2);
        int m = idx.dim_size(1);
        int s = idx.dim_size(2);

        Tensor* grad_points = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, points.shape(), &grad_points));

        auto stream = ctx->eigen_device<Eigen::GpuDevice>().stream();
        GroupPointGradLauncher(b, n, c, m, s,
                               grad_out.flat<float>().data(),
                               idx.flat<int>().data(),
                               grad_points->flat<float>().data(),
                               stream);
    }
};

// ------------------------------------------------------------
// Register kernels
// ------------------------------------------------------------
REGISTER_KERNEL_BUILDER(Name("GroupPoint").Device(DEVICE_CPU), GroupPointOpCPU<float>);
REGISTER_KERNEL_BUILDER(Name("GroupPointGrad").Device(DEVICE_CPU), GroupPointGradOpCPU<float>);
REGISTER_KERNEL_BUILDER(Name("GroupPoint").Device(DEVICE_GPU), GroupPointOpGPU<float>);
REGISTER_KERNEL_BUILDER(Name("GroupPointGrad").Device(DEVICE_GPU), GroupPointGradOpGPU<float>);
