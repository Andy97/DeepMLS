#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

template <typename T>
void neighbor_points_spatial_grid_local_self(OpKernelContext* context,
    const int* n_source_points, const int batch_size, const T* in_source_points, 
    const int knn, const int batch_points_total, int* neighbor_points_id_ptr);

REGISTER_OP("GetNeighborSpatialGridLocalSelf")
.Attr("T: {float, double}")
.Input("source_points: T")
.Input("source_points_num: int32")
.Attr("knn: int = 10")
.Output("neighbor_points: int32")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->MakeShape({c->Dim(c->input(0),0), c->UnknownDim()}));
  return Status::OK();
})
.Doc(R"doc(
Find the neighbor points for each points using spatial partition and efficient insertion sort.
)doc");

template <typename T>
class GetNeighborSpatialGridLocalSelfOp : public OpKernel {
 public:
  explicit GetNeighborSpatialGridLocalSelfOp(OpKernelConstruction* context)
      : OpKernel(context) 
	  {
		OP_REQUIRES_OK(context, context->GetAttr("knn",
                                             &this->knn_));
	  }

  void Compute(OpKernelContext* context) override {
    // in source points [\sum(n_source_points), 6]
    const Tensor& in_source_points = context->input(0);
    auto in_source_points_ptr = in_source_points.flat<T>().data();
    batch_points_total = in_source_points.dim_size(0);
    
    const Tensor& in_source_num = context->input(1);
    auto in_source_num_ptr = in_source_num.flat<int>().data();
    batch_size_ = in_source_num.dim_size(0);
    
    // out loss
    Tensor* out_vec = nullptr;
    TensorShape out_shape({batch_points_total, knn_});
    OP_REQUIRES_OK(context, context->allocate_output("neighbor_points", out_shape, &out_vec));
    auto out_ptr = out_vec->flat<int>().data();

    // find neighbor point
    neighbor_points_spatial_grid_local_self<T>(context, in_source_num_ptr, batch_size_, in_source_points_ptr, knn_, batch_points_total, out_ptr);
  }

 private:
  int batch_points_total;
  int batch_size_;
  int knn_;
};

REGISTER_KERNEL_BUILDER(Name("GetNeighborSpatialGridLocalSelf").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    GetNeighborSpatialGridLocalSelfOp<float>);
	
REGISTER_KERNEL_BUILDER(Name("GetNeighborSpatialGridLocalSelf").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    GetNeighborSpatialGridLocalSelfOp<double>);
}  // namespace tensorflow
