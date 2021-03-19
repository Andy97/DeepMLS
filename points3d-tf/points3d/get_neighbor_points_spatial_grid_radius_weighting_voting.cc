#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

template <typename T>
void neighbor_points_spatial_grid_no_normal_weight_local_varying_radius_voting(OpKernelContext* context,
    const int n_source_points, const int* n_target_points, const int batch_size,
    const T* in_source_points, const T* in_target_points, const T* squared_radius, const int knn,
    int* neighbor_points_id_ptr);

REGISTER_OP("GetNeighborSpatialGridRadiusVVoting")
.Attr("T: {float, double}")
.Input("source_points: T")
.Input("target_points: T")
.Input("target_point_num: int32")
.Input("squared_radius: T")
.Attr("knn: int = 10")
.Output("neighbor_points: int32")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->MakeShape({c->Dim(c->input(0),0),c->UnknownDim(),c->UnknownDim()}));
  return Status::OK();
})
.Doc(R"doc(
Find the neighbor points for each source points in target pointset using spatial partition and efficient insertion sort.
)doc");

template <typename T>
class GetNeighborSpatialGridRadiusVVotingOp : public OpKernel {
 public:
  explicit GetNeighborSpatialGridRadiusVVotingOp(OpKernelConstruction* context)
      : OpKernel(context) 
	  {
		OP_REQUIRES_OK(context, context->GetAttr("knn",
                                             &this->knn_));
	  }

  void Compute(OpKernelContext* context) override {
    // in source points [bs, 3*n_source_points]
    const Tensor& in_source_points = context->input(0);
    auto in_source_points_ptr = in_source_points.flat<T>().data();
    batch_size_ = in_source_points.dim_size(0);
    n_source_points_ = in_source_points.dim_size(1)/3;

    // in target points [\sum(n_target_points), 6]
    const Tensor& in_target_points = context->input(1);
    auto in_target_points_ptr = in_target_points.flat<T>().data();
    CHECK_EQ(in_target_points.dim_size(1), 6);
    
    const Tensor& in_target_num = context->input(2);
    auto in_target_num_ptr = in_target_num.flat<int>().data();
    CHECK_EQ(in_target_num.dim_size(0), batch_size_);
    
    //in shape [\sum(n_target_points)]
    const Tensor& in_squared_radius = context->input(3);
    auto in_squared_radius_ptr = in_squared_radius.flat<T>().data();
    CHECK_EQ(in_squared_radius.dim_size(0), in_target_points.dim_size(0));
        
    // out tensor
    Tensor* out_vec = nullptr;
    TensorShape out_shape({batch_size_, n_source_points_, knn_});
    OP_REQUIRES_OK(context, context->allocate_output("neighbor_points", out_shape, &out_vec));
    auto out_ptr = out_vec->flat<int>().data();

    // find neighbor points
    neighbor_points_spatial_grid_no_normal_weight_local_varying_radius_voting<T>(context, n_source_points_, in_target_num_ptr,
        batch_size_, in_source_points_ptr, in_target_points_ptr, in_squared_radius_ptr, knn_, out_ptr);
  }

 private:
  int n_source_points_;
  int batch_size_;
  int knn_;
};

REGISTER_KERNEL_BUILDER(Name("GetNeighborSpatialGridRadiusVVoting").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    GetNeighborSpatialGridRadiusVVotingOp<float>);
	
REGISTER_KERNEL_BUILDER(Name("GetNeighborSpatialGridRadiusVVoting").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    GetNeighborSpatialGridRadiusVVotingOp<double>);
}  // namespace tensorflow
