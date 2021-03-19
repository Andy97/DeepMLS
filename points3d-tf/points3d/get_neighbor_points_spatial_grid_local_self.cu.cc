#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda_runtime.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

// Macro definition
#define CudaAssert( X ) if ( !(X) ) {printf( "Thread %d:%d failed assert at %s:%d!\n", blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); return;}

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
static __global__ void find_knn_grids_local_self(
    const int nthreads, const int* n_source_points, const T* in_source_points, const int batch_size, const int knn, const int CUBES_TOTAL, const int MAX_POINTS_IN_CUBE,
    const int *cube_points_num_ptr, const int *cube_points_indices_ptr, int *neighbor_points_id_ptr, T *neighbor_points_dist_ptr) {
                              //    0       1   2   3  4  5  6        7          10              14            18         19       21        23        25
    __shared__ int disp_x[27];// = {0,     -1,  0,  0, 0, 0, 1,      -1, -1, -1, -1,  0,  0,  0, 0,  1, 1,  1, 1,        -1,  -1,  -1,  -1,   1,   1,   1,   1};
    __shared__ int disp_y[27];// = {0,      0, -1,  0, 0, 1, 0,      -1,  0,  0,  1, -1, -1,  1, 1,  0, 0, -1, 1,         1,   1,  -1,  -1,  -1,  -1,   1,   1};
    __shared__ int disp_z[27];// = {0,      0,  0, -1, 1, 0, 0,       0, -1,  1,  0, -1,  1, -1, 1, -1, 1,  0, 0,        -1,   1,  -1,   1,  -1,   1,  -1,   1};
    
    //fill the shared memory
    if(threadIdx.x < 27)
    {
      int i = threadIdx.x;
      
      if(i < 9)
        disp_x[i] = 0;
      else if (i < 18)
        disp_x[i] = -1;
      else
        disp_x[i] = 1;
      
      if ((i/3) % 3 == 0)
        disp_y[i] = 0;
      else if ((i/3) % 3 == 1)
        disp_y[i] = -1;
      else
        disp_y[i] = 1;
      
      if(i % 3 == 0)
        disp_z[i] = 0;
      else if(i % 3 == 1)
        disp_z[i] = -1;
      else
        disp_z[i] = 1;      
    }
    __syncthreads();
    
    int batch_index = 0;
    
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    
    for(int ii = batch_index; ii < batch_size; ii++)
      if(index < n_source_points[ii])
      {
        batch_index = ii;
        break;
      }
    
    int* cur_neighbor_points_id_ptr = neighbor_points_id_ptr + index*knn;
    T *cur_neighbor_points_dist_ptr = neighbor_points_dist_ptr + index*knn;
    
    for(int ii = 0; ii < knn; ii++)
      cur_neighbor_points_id_ptr[ii] = -1;
        
    T sx = in_source_points[6 * index];
    T sy = in_source_points[6 * index + 1];
    T sz = in_source_points[6 * index + 2];
    
    //find current cube and nearby cubes
    int scx = int((sx + 1) * 5);
    int scy = int((sy + 1) * 5);
    int scz = int((sz + 1) * 5);
    
    if(scx < 0)
      scx = 0;
    else if(scx > 9)
      scx = 9;
    
    if(scy < 0)
      scy = 0;
    else if (scy > 9)
      scy = 9;
    
    if(scz < 0)
      scz = 0;
    else if(scz > 9)
      scz = 9;
    
    CudaAssert(scx>=0 && scy>=0 && scz>=0 && scx<10 && scy<10 && scz<10);
      
    int k = 0;
    //T tnx,tny,tnz,tn_norm;
    T dx,dy,dz,cur_distance;
    //T dot_product;
        
    for (int j = 0; j < 27; j++)
    {
      int tcx = scx + disp_x[j];
      int tcy = scy + disp_y[j];
      int tcz = scz + disp_z[j];
      if(tcx < 0 || tcx >=10)
        continue;
      if(tcy < 0 || tcy >=10)
        continue;
      if(tcz < 0 || tcz >=10)
        continue;
      int cube_index = 100*tcx + 10*tcy + tcz;
      const int points_in_cube = cube_points_num_ptr[batch_index*CUBES_TOTAL + cube_index];
      const int *cube_indices = cube_points_indices_ptr + batch_index*CUBES_TOTAL*MAX_POINTS_IN_CUBE + cube_index * MAX_POINTS_IN_CUBE;
            
      for (int pt = 0; pt < points_in_cube; pt++)
      {
        int tid = cube_indices[pt];
        if(tid == index)
          continue;
        
        dx = sx - in_source_points[6 * tid];
        dy = sy - in_source_points[6 * tid + 1];
        dz = sz - in_source_points[6 * tid + 2];
        
        cur_distance = dx * dx + dy * dy + dz * dz;
        
        int iii;
        
				if(k < knn)
				{
					//do a insertion less than knn
					iii = k;
				}
				else if(cur_distance < cur_neighbor_points_dist_ptr[knn-1])
				{
					//do the insertion
					iii = knn-1;
				}
				else 
					continue;
				
				//the actual comparison
				for(; iii > 0; iii--)
				{
					if(cur_distance < cur_neighbor_points_dist_ptr[iii-1])
					{
						cur_neighbor_points_dist_ptr[iii] = cur_neighbor_points_dist_ptr[iii-1];
						cur_neighbor_points_id_ptr[iii] = cur_neighbor_points_id_ptr[iii-1];
					}
					else
						break;
				}
				cur_neighbor_points_dist_ptr[iii] = cur_distance;
        cur_neighbor_points_id_ptr[iii] = tid;
        k++;
      }
    }
    
    //CudaAssert(k>=knn);
    if(k == 0)
    {
      cur_neighbor_points_dist_ptr[0] = 1e20;
      cur_neighbor_points_id_ptr[0] = -1;
      
      int start_index = 0;
      if(batch_index != 0)
        start_index = n_source_points[batch_index - 1];
      
      //continue search until find nearest point at least
      for (int pt = start_index; pt < n_source_points[batch_index]; pt++)
      {
        if(pt == index)
          continue;
        
        dx = sx - in_source_points[6 * pt];
        dy = sy - in_source_points[6 * pt + 1];
        dz = sz - in_source_points[6 * pt + 2];
        
        cur_distance = dx * dx + dy * dy + dz * dz;
        
        if(cur_distance < cur_neighbor_points_dist_ptr[0])
        {
          cur_neighbor_points_dist_ptr[0] = cur_distance;
          cur_neighbor_points_id_ptr[0] = pt;
        }
      }
    }
    CudaAssert(cur_neighbor_points_id_ptr[0] >= 0);
  }
}

template <typename T>
static __global__ void gather_cube_pointsb_local(
    const int nthreads, const int* n_target_points, const T* in_target_points, const int MAX_POINTS_IN_CUBE, 
    const int CUBES_TOTAL, int *cube_points_num_ptr, int *cube_points_indices_ptr) {
  CUDA_1D_KERNEL_LOOP(batch_index, nthreads) {
    int *cube_indices = cube_points_indices_ptr + batch_index * MAX_POINTS_IN_CUBE * CUBES_TOTAL;
    int *batch_cube_points_num_ptr = cube_points_num_ptr + batch_index*CUBES_TOTAL;
    
    int start_index = 0;
    if(batch_index != 0)
      start_index = n_target_points[batch_index - 1];
    
    for (int n_target_index = start_index; n_target_index < n_target_points[batch_index]; n_target_index++)
    {
      T tx = in_target_points[6*n_target_index];
      T ty = in_target_points[6*n_target_index + 1];
      T tz = in_target_points[6*n_target_index + 2];
      //find if point is inside cube
      int x = int((tx + 1) * 5);
      int y = int((ty + 1) * 5);
      int z = int((tz + 1) * 5);
      if(x < 0)
        x = 0;
      else if(x > 9)
        x = 9;
      
      if(y < 0)
        y = 0;
      else if (y > 9)
        y = 9;
      
      if(z < 0)
        z = 0;
      else if(z > 9)
        z = 9;
      
      int cube_index = 100*x+10*y+z;
      cube_indices[cube_index*MAX_POINTS_IN_CUBE + batch_cube_points_num_ptr[cube_index]] = n_target_index;
      batch_cube_points_num_ptr[cube_index]++;
      CudaAssert(batch_cube_points_num_ptr[cube_index] <= MAX_POINTS_IN_CUBE);
    }
  }
}

template <typename T>
void neighbor_points_spatial_grid_local_self(OpKernelContext* context,
    const int* n_source_points, const int batch_size, const T* in_source_points,
    const int knn, const int batch_points_total, int* neighbor_points_id_ptr) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;
  
  const int CUBES_TOTAL = 1000;
  const int MAX_POINTS_IN_CUBE = 4096;
  
  //assume input points are bounded by a unit sphere, and we partition it to 10x10x10 grids
  const TensorShape cube_index_shape({batch_size, CUBES_TOTAL, MAX_POINTS_IN_CUBE});
  Tensor cube_points_indices;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, cube_index_shape, &cube_points_indices));
  auto cube_points_indices_ptr = cube_points_indices.flat<int>().data();
  
  const TensorShape cube_points_num_shape({batch_size, CUBES_TOTAL});
  Tensor cube_points_num;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, cube_points_num_shape, &cube_points_num));
  auto cube_points_num_ptr = cube_points_num.flat<int>().data();
  cudaMemset(cube_points_num_ptr, 0, sizeof(int)*batch_size*CUBES_TOTAL);
  
  const TensorShape neighbor_points_distance_shape({batch_points_total, knn});
  Tensor neighbor_points_distance;
  OP_REQUIRES_OK(context, context->allocate_temp(sizeof(T)==4?DT_FLOAT:DT_DOUBLE, neighbor_points_distance_shape, &neighbor_points_distance));
  auto neighbor_points_distance_ptr = neighbor_points_distance.flat<T>().data();
  
	nthreads = batch_size;
  config = GetCudaLaunchConfig(nthreads, d);
  gather_cube_pointsb_local
		  <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			nthreads, n_source_points, in_source_points, MAX_POINTS_IN_CUBE, CUBES_TOTAL, cube_points_num_ptr, cube_points_indices_ptr);
    
  nthreads = batch_points_total;
  config = GetCudaLaunchConfig(nthreads, d);
  //printf("%d block and %d threads per block\n", config.block_count, config.thread_per_block);
  find_knn_grids_local_self
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      nthreads, n_source_points, in_source_points, batch_size, knn, CUBES_TOTAL, MAX_POINTS_IN_CUBE,
      cube_points_num_ptr, cube_points_indices_ptr, neighbor_points_id_ptr, neighbor_points_distance_ptr);
  cudaDeviceSynchronize();
    
  cudaError_t err2 = cudaGetLastError();
  if (err2 != cudaSuccess)
    printf("Cuda error: %s\n", cudaGetErrorString(err2));
  
}

//explicit instantiation
template void neighbor_points_spatial_grid_local_self<float>(OpKernelContext* context,
    const int* n_source_points, const int batch_size,
    const float* in_source_points, const int knn, const int batch_points_total, int* neighbor_points_id_ptr);
	
template void neighbor_points_spatial_grid_local_self<double>(OpKernelContext* context,
    const int* n_source_points, const int batch_size, 
    const double* in_source_points, const int knn, const int batch_points_total, int* neighbor_points_id_ptr);

}  // namespace tensorflow
