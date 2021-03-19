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

//assume mls weighting function is exp(-3d^2/r^2)

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
static __global__ void find_knn_grids_no_normalw_local_varying_radius_voting(
    const int nthreads, const int n_source_points, const int* n_target_points, const T* in_source_points, const T* in_target_points, 
    const T* in_squared_radius, const int knn, const int CUBES_TOTAL, const int MAX_POINTS_IN_CUBE,
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
      
      //printf("%d %d %d\n", threadIdx.x, blockIdx.x, gridDim.x);
    }
    __syncthreads();
    
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index =    index / n_source_points;
    int n_source_index = index % n_source_points;
        
    int* cur_neighbor_points_id_ptr = neighbor_points_id_ptr + index*knn;
    T *cur_neighbor_points_dist_ptr = neighbor_points_dist_ptr + index*knn;
    
    for(int ii = 0; ii < knn; ii++)
      cur_neighbor_points_id_ptr[ii] = -1;
        
    T sx = in_source_points[(batch_index*3 + 0)*n_source_points + n_source_index];
    T sy = in_source_points[(batch_index*3 + 1)*n_source_points + n_source_index];
    T sz = in_source_points[(batch_index*3 + 2)*n_source_points + n_source_index];
    
    //filter out padded data
    if(sx < -10)
    {
      cur_neighbor_points_id_ptr[0] = 0;
      continue;
    }
    
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
    
    T dx, dy, dz, cur_distance;
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
                	
        dx = sx - in_target_points[6*tid];
        dy = sy - in_target_points[6*tid + 1];
        dz = sz - in_target_points[6*tid + 2];
        
        T squared_radius = in_squared_radius[tid];
        cur_distance = dx * dx + dy * dy + dz * dz;        
        //compute mls weights
        cur_distance = cur_distance / squared_radius; //sort in ascent order
        
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
    
    if(k <= 1)
    {
      //first try another ring
      for(int dx1 = -2; dx1 <= 2; dx1++)
        for(int dx2 = -2; dx2 <= 2; dx2++)
          for(int dx3 = -2; dx3 <= 2; dx3++)
          {
            if(abs(dx1) != 2 && abs(dx2) != 2 && abs(dx3) != 2)
              continue;
            int tcx = scx + dx1;
            int tcy = scy + dx2;
            int tcz = scz + dx3;
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
        
              dx = sx - in_target_points[6*tid];
              dy = sy - in_target_points[6*tid + 1];
              dz = sz - in_target_points[6*tid + 2];
              
              T squared_radius = in_squared_radius[tid];//[batch_index*n_target_points + tid];
              cur_distance = dx * dx + dy * dy + dz * dz;        
              //compute mls weights
              cur_distance = cur_distance / squared_radius; //sort in ascent order
              
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
    }
    
    //CudaAssert(k>=knn);
    if(k <= 1)
    {
      //rare case
      k = 0;
      cur_neighbor_points_dist_ptr[0] = 1e20;
      cur_neighbor_points_id_ptr[0] = -1;
      
      int start_index = 0;
      if(batch_index != 0)
        start_index = n_target_points[batch_index - 1];
      
      //continue search until find nearest point at least
      for (int pt = start_index; pt < n_target_points[batch_index]; pt++)
      {	
        dx = sx - in_target_points[6 * pt];
        dy = sy - in_target_points[6 * pt + 1];
        dz = sz - in_target_points[6 * pt + 2];
        
        cur_distance = dx * dx + dy * dy + dz * dz;
        T squared_radius = in_squared_radius[pt];//[batch_index*n_target_points + pt];        
        //compute mls weights
        cur_distance = cur_distance / squared_radius; //sort in ascent order
        
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
        cur_neighbor_points_id_ptr[iii] = pt;
        k++;
      }
    }
    
    CudaAssert(cur_neighbor_points_id_ptr[0] >= 0);
    //voting according to normal
    if(k > 1)
    {
      if(k > knn)
        k = knn;
      T tnx,tny,tnz, tn_norm;
      T sn_norm;
      //filter according to normal
      tnx = 0;
      tny = 0;
      tnz = 0;
      //get average normal
      for(int iii = 0; iii < k; iii++)
      {
        int tid = cur_neighbor_points_id_ptr[iii];
        T mls_weight = exp(cur_neighbor_points_dist_ptr[0] - cur_neighbor_points_dist_ptr[iii]);
        tnx += mls_weight * in_target_points[6 * tid + 3];
        tny += mls_weight * in_target_points[6 * tid + 4];
        tnz += mls_weight * in_target_points[6 * tid + 5];
      }
      //normalize normal
      tn_norm = sqrt(tnx*tnx+tny*tny+tnz*tnz);
      CudaAssert(tn_norm > 1e-20);
      tnx /= tn_norm;
      tny /= tn_norm;
      tnz /= tn_norm;
      
      //remove neighbors with opposite normal direction
      for(int iii = k - 1; iii >= 0; iii--)
      {
        int tid = cur_neighbor_points_id_ptr[iii];
        T dot_product = tnx*in_target_points[6 * tid + 3];
        dot_product  += tny*in_target_points[6 * tid + 4];
        dot_product  += tnz*in_target_points[6 * tid + 5];
        sn_norm = in_target_points[6 * tid + 3]*in_target_points[6 * tid + 3] + in_target_points[6 * tid + 4]*in_target_points[6 * tid + 4] + in_target_points[6 * tid + 5]*in_target_points[6 * tid + 5];
        
        if(dot_product < -0.5*sqrt(sn_norm))
        {
          //remove point from neighbor list
          //cur_neighbor_points_id_ptr[iii] = cur_neighbor_points_id_ptr[k - 1];
          for(int jjj = iii; jjj < k - 1; jjj++)
            cur_neighbor_points_id_ptr[jjj] = cur_neighbor_points_id_ptr[jjj + 1];
          cur_neighbor_points_id_ptr[k - 1] = -1;
          k--;
        }
      }
    }
    
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
void neighbor_points_spatial_grid_no_normal_weight_local_varying_radius_voting(OpKernelContext* context,
    const int n_source_points, const int* n_target_points, const int batch_size,
    const T* in_source_points, const T* in_target_points,
    const T* in_squared_radius, const int knn,
    int* neighbor_points_id_ptr) {
      
  //time_t start_time = clock();
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
  
  const TensorShape neighbor_points_distance_shape({batch_size, n_source_points, knn});
  Tensor neighbor_points_distance;
  OP_REQUIRES_OK(context, context->allocate_temp(sizeof(T)==4?DT_FLOAT:DT_DOUBLE, neighbor_points_distance_shape, &neighbor_points_distance));
  auto neighbor_points_distance_ptr = neighbor_points_distance.flat<T>().data();
  
	nthreads = batch_size;
  config = GetCudaLaunchConfig(nthreads, d);
  gather_cube_pointsb_local
		  <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			nthreads, n_target_points, in_target_points, MAX_POINTS_IN_CUBE, CUBES_TOTAL, cube_points_num_ptr, cube_points_indices_ptr);

  nthreads = batch_size * n_source_points;
  config = GetCudaLaunchConfig(nthreads, d);
  //printf("%d block and %d threads per block\n", config.block_count, config.thread_per_block);
  find_knn_grids_no_normalw_local_varying_radius_voting
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      nthreads, n_source_points, n_target_points, in_source_points, in_target_points, in_squared_radius, knn, CUBES_TOTAL, MAX_POINTS_IN_CUBE,
      cube_points_num_ptr, cube_points_indices_ptr, neighbor_points_id_ptr, neighbor_points_distance_ptr);
  cudaDeviceSynchronize();
  
  cudaError_t err2 = cudaGetLastError();
  if (err2 != cudaSuccess)
    printf("Cuda error: %s\n", cudaGetErrorString(err2));
  
}

//explicit instantiation
template void neighbor_points_spatial_grid_no_normal_weight_local_varying_radius_voting<float>(OpKernelContext* context,
    const int n_source_points, const int* n_target_points, const int batch_size,
    const float* in_source_points, const float* in_target_points,
    const float* in_squared_radius, const int knn,
    int* neighbor_points_id_ptr);
	
template void neighbor_points_spatial_grid_no_normal_weight_local_varying_radius_voting<double>(OpKernelContext* context,
    const int n_source_points, const int* n_target_points, const int batch_size,
    const double* in_source_points, const double* in_target_points,
    const double* in_squared_radius, const int knn,
    int* neighbor_points_id_ptr);

}  // namespace tensorflow
