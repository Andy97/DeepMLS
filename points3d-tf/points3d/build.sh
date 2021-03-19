#!/bin/sh
TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

nvcc -std=c++11 -c get_neighbor_points_spatial_grid_local_self.cu.cc $TF_CFLAGS -I /usr/local/cuda-10.0/include -x cu -Xcompiler -fPIC -D GOOGLE_CUDA=1 -I /usr/local -expt-relaxed-constexpr -DNDEBUG --expt-relaxed-constexpr --ptxas-options=--verbose -maxrregcount=55

nvcc -std=c++11 -c get_neighbor_points_spatial_grid_radius_weighting_voting.cu.cc $TF_CFLAGS -I /usr/local/cuda-10.0/include -x cu -Xcompiler -fPIC -D GOOGLE_CUDA=1 -I /usr/local -expt-relaxed-constexpr -DNDEBUG --expt-relaxed-constexpr --ptxas-options=--verbose -maxrregcount=55

g++ -std=c++11 -shared -o libpoints3d.so get_neighbor_points_spatial_grid_local_self.cc get_neighbor_points_spatial_grid_radius_weighting_voting.cc get_neighbor_points_spatial_grid_local_self.cu.o get_neighbor_points_spatial_grid_radius_weighting_voting.cu.o $TF_CFLAGS -I /usr/local/cuda-10.0/include -L /usr/local/cuda-10.0/lib64 -D GOOGLE_CUDA=1 $TF_LFLAGS -fPIC -lcudart

rm *.o