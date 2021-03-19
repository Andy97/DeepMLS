import os
import sys
import numpy as np
import tensorflow as tf
from scipy import ndimage
import mcubes
import math
import time

sys.path.append("points3d-tf")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
KNN_PREFERRED = 20
print("KNN={}".format(KNN_PREFERRED))
assert(KNN_PREFERRED >= 2)
print('====')
sys.stdout.flush()

struct2 = ndimage.generate_binary_structure(3, 3)

import argparse
parser = argparse.ArgumentParser(description='Marching Cube Input Output Arguments')
parser.add_argument('--i', type=str, help='input grid sdf file or points file or directory') #if args.i is dir, we will process all points file under the folder
parser.add_argument('--o', type=str, help='output marching cube file name')
parser.add_argument('--res', type=int, default=7, help='grid resolution')
parser.add_argument('--gpu', type=str, help='used gpu for program')
parser.add_argument('--scale', action='store_true', help='scale mesh back')
parser.add_argument('--overwrite', action='store_true', help='overwrite results')

args = parser.parse_args()
if(args.gpu is not None):
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

mls_points_ph = tf.placeholder(tf.float32)
mls_radius_ph = tf.placeholder(tf.float32)
query_points_ph = tf.placeholder(tf.float32)

from points3d import get_neighbor_spatial_grid_radius_v_voting

def rowwise_l2_norm_squared(feature):
  #assum input with size[n,f] out shape = [n,1]
  return tf.reduce_sum(tf.math.square(feature), axis=-1)

def write_obj(obj_filename, vertices, triangles, output_max_component=True, scale=None):
  #normalize to -1,1
  resolution = 2**args.res
  print("mesh resolution in {}".format(resolution))
  
  vertices /= (resolution / 2.0)
  vertices -= 1
  
  if(scale is not None):
    vertices *= scale
  
  with open(obj_filename,"w") as wf:
    for i in range(vertices.shape[0]):
      wf.write("v %lf %lf %lf\n" %(vertices[i][0], vertices[i][1], vertices[i][2]))
    for i in range(triangles.shape[0]):
      wf.write("f %d %d %d\n" %(triangles[i][0]+1, triangles[i][1]+1, triangles[i][2]+1))

def eval_sdf_from_mls(cur_projection, target_position, target_normal_normalized, src_points_num, target_points_num, \
  per_point_squared_ball_radius, s2t_neighbor=None): 
  #here we assert batch size=1
  if(s2t_neighbor == None):
    projection_points = tf.reshape(tf.transpose(tf.reshape(cur_projection, [1, -1, 3]), perm=[0,2,1]), [1, -1])
    target_points = tf.concat([tf.reshape(target_position, [-1, 3]), tf.reshape(target_normal_normalized, [-1, 3])], axis=1)
    #fetch corresponding patch, now we get indices matrix [batch_size*n, KNN]
    s2t_neighbor = tf.reshape(get_neighbor_spatial_grid_radius_v_voting(projection_points, target_points, [target_points_num],\
      tf.reshape(per_point_squared_ball_radius, [-1]), knn=KNN_PREFERRED), [-1,KNN_PREFERRED])
  
  invalid_index_mask = tf.cast(tf.less(s2t_neighbor, 0), dtype=s2t_neighbor.dtype)
  #make -1 to 0
  s2t_neighbor += invalid_index_mask
  s2t_neighbor = tf.expand_dims(s2t_neighbor, axis=-1)
  
  #get per vertex patch vertices position & normal [batch_size*n, KNN, 3]+[batch_size*n, KNN, 3]
  s2t_patch_pos = tf.gather_nd(target_position, s2t_neighbor)
  s2t_patch_normal = tf.gather_nd(target_normal_normalized, s2t_neighbor)
  s2t_patch_squared_radius = tf.gather_nd(tf.reshape(per_point_squared_ball_radius, [-1]), s2t_neighbor)
  #compute mls weights
  s2t_patch_pos_diff = tf.tile(tf.reshape(cur_projection,               [-1,1,3]),multiples=[1,KNN_PREFERRED,1]) - s2t_patch_pos
  s2t_patch_distance = rowwise_l2_norm_squared(s2t_patch_pos_diff)
  
  valid_neighbor_mask = tf.cast(1 - invalid_index_mask, dtype=s2t_patch_distance.dtype)
  #mls weight is stored in matrix with shape=[batch_size*n, KNN]
  s2t_mls_weight = -s2t_patch_distance / s2t_patch_squared_radius 
  s2t_mls_weight -= tf.tile(tf.expand_dims(s2t_mls_weight[:,0], axis=-1), multiples=[1, KNN_PREFERRED])
  s2t_mls_weight = tf.math.exp(s2t_mls_weight)
  
  #mask the patch verts out of ball
  s2t_mls_weight = s2t_mls_weight*valid_neighbor_mask
  #select projection points which are far away from surface
  s2t_mls_weight += tf.pad(tf.expand_dims(1 - valid_neighbor_mask[:,0], axis=-1), [[0,0],[0,KNN_PREFERRED-1]])
  #mls weights should also be normalized
  s2t_mls_weight = s2t_mls_weight / tf.tile(tf.reshape(tf.reduce_sum(s2t_mls_weight,axis=-1),[-1,1]),multiples=[1,KNN_PREFERRED])
  s2t_sdf = tf.reduce_sum(s2t_mls_weight*tf.reduce_sum(s2t_patch_pos_diff*s2t_patch_normal, axis=-1), axis=-1)
  return s2t_sdf

def get_sdf(mls_points, mls_radius, query_points):
  #assume mls_points in shape [n,6] and query_points in shape[q,3]
  n_mls_points = tf.shape(mls_points)[0]
  n_query_points = query_points.shape[0]
  mls_points = tf.cast(mls_points, tf.float64)
  query_points = tf.cast(query_points, tf.float64)
  assert(mls_radius is not None)
  mls_radius = tf.cast(mls_radius, tf.float64)
  return eval_sdf_from_mls(query_points, mls_points[:,:3], mls_points[:,3:], \
    n_query_points, n_mls_points, mls_radius)

class MLS_Surface:
  def __init__(self, mls_points, mls_radius, sess=None, sdf_node=None, scale=None):
    self.mls_points = mls_points
    self.mls_radius = mls_radius
    if(sess is None):
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(config=config)
    else:
      self.sess = sess
    
    self.sdf_node = sdf_node
    self.scale = scale
  
  def marching_cubes(self, output_obj_filename, voxel_resolution):
    voxel_dim = 2**voxel_resolution
    voxel_length = 2.0 / voxel_dim
    voxel_dim += 1
    
    grids = np.ones([voxel_dim, voxel_dim, voxel_dim])*1000000 #which means initial values are all not available (will be considered in our implemented marching_cubes_partial)
    assert(self.mls_radius is not None)
    
    mls_points_id = np.round((self.mls_points[:,:3] + 1) / voxel_length)
    
    np.clip(mls_points_id, 0, voxel_dim - 1, out = mls_points_id)
    mls_points_id = mls_points_id.astype(int)  
    active_grids = np.zeros([voxel_dim, voxel_dim, voxel_dim])
    active_grids[mls_points_id[:,0], mls_points_id[:, 1], mls_points_id[:, 2]] = 1.0;
    
    #might use larger dilation_layer_num
    max_radius = np.sqrt(np.max(self.mls_radius)) * 2
    dilation_layer_num = (int)(round(max_radius / voxel_length)) + 1
    print ("dilation layer number: ", dilation_layer_num)
    active_grids_large = ndimage.binary_dilation(active_grids, structure=struct2, iterations = dilation_layer_num)
    
    nonzeros = np.nonzero(active_grids_large)
    evaluated_points = np.stack(nonzeros, axis=1)*voxel_length - 1
    print("active number: ", nonzeros[0].shape)
        
    max_num_per_iter = 200000
    num_iter = math.ceil(nonzeros[0].shape[0] / max_num_per_iter)
    for i in range(0, num_iter):
      startid = i * max_num_per_iter
      endid = (i + 1) * max_num_per_iter
      if i == num_iter-1:
        endid = nonzeros[0].shape[0]
      if(self.sdf_node == None):
        sdf = self.sess.run(get_sdf(self.mls_points, self.mls_radius, evaluated_points[startid:endid]))
      else:
        sdf = self.sess.run(self.sdf_node, feed_dict={mls_points_ph:self.mls_points, mls_radius_ph:self.mls_radius, query_points_ph:evaluated_points[startid:endid]})
      grids[nonzeros[0][startid:endid], nonzeros[1][startid:endid], nonzeros[2][startid:endid]] = sdf
   
    print('outputfilename: ', output_obj_filename)
    vertices, triangles = mcubes.marching_cubes_partial(-grids, 0.0)
    write_obj(output_obj_filename, vertices, triangles, scale=self.scale)

def get_mls_points_radius_from_file(points_filename):
  mls_points = np.loadtxt(points_filename)
  if(mls_points.shape[1] != 6):
    #the xyz file does not contain normals: indicate this is not a mls file
    if(args.scale):
      return mls_points, None, None
    else:
      return mls_points, None
  if(os.path.exists(points_filename.replace(".xyz", "_radius.txt"))):
    mls_radius = np.loadtxt(points_filename.replace(".xyz", "_radius.txt"))
  else:
    print("radius of mls points not found")
    raise NotImplementedError
    mls_radius = None
  
  if(mls_radius.shape == ()):
    mls_radius = mls_radius*np.ones([mls_points.shape[0]])
  assert(mls_points.shape[0] == mls_radius.shape[0])
  
  if(args.scale):
    scale_file = points_filename.replace(".xyz", "_scale.txt")
    assert(os.path.exists(scale_file))
    scale = 1.0 / np.loadtxt(scale_file)
    return mls_points, mls_radius, scale
  
  #normal of mls points should be normalized
  mls_postion = mls_points[:,:3]
  mls_normal = mls_points[:,3:] / np.linalg.norm(mls_points[:,3:], axis=1, keepdims=True)
  mls_points = np.concatenate([mls_postion, mls_normal], axis=1)
  
  return mls_points, mls_radius

def process_folder(sess):
  #build graph in advance
  sdf_node = get_sdf(mls_points_ph, mls_radius_ph, query_points_ph)
  list = os.listdir(args.i)
  for file in list:
    if(file.endswith(".xyz")):
      points_filename = os.path.join(args.i, file)
      output_file_name = points_filename.replace(".xyz", "_mc.obj")
      if(not args.overwrite and os.path.exists(output_file_name)):
        continue
      if(args.scale):
        mls_points, mls_radius, scale = get_mls_points_radius_from_file(points_filename)
      else:
        mls_points, mls_radius = get_mls_points_radius_from_file(points_filename)
        scale = None
      if(mls_points.shape[1] != 6):
        continue
      mls_object = MLS_Surface(mls_points, mls_radius, sess=sess, sdf_node=sdf_node, scale=scale)
      
      if(args.overwrite or not os.path.exists(output_file_name)):
        mls_object.marching_cubes(output_file_name, args.res)

def main():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  
  start = time.time() 
  if(args.i is not None and os.path.isdir(args.i)):
    #marching cube for all files under directory
    return process_folder(sess)
  
  #process single input file
  assert(args.i is not None and args.i.endswith(".xyz"))
  points_filename = args.i
  
  if(args.scale):
    mls_points, mls_radius, scale = get_mls_points_radius_from_file(points_filename)
  else:
    mls_points, mls_radius = get_mls_points_radius_from_file(points_filename)
    scale = None
  assert(mls_points.shape[1] == 6)
  mls_object = MLS_Surface(mls_points, mls_radius, sess=sess, scale=scale)
  
  #marching cube from implicit distance fields defined by mls control points
  mls_object.marching_cubes("marching_cube_128.obj" if args.o is None else args.o, args.res)

  end = time.time()
  print("{}s elapsed".format(end - start))
  
if __name__ == '__main__':
  print('Program running...')
  main()