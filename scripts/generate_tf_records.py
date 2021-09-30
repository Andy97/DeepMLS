import os
import random
import struct

import numpy as np
import tensorflow as tf
import trimesh

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
def _int_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def get_octree_from_points(points, octree_depth=8):
  #get representative points for building the groundtruth octree
  assert(points.shape[1] == 6 or points.shape[1] == 3)
  points_pos = points[:,:3]
  octree_node_length = 2.0 / (2**octree_depth)
  octree_node_index = np.unique(np.clip(np.floor((points_pos + 1) / octree_node_length).astype(np.int32), a_min = 0, a_max = (2**octree_depth) - 1), axis=0)
  
  return octree_node_index

def write_data_to_tfrecords(sample_list, tfrecords_filename):
  #options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
  writer = tf.python_io.TFRecordWriter(tfrecords_filename)#, options=options
  n_data = len(sample_list)
  for i in range(n_data):
    if i % 100 == 0 or i == n_data-1:
      print('=========================\n')
      print('data loaded: {} / {}\n'.format(i+1, n_data))
      print('=========================\n')
    sample_category = sample_list[i][0]
    sample_id = sample_list[i][2]
    #print("{}:{}".format(sample_category, sample_id))
    signature = "{}:{}".format(sample_category, sample_id)
    
    #points_files contains the 100k points sampled from groundtruth surface, 3k input points to the network will be drawn from this set 
    points_file = os.path.join("occupancy_networks-master\\data\\ShapeNet.build", sample_category, "4_pointcloud", sample_id+".npz")
    assert(os.path.exists(points_file))
    points_dict = np.load(points_file)
    points_position = points_dict['points']
    points_normal = points_dict['normals']
    assert(points_position.shape == points_normal.shape)
    print("points in bbox:{} {}".format(points_position.min(), points_position.max()))
    assert(points_position.min() >= -1 and points_position.max() <= 1)
    points = np.concatenate((points_position, points_normal), axis=-1).astype(np.float32)
    assert(points.shape[0] == 100000 and points.shape[1] ==6)
    
    mesh_file = sample_list[i][1] + ".obj"
    assert(os.path.exists(mesh_file))
    #prepare ground truth octree for decoder supervision
    #sample 5M points to build octree as gt octree for decode shape
    mesh = trimesh.load(mesh_file)
    sampled_points = trimesh.sample.sample_surface(mesh, 5000000)[0]
    #scale to [-1,1] with 5% padding
    #sampled_points *= 1.9
    assert(sampled_points.min() >= -1 and sampled_points.max() <= 1)
    octree = get_octree_from_points(sampled_points, 8)
    assert(octree.min() >=0 and octree.max() <= 255)
    octree_points = (octree*np.array([256*256, 256, 1])).sum(axis=-1)
    assert(octree_points.dtype == np.int)
    assert(octree_points.min() >=0 and octree_points.max() <= 16777215)
    #octree_points_file = sample_list[i][1] + "_octreep.npz"
    #np.savez_compressed(octree_points_file, octree_points=octree_points)
    #print("{} octree points".format(octree_points.shape))
    
    octree_points_seq_length = 2000
    #pad octree_points
    octree_points = np.reshape(octree_points, [-1])
    octree_points_pad_elements = (octree_points_seq_length - (octree_points.shape[0] % octree_points_seq_length)) % octree_points_seq_length
    octree_points = np.reshape(np.pad(octree_points, (0, octree_points_pad_elements), mode='constant', constant_values=octree_points[0]), [-1, octree_points_seq_length])
    
    sdf_samples_file = sample_list[i][1] + ".mat.bin"
    if(not os.path.exists(sdf_samples_file)):
      os.system("vdb_tsdf {} {}".format(mesh_file, sdf_samples_file))
    assert(os.path.exists(sdf_samples_file))
    
    sdf_samples = np.reshape(np.fromfile(sdf_samples_file, dtype=np.float32), [-1, 7])
    assert(sdf_samples.shape[1] == 7)
    sdf_points = sdf_samples[:,:3]
    assert(sdf_points.min() >= -1 and sdf_points.max() <= 1)    
    
    #first get samples on 64^3 grid
    x = sdf_samples[:,0]
    y = sdf_samples[:,1]
    z = sdf_samples[:,2]
    coordx = np.round((x+1)*128).astype(int)
    coordy = np.round((y+1)*128).astype(int)
    coordz = np.round((z+1)*128).astype(int)
    depth6_ind = np.where(np.logical_and(np.logical_and(coordx % 4 == 0, coordy % 4 == 0), coordz % 4 == 0))
    depth6_sdf_samples = sdf_samples[depth6_ind]
    #print("{} samples in depth 6".format(depth6_sdf_samples.shape))
    
    #filter sdf samples with sdf > 2/32
    sdf_samples = sdf_samples[np.where(np.abs(sdf_samples[:,3]) < 2.0/32)]
    x = sdf_samples[:,0]
    y = sdf_samples[:,1]
    z = sdf_samples[:,2]
    coordx = np.round((x+1)*128).astype(int)
    coordy = np.round((y+1)*128).astype(int)
    coordz = np.round((z+1)*128).astype(int)
    depth7_ind = np.where(np.logical_and(np.logical_and(coordx % 2 == 0, coordy % 2 == 0), coordz % 2 == 0))
    depth7_sdf_samples = sdf_samples[depth7_ind]
    #print("{} samples in depth 7".format(depth7_sdf_samples.shape))
    
    def efficient_padding(sdf_samples):
      xyz_coord = np.round((sdf_samples[:,:3] + 1)*128).astype(np.int32)
      assert(xyz_coord.min() >= 0 and xyz_coord.max() <= 256)
      xyz_id = (xyz_coord*np.array([257*257, 257, 1])).sum(axis=-1)
      sdf_pad_elements = (1000 - (xyz_id.shape[0] % 1000)) % 1000
      
      xyz_id = np.pad(xyz_id, (0, sdf_pad_elements), mode='constant', constant_values=-100) #should be carefully dealt when parsing
      sdf_values = np.pad(np.reshape(sdf_samples[:,3:], [-1]), (0, 4*sdf_pad_elements), mode='constant', constant_values=-100)
      assert(xyz_id.shape[0] % 1000 == 0 and xyz_id.dtype == np.int32)
      assert(sdf_values.shape[0] == xyz_id.shape[0]*4 and sdf_values.dtype==np.float32)
      return np.reshape(xyz_id, [-1,1000]), np.reshape(sdf_values, [-1, 4000])
      
    depth7_coord, depth7_sdf_samples = efficient_padding(depth7_sdf_samples)
    depth6_coord, depth6_sdf_samples = efficient_padding(depth6_sdf_samples)
    assert(depth7_coord.dtype == np.int32 and depth6_coord.dtype == np.int32)
    assert(depth7_sdf_samples.dtype == np.float32 and depth6_sdf_samples.dtype == np.float32)
    
    sequence_feature = {"depth7_coord":tf.train.FeatureList(feature=[_int_feature(coord_row.tolist()) for coord_row in depth7_coord]),
                      "depth6_coord":tf.train.FeatureList(feature=[_int_feature(coord_row.tolist()) for coord_row in depth6_coord]),
                      "sdf_samples_depth7":tf.train.FeatureList(feature=[_float_feature(sdf_row.tolist()) for sdf_row in depth7_sdf_samples]),
                      "sdf_samples_depth6":tf.train.FeatureList(feature=[_float_feature(sdf_row.tolist()) for sdf_row in depth6_sdf_samples]),
                      "octree_points_idx":tf.train.FeatureList(feature=[_int_feature(idx_row.tolist()) for idx_row in octree_points])}
    feature = {'points': _bytes_feature(np.reshape(np.transpose(points), [-1]).astype(np.float16).tobytes()),
               "filename":_bytes_feature(signature.encode('utf8')),
               }
    example = tf.train.SequenceExample(context=tf.train.Features(feature=feature),
              feature_lists=tf.train.FeatureLists(feature_list=sequence_feature))
    writer.write(example.SerializeToString())
  writer.close()

def main():
  cat = ['02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04401088', '04530566']
  rootpc = "occupancy_networks-master\\data\\ShapeNet.build\\"
  
  list=[]
  for item in cat:
    print("working on {}".format(item))
    dir_point = os.path.join(rootpc, item, 'watertight_normalized')
    assert(os.path.exists(dir_point))
    
    #training data split file follow occupancy networks
    split_file = os.path.join("convolutional_occupancy_networks-master\\data\\ShapeNet", item, "train.lst") 
    assert(os.path.exists(split_file))
    split_file = open(split_file, "r")
    fns = [line.replace("\r", "").replace("\n", "") for line in split_file]
    split_file.close()
    
    assert(len(fns) != 0)
    for fn in fns:
      list.append((item, os.path.join(dir_point, fn), fn))
  
  
  print("{} samples total".format(len(list)))
  random.shuffle(list)
  
  tfrecords_filename = 'ShapeNet_points_{}_w_octree_grid_occupancy_compress.tfrecords'.format("all_train")
  write_data_to_tfrecords(list, tfrecords_filename)

if __name__ == '__main__':
  main()
