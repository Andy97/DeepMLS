import sys
import tensorflow as tf
import numpy as np
sys.path.append("Octree/ocnn-tf")
from libs import *

def IntList_to_Bytes(int_list):
    #list_bytes = struct.pack('i'*len(int_list), *int_list)
    x = np.array(int_list, dtype=np.int32)
    list_bytes = x.tobytes()
    return list_bytes

def DoubleList_to_Bytes(float_list):
    #list_bytes = struct.pack('d'*len(float_list), *float_list)
    x = np.array(float_list, dtype=np.float64)
    list_bytes = x.tobytes()
    return list_bytes

def Float32List_to_Bytes(float_list):
    #list_bytes = struct.pack('f'*len(float_list), *float_list)
    x = np.array(float_list, dtype=np.float32)
    list_bytes = x.tobytes()
    return list_bytes

class PointsPreprocessor:
  def __init__(self, depth, points_num, full_depth=2, sample_grid_points_number=None, data_sources=-1, return_gt_points=False, noise_stddev=0.0095):
    self._depth = depth
    self._full_depth = full_depth
    self._points_num = points_num
    self._sample_grid_points_number = sample_grid_points_number
    self._data_sources = data_sources
    self._return_gt_points = return_gt_points
    self._noise_stddev = noise_stddev


  def __call__(self, record):    
    points_array_f16, filename, sdf_samples_depth6, sdf_samples_depth7, depth6_coord, depth7_coord, octree_d7_points_idx = self.parse_example_array(record)
    
    octree_points_d7_x = octree_d7_points_idx // 65536
    octree_points_d7_y = (octree_d7_points_idx // 256) % 256
    octree_points_d7_z = octree_d7_points_idx % 256
    octree_points_d7 = tf.cast(tf.stack([octree_points_d7_x, octree_points_d7_y, octree_points_d7_z], axis=1), tf.float32) / 128.0 - 1
    
    gt_points = tf.cast(tf.transpose(tf.reshape(points_array_f16, [6, 100000])), tf.float32)
       
    #sample 3000 points and add noise
    selected_points_idx = tf.random.uniform(shape=[3000], minval=0, maxval=100000, dtype=tf.int32)
    selected_points = tf.gather(gt_points, selected_points_idx)
    #add noise
    assert(self._points_num == 3000)
    if(self._noise_stddev > 1e-6):
      selected_points_position = selected_points[:,:3] + tf.random.normal([3000,3], stddev=self._noise_stddev)#0.0095
    else:
      selected_points_position = selected_points[:,:3]
    #selected_points_normal = selected_points[:,3:]
    #selected_points = tf.concat([selected_points_position, selected_points_normal], axis=-1)
    
    points = points_new(selected_points_position, [], selected_points_position, [])
    input_octree = points2octree(points, depth=self._depth, full_depth=self._full_depth, 
                           node_dis=False, split_label=True) #node_feature=True,
    
    gt_octree = points2octree(points_new(octree_points_d7, [], tf.ones(tf.shape(octree_points_d7_x)), []), depth=self._depth, full_depth=self._full_depth, 
                           node_dis=False, split_label=True) #node_feature=True, 
    if(self._data_sources == 6):
      sdf_samples = sdf_samples_depth6
      sdf_samples_id = depth6_coord
    elif(self._data_sources == 7):
      sdf_samples = sdf_samples_depth7
      sdf_samples_id = depth7_coord
    else:
      raise NotImplementedError
    
    if(self._sample_grid_points_number is None):
      raise NotImplementedError
    else:
      padding_limit = self._sample_grid_points_number
      #random select sample_grid_points_number samples
      sdf_samples = tf.reshape(sdf_samples, [-1,4])
      num_valid_samples = tf.reduce_sum(tf.cast(tf.greater_equal(sdf_samples_id, 0), tf.int32))
      
      #randomly select points(uniformly)
      selected_idx = tf.random.shuffle(tf.range(num_valid_samples))[:padding_limit]
      selected_samples = tf.gather(sdf_samples, selected_idx)
      selected_samples_coord = tf.gather(sdf_samples_id, selected_idx)
      
      selected_samples_coordx = selected_samples_coord // 66049
      selected_samples_coordy = (selected_samples_coord // 257) % 257
      selected_samples_coordz = selected_samples_coord % 257
      
      selected_position = tf.cast(tf.stack([selected_samples_coordx, selected_samples_coordy, selected_samples_coordz], axis=1), tf.float32) / 128.0 - 1.0
      selected_sdf = selected_samples
      #pad if needed
      selected_position = tf.pad(selected_position, [[0, padding_limit - tf.shape(selected_position)[0]],[0,0]], constant_values=-100) 
      selected_sdf      = tf.pad(selected_sdf,      [[0, padding_limit - tf.shape(selected_sdf)[0]],[0,0]],      constant_values=0)
      if(self._return_gt_points):
        return input_octree, gt_octree, tf.transpose(selected_points), filename, selected_position, selected_sdf, gt_points
      else:
        return input_octree, gt_octree, tf.transpose(selected_points), filename, selected_position, selected_sdf
             
  def parse_example_array(self, record):
    features = { 'points': tf.FixedLenFeature([], tf.string),
                 "filename": tf.FixedLenFeature([], tf.string)}
    SeqFeatures = {"sdf_samples_depth6": tf.FixedLenSequenceFeature([4000], tf.float32),
                   "sdf_samples_depth7": tf.FixedLenSequenceFeature([4000], tf.float32),
                   "depth6_coord": tf.FixedLenSequenceFeature([1000], tf.int64),
                   "depth7_coord": tf.FixedLenSequenceFeature([1000], tf.int64),
                   "octree_points_idx": tf.FixedLenSequenceFeature([2000], tf.int64)}
    parsed_all = tf.parse_single_sequence_example(record, context_features=features, sequence_features=SeqFeatures)
    parsed = parsed_all[0]
    feature_parsed = parsed_all[1]
    return tf.decode_raw(parsed['points'], tf.float16), parsed['filename'], feature_parsed['sdf_samples_depth6'], feature_parsed['sdf_samples_depth7'], tf.reshape(feature_parsed['depth6_coord'], [-1]), tf.reshape(feature_parsed['depth7_coord'], [-1]), tf.reshape(feature_parsed['octree_points_idx'], [-1])

def points_dataset_AE(record_name, batch_size, points_num, depth=5, full_depth=2, shuffle_data=True, sample_grid_points_number=None, data_sources=-1, return_gt_points=False, noise_stddev=0.0095):
  def merge_octrees(input_octrees, gt_octrees, points_arrays, filenames, grid_points, grid_fvalue):
    input_octree = octree_batch(input_octrees)
    gt_octree = octree_batch(gt_octrees)
    return input_octree, gt_octree, points_arrays, filenames, grid_points, grid_fvalue
  
  def merge_octrees_gtpoints(input_octrees, gt_octrees, points_arrays, filenames, grid_points, grid_fvalue, gt_points):
    input_octree = octree_batch(input_octrees)
    gt_octree = octree_batch(gt_octrees)
    return input_octree, gt_octree, points_arrays, filenames, grid_points, grid_fvalue, gt_points
  
  if("all_test_1000" in record_name):
    record_num = 1000
  else:
    record_num = sum(1 for _ in tf.python_io.tf_record_iterator(record_name))
  shuffle_buffer_size = min(1000, 2*record_num)
  print("dataset using data sources:{}".format(data_sources))
  print("{} records in {}".format(record_num, record_name))
  if(shuffle_data):
    print("shuffle buffer size = {}".format(shuffle_buffer_size))
  with tf.name_scope('points_dataset'):
    preprocess = PointsPreprocessor(depth, points_num, full_depth, sample_grid_points_number=sample_grid_points_number, data_sources=data_sources, return_gt_points=return_gt_points, noise_stddev=noise_stddev)
    if(return_gt_points):
      merge_octrees_op = merge_octrees_gtpoints
    else:
      merge_octrees_op = merge_octrees
    if(shuffle_data):
      return (record_num,) + tf.data.TFRecordDataset([record_name]).repeat().shuffle(shuffle_buffer_size) \
                  .map(preprocess, num_parallel_calls=8) \
                  .batch(batch_size).map(merge_octrees_op, num_parallel_calls=8) \
                  .prefetch(8).make_one_shot_iterator().get_next()
    else:
      return (record_num,) + tf.data.TFRecordDataset([record_name]).repeat() \
                  .map(preprocess, num_parallel_calls=8) \
                  .batch(batch_size).map(merge_octrees_op, num_parallel_calls=8) \
                  .prefetch(8).make_one_shot_iterator().get_next()

def points_dataset_AE_multiple_GPU(record_name, batch_size, points_num, depth=5, full_depth=2, gpu_num=1, sample_grid_points_number=None, data_sources=-1, noise_stddev=0.0095):
  def merge_octrees(input_octrees, gt_octrees, points_arrays, filenames, grid_points, grid_fvalue):
    gpu_batch_size = batch_size // gpu_num
    return_list = []
    for i in range(gpu_num):
      return_list.append(octree_batch(input_octrees[i*gpu_batch_size:(i+1)*gpu_batch_size]))
    for i in range(gpu_num):
      return_list.append(octree_batch(gt_octrees[i*gpu_batch_size:(i+1)*gpu_batch_size]))
    return_list += [points_arrays, filenames, grid_points, grid_fvalue]
    return return_list
  
  if("all_train" in record_name):
    record_num = 30661
  else:
    record_num = sum(1 for _ in tf.python_io.tf_record_iterator(record_name))
  shuffle_buffer_size = min(1000, 2*record_num)
  print("multi-GPU dataset using data sources:{}".format(data_sources))
  print("{} records in {}".format(record_num, record_name))
  print("shuffle buffer size = {}".format(shuffle_buffer_size))
  with tf.name_scope('points_dataset'):
    preprocess = PointsPreprocessor(depth, points_num, full_depth, sample_grid_points_number=sample_grid_points_number, data_sources=data_sources, noise_stddev=noise_stddev)
    return (record_num,) + tf.data.TFRecordDataset([record_name]).repeat().shuffle(shuffle_buffer_size) \
                  .map(preprocess, num_parallel_calls=8) \
                  .batch(batch_size).map(merge_octrees, num_parallel_calls=8) \
                  .prefetch(8).make_one_shot_iterator().get_next()