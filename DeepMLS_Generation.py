import os
import sys
import numpy as np
import tensorflow as tf
import argparse
from utils import *
import tqdm

sys.path.append("points3d-tf")
from pointsdataset.points_dataset_octreed7 import *
from network_architecture import *
from points3d import get_neighbor_spatial_grid_local_self
from points3d import get_neighbor_spatial_grid_radius_v_voting

parser = argparse.ArgumentParser(description='Point-based Shape Generation')
parser.add_argument('config', type=str, metavar='N', help='config json file')
parser.add_argument('--test', action='store_true', help='forward the test data(inference)')
args = parser.parse_args()
config = config_reader(args.config)

DBL_EPSILON = 1E-22

class param:
  def __init__(self, config):
    self.train_data = config['train_data']
    self.train_batch_size = config['train_batch_size']
    self.learning_rate = config['learning_rate']
    if('learning_rate_lower_bound' in config):
      self.learning_rate_lower_bound = config['learning_rate_lower_bound']
    else:
      self.learning_rate_lower_bound = 1e-4
    self.lr_decay_epochs = config['lr_decay_epochs']
    
    #choose different sdf samples for training(generate on depth-6 grids or depth-7 grids)
    self.sdf_data_sources = config['sdf_data_sources']   
    assert(self.sdf_data_sources == 6 or self.sdf_data_sources == 7)
    if(self.sdf_data_sources == 6):
      self.max_training_epochs = config['max_training_epochs_d6']
    elif(self.sdf_data_sources == 7):
      self.max_training_epochs = config['max_training_epochs_d7']
    
    self.exp_folder = config['exp_name']
    self.ckpt = config['ckpt']
    
    self.test = args.test
    self.gpu = config['gpu']
    self.num_of_gpus = len(self.gpu.split(","))
    self.num_of_input_points = config['num_of_input_points']
    self.num_neighbors_to_search = config['num_neighbors_to_search']
    
    #loss weighting
    self.sdf_loss_weight = config['sdf_loss_weight']
    self.sdf_grad_loss_weight = config['sdf_grad_loss_weight']
    self.geo_reg = config['geo_reg']
    self.repulsion_weight = config['repulsion_weight']
    self.normal_norm_reg_weight = config['normal_norm_reg_weight']
    self.patch_radius_smoothness = config['patch_radius_smoothness']
    self.octree_split_loss_weighting = config['octree_split_loss_weighting']
    self.weight_decay = config['weight_decay']
    
    #for octree data
    self.input_normal_signals = config['input_normal_signals']
    if(self.input_normal_signals):
      self.channel = 4
    else:
      self.channel = 3
    self.depth = config['octree_depth']
    if("decoder_octree_depth" in config):
      self.decoder_octree_depth = config['decoder_octree_depth']
    else:
      self.decoder_octree_depth = 6
    print("decoder_octree_depth setting to {}".format(self.decoder_octree_depth))
    
    #predict how many mls points in each non-empty octree leaf node
    self.points_per_node = config['points_per_node']
    self.constant_radius = config['constant_radius']    
    assert(self.sdf_data_sources == 6 or self.sdf_data_sources == 7)
    if(self.sdf_data_sources == 7):
      self.sdf_loss_weight *= 4
    
    self.sdf_samples_each_iter = config['sdf_samples_each_iter']
    print("using {} sdf samples for evaluation in each iteration".format(self.sdf_samples_each_iter))
        
    self.node_receptive_field = config['node_receptive_field']
    print("node receptive field times: {}".format(self.node_receptive_field))
    
    self.radius_range = config['radius_range']
    print("radius range = {}".format(self.radius_range))
        
    if("noise_stddev" in config):
      self.noise_stddev = config['noise_stddev']
    else:
      #our model is normaled to [-1, 1]^3 bounding box with 5% padding
      #to achieve same noise level with conv-onet
      self.noise_stddev = 0.0095

FLAGS = param(config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
KNN_PREFERRED = FLAGS.num_neighbors_to_search
print("KNN={}".format(KNN_PREFERRED))
assert(KNN_PREFERRED >= 2)

train_batch_size_gpu = FLAGS.train_batch_size // FLAGS.num_of_gpus
assert(train_batch_size_gpu * FLAGS.num_of_gpus == FLAGS.train_batch_size)
print("utilize {} gpus with total training batch size={}".format(FLAGS.num_of_gpus, FLAGS.train_batch_size))

lr_placeholder = tf.placeholder(tf.float32)
print('=====')
sys.stdout.flush()

#normal unit norm regularization to avoid degenerated normal (should be very gentle)
def normal_unit_norm_regularization(normals, points_segment, points_num):
  #input shape [batch_size*n,3]
  per_vertex_loss = tf.math.square(tf.reduce_sum(tf.math.square(normals), axis=1) - 1)
  loss = tf.segment_sum(per_vertex_loss, points_segment) / tf.cast(points_num, normals.dtype)
  return tf.reduce_mean(loss)

def self_regularization_loss(predict_position_matrix, predict_normal_normalized, points_segment, points_num, per_point_squared_ball_radius):
  #first get neighbor info and weights
  predict_points = tf.concat([tf.reshape(predict_position_matrix, [-1, 3]), tf.reshape(predict_normal_normalized, [-1, 3])], axis=1)
  
  p2p_neighbor = tf.reshape(get_neighbor_spatial_grid_local_self(predict_points, tf.cumsum(points_num), knn=KNN_PREFERRED), [-1,KNN_PREFERRED])
  invalid_index_mask = tf.cast(tf.less(p2p_neighbor, 0), dtype=p2p_neighbor.dtype)
  #make -1 to 0
  p2p_neighbor += invalid_index_mask
  p2p_neighbor = tf.expand_dims(p2p_neighbor, axis=-1)
  p2p_neighbor = tf.stop_gradient(p2p_neighbor)
  
  p2p_patch_pos = tf.gather_nd(predict_position_matrix, p2p_neighbor)
  p2p_patch_normal = tf.gather_nd(predict_normal_normalized, p2p_neighbor)
  per_point_radius = tf.math.sqrt(per_point_squared_ball_radius + 1e-19)
  p2p_patch_radius = tf.gather_nd(tf.reshape(per_point_radius, [-1]), p2p_neighbor)
  
  p2p_patch_pos_diff = tf.tile(tf.reshape(predict_position_matrix, [-1,1,3]), multiples=[1,KNN_PREFERRED,1]) - p2p_patch_pos
  p2p_patch_pos_diff_norm_squared = rowwise_l2_norm_squared(p2p_patch_pos_diff)
  p2p_patch_pos_diff_norm = tf.math.sqrt(p2p_patch_pos_diff_norm_squared + 1e-10)
  
  #using dot product [batch_size, n]
  p2p_patch_normal_dot = tf.reduce_sum(tf.tile(tf.reshape(predict_normal_normalized, [-1,1,3]),multiples=[1,KNN_PREFERRED,1])*p2p_patch_normal, axis=-1)
  p2p_patch_distance = rowwise_l2_norm_squared(p2p_patch_pos_diff)
  
  if(FLAGS.constant_radius):
    squared_ball_radius = per_point_squared_ball_radius[0]
  else:
    squared_ball_radius = tf.gather_nd(per_point_squared_ball_radius, p2p_neighbor)
  
  valid_neighbor_mask = tf.cast(1 - invalid_index_mask, dtype=p2p_patch_distance.dtype)
  p2p_patch_distance += squared_ball_radius * (1 - p2p_patch_normal_dot)
  
  #mls weight is stored in matrix with shape=[batch_size*n, KNN]
  p2p_mls_weight = tf.math.exp(-p2p_patch_distance / squared_ball_radius)
  p2p_mls_weight = p2p_mls_weight*valid_neighbor_mask
  #mls weights should also be normalized
  #p2p_mls_weight = p2p_mls_weight / tf.tile(tf.reshape(tf.reduce_sum(p2p_mls_weight,axis=-1),[-1,1]),multiples=[1,KNN_PREFERRED])
    
  stop_regularization_weight_gradient = True
  if(stop_regularization_weight_gradient):
    p2p_mls_weight = tf.stop_gradient(p2p_mls_weight)
  p2p_mls_weight_sum = tf.reduce_sum(p2p_mls_weight, axis=-1)
  p2p_mls_weight_sum_dim3 = tf.tile(tf.expand_dims(p2p_mls_weight_sum, axis=-1), multiples=[1,3])
  
  Laplacian_radius = tf.math.square(p2p_mls_weight_sum*tf.reshape(per_point_radius, [-1]) - tf.reduce_sum(p2p_mls_weight*p2p_patch_radius, axis=-1)) / tf.math.square(p2p_mls_weight_sum + 1e-6)
  
  #local geometry regularization(planar shape)
  #get neighbor point distance to tangent plane in shape[batch_size*n_point, KNN_PREFERRED]
  p2p_patch_pos_diff_normal = tf.reduce_sum(p2p_patch_pos_diff*tf.tile(tf.reshape(predict_normal_normalized, [-1, 1, 3]), multiples=[1, KNN_PREFERRED, 1]), axis=-1)
  p2p_patch_pos_diff_tangent = p2p_patch_pos_diff - tf.tile(tf.expand_dims(p2p_patch_pos_diff_normal, axis=-1), multiples=[1,1,3])*tf.tile(tf.reshape(predict_normal_normalized, [-1, 1, 3]), multiples=[1, KNN_PREFERRED, 1])
  p2p_patch_pos_diff_tangent_norm_squared = rowwise_l2_norm_squared(p2p_patch_pos_diff_tangent)
  p2p_patch_pos_diff_tangent_norm = tf.math.sqrt(p2p_patch_pos_diff_tangent_norm_squared + 1e-10)
  
  tangent_distance = tf.math.square(p2p_patch_pos_diff_normal)
  if(stop_regularization_weight_gradient):
    p2p_mls_weight_plane = tf.stop_gradient(p2p_mls_weight*tf.math.exp(-tangent_distance/squared_ball_radius))
  else:
    p2p_mls_weight_plane = p2p_mls_weight*tf.math.exp(-tangent_distance/squared_ball_radius)
  
  tangent_distance *= p2p_mls_weight_plane
  
  #-d repulsion
  if(True):
    #repulsion in tangent direction
    repulsion_vec = - p2p_patch_pos_diff_tangent_norm / (octree_mls_points_squared_radius()**0.5)
  else:
    repulsion_vec = - p2p_patch_pos_diff_norm / (octree_mls_points_squared_radius()**0.5)
    
  #then we get a tensor with shape [\sum(all_mls_points)] 
  repulsion_force = tf.reduce_mean(p2p_mls_weight * repulsion_vec, axis=-1)
  repulsion_force = tf.segment_sum(repulsion_force, points_segment) / tf.cast(points_num, repulsion_force.dtype)
  
  radius_smoothness_loss = tf.segment_sum(Laplacian_radius, points_segment) / tf.cast(points_num, Laplacian_radius.dtype)  
  local_plane_regularization = tf.segment_sum(tf.reduce_sum(tangent_distance, axis=-1), points_segment) / tf.cast(points_num, repulsion_force.dtype)
  
  repulsion_force = tf.reduce_mean(repulsion_force)
  local_plane_regularization = tf.reduce_mean(local_plane_regularization)
  radius_smoothness_loss = tf.reduce_mean(radius_smoothness_loss)
    
  return repulsion_force, local_plane_regularization, radius_smoothness_loss

def eval_sdf_from_mls(src_position, target_position, target_normal_normalized, per_point_squared_ball_radius, s2t_neighbor):
  invalid_index_mask = tf.cast(tf.less(s2t_neighbor, 0), dtype=s2t_neighbor.dtype)
  
  #make -1 to 0(index -1 indicates invalid neighbor index, which means we cannot find up to K neighbors)
  s2t_neighbor += invalid_index_mask
  s2t_neighbor = tf.expand_dims(s2t_neighbor, axis=-1)
  
  #get per vertex patch vertices position & normal [batch_size*n, KNN, 3]+[batch_size*n, KNN, 3]
  s2t_patch_pos = tf.gather_nd(target_position, s2t_neighbor)
  s2t_patch_normal = tf.gather_nd(target_normal_normalized, s2t_neighbor)
  #compute mls weights
  s2t_patch_pos_diff    = tf.tile(tf.reshape(src_position,               [-1,1,3]),multiples=[1,KNN_PREFERRED,1]) - s2t_patch_pos  
  s2t_patch_distance = rowwise_l2_norm_squared(s2t_patch_pos_diff)
  
  valid_neighbor_mask = tf.cast(1 - invalid_index_mask, dtype=s2t_patch_distance.dtype)
  
  #avoid divide by zero error
  if(FLAGS.constant_radius):
    s2t_mls_weight = -s2t_patch_distance / per_point_squared_ball_radius[0]
  else:
    s2t_mls_weight = -s2t_patch_distance / tf.gather_nd(tf.reshape(per_point_squared_ball_radius, [-1]), s2t_neighbor)
  
  s2t_mls_weight -= tf.stop_gradient(tf.tile(tf.expand_dims(s2t_mls_weight[:,0], axis=-1), multiples=[1, KNN_PREFERRED]))
  
  #mls weight is stored in matrix with shape=[batch_size*n, KNN]
  s2t_mls_weight = s2t_mls_weight*valid_neighbor_mask
  s2t_mls_weight = tf.math.exp(s2t_mls_weight)
  s2t_mls_weight = s2t_mls_weight*valid_neighbor_mask
    
  #mls weights should also be normalized
  s2t_mls_weight = s2t_mls_weight / tf.tile(tf.reshape(tf.reduce_sum(s2t_mls_weight,axis=-1),[-1,1]),multiples=[1,KNN_PREFERRED])
  
  s2t_sdf = tf.reduce_sum(s2t_mls_weight*tf.reduce_sum(s2t_patch_pos_diff*s2t_patch_normal, axis=-1), axis=-1)
  #get weighted average patch points
  s2t_patch_pos_mean = tf.reduce_sum(tf.tile(tf.expand_dims(s2t_mls_weight, axis=-1),multiples=[1,1,3])*s2t_patch_pos, axis=1)
  #get weighted average normal: gradient
  s2t_patch_normal_mean = tf.reduce_sum(tf.tile(tf.expand_dims(s2t_mls_weight, axis=-1),multiples=[1,1,3])*s2t_patch_normal, axis=1)
  s2t_sdf_grad = tf.math.l2_normalize(s2t_patch_normal_mean, axis=1)
  
  return s2t_sdf_grad, s2t_sdf

def MLS_sdf_Loss_Pack(evaluate_position_matrix, predict_position_matrix, predict_normal_normalized, evaluate_points_num, ps2p_neighbor_index, \
  per_point_squared_ball_radius, sdf_precompute):
  
  evaluate_position_matrix = tf.reshape(evaluate_position_matrix, [-1, 3])
  sdf_precompute = tf.reshape(sdf_precompute, [-1, 4])
  
  #select valid grid points out of padded data
  valid_data_mask = tf.cast(tf.greater(evaluate_position_matrix[:,0], -5), dtype=evaluate_position_matrix.dtype)  
    
  predict_sdf_grad, predict_sdf = \
    eval_sdf_from_mls(evaluate_position_matrix, predict_position_matrix, predict_normal_normalized, per_point_squared_ball_radius, ps2p_neighbor_index)
  
  valid_data_mask = tf.reshape(valid_data_mask, [-1, evaluate_points_num])
  valid_data_mask = tf.stop_gradient(valid_data_mask)
  #select the normal for grid points
  predict_sdf_grad   = tf.reshape(predict_sdf_grad, [-1, evaluate_points_num, 3])
  predict_sdf = tf.reshape(predict_sdf, [-1, evaluate_points_num])  
      
  #using precomputed sdf value and gradients
  gt_sdf = tf.reshape(sdf_precompute[:,0], [-1, evaluate_points_num])
  gt_sdf_grad = tf.reshape(sdf_precompute[:,1:],   [-1, evaluate_points_num, 3])
  
  gt_sdf = tf.stop_gradient(gt_sdf)
  gt_sdf_grad = tf.stop_gradient(tf.nn.l2_normalize(gt_sdf_grad, axis=-1))
  
  #L2 loss of difference between predict sdf and gt sdf (as well as gradients)
  grid_sdf_squared_diff = valid_data_mask * tf.math.square(gt_sdf - predict_sdf)
  
  if(False):
    #l2 loss
    grid_sdf_grad_squared_diff = valid_data_mask*rowwise_l2_norm_squared(gt_sdf_grad - predict_sdf_grad)
  else:
    #using dot product
    grid_sdf_grad_squared_diff = valid_data_mask*(1 - tf.reduce_sum(gt_sdf_grad*predict_sdf_grad, axis=-1))
        
  valid_count = tf.reduce_sum(tf.reshape(valid_data_mask, [-1, evaluate_points_num]), axis=-1)
  
  predict_sdf_diff_loss = tf.reduce_sum(grid_sdf_squared_diff, axis=-1) / valid_count
  predict_sdf_grad_loss = tf.reduce_sum(grid_sdf_grad_squared_diff, axis=-1) / valid_count
  
  predict_sdf_diff_loss = tf.reduce_mean(predict_sdf_diff_loss)
  predict_sdf_grad_loss = tf.reduce_mean(predict_sdf_grad_loss)
  
  return predict_sdf_diff_loss, predict_sdf_grad_loss

def network_loss(predict_points, points_segment, points_num, per_point_squared_ball_radius, sampled_position_matrix, sdf_precompute, name="network_loss"):
  #prepare ingredients for cooking
  predict_position_matrix = predict_points[:,:3]
  predict_normal_matrix = predict_points[:,3:]
  predict_normal_normalized = tf.math.l2_normalize(predict_normal_matrix, axis=1)
    
  loss_dict = {}
  train_loss = 0
  
  sampled_points = sampled_position_matrix
  evaluated_points_num = tf.cast(tf.shape(sampled_points)[1], tf.int32)
  sampled_points = tf.reshape(tf.transpose(sampled_points, perm=[0,2,1]), [-1, evaluated_points_num*3])
  
  #compute neighbor mls points for sdf samples
  ps2p_neighbor_index = tf.reshape(get_neighbor_spatial_grid_radius_v_voting(sampled_points, predict_points, tf.cumsum(points_num),\
        per_point_squared_ball_radius, knn=KNN_PREFERRED),[-1, KNN_PREFERRED])
  
  if(FLAGS.sdf_loss_weight > DBL_EPSILON or FLAGS.sdf_grad_loss_weight > DBL_EPSILON):
    mls_sdf_loss, mls_sdf_grad_loss = MLS_sdf_Loss_Pack(tf.reshape(sampled_position_matrix, [-1,3]), predict_position_matrix, predict_normal_normalized, evaluated_points_num, ps2p_neighbor_index, \
      per_point_squared_ball_radius, sdf_precompute=sdf_precompute)
    mls_sdf_loss *= FLAGS.sdf_loss_weight
    mls_sdf_grad_loss *= FLAGS.sdf_grad_loss_weight
    
    if(FLAGS.sdf_loss_weight > DBL_EPSILON):
      loss_dict[name + "_mls_sdf_loss"] = mls_sdf_loss
    
    if(FLAGS.sdf_grad_loss_weight > DBL_EPSILON):
      loss_dict[name + "_mls_sdf_grad_loss"] = mls_sdf_grad_loss
    
    train_loss += mls_sdf_loss + mls_sdf_grad_loss
  
  wLop_repulsion_loss, local_plane_regularization, radius_smoothness_loss = \
    self_regularization_loss(predict_position_matrix, predict_normal_normalized, points_segment, points_num, per_point_squared_ball_radius)
  
  if(FLAGS.repulsion_weight > DBL_EPSILON):
    LOP_regularization_loss = FLAGS.repulsion_weight * wLop_repulsion_loss
    train_loss += LOP_regularization_loss
    loss_dict[name + '_repulsion'] = LOP_regularization_loss
    
  if(FLAGS.normal_norm_reg_weight > DBL_EPSILON):
    normal_unit_norm_loss = FLAGS.normal_norm_reg_weight*normal_unit_norm_regularization(predict_normal_matrix, points_segment, points_num)
    loss_dict[name +'_normal_unit_norm_reg'] = normal_unit_norm_loss
    train_loss += normal_unit_norm_loss
    
  if(FLAGS.geo_reg > DBL_EPSILON):
    local_plane_regularization *= FLAGS.geo_reg
    train_loss += local_plane_regularization
    loss_dict[name + "_local_plane_regularization"] = local_plane_regularization
    
  if(not FLAGS.constant_radius and FLAGS.patch_radius_smoothness > DBL_EPSILON):
    radius_smoothness_loss *= FLAGS.patch_radius_smoothness
    train_loss += radius_smoothness_loss
    loss_dict["radius_smoothness"] = radius_smoothness_loss
  
  if(FLAGS.weight_decay > DBL_EPSILON):
    network_weights_decay_loss = l2_regularizer("ocnn", FLAGS.weight_decay)
    loss_dict['network_weight_decay_loss'] = network_weights_decay_loss
    train_loss += network_weights_decay_loss
  
  loss_dict[name + "_total_loss"] = train_loss
  return train_loss, loss_dict

def train_data_loader():
  data_list = points_dataset_AE_multiple_GPU(FLAGS.train_data, FLAGS.train_batch_size, points_num=FLAGS.num_of_input_points, depth=FLAGS.depth, \
    gpu_num=FLAGS.num_of_gpus, sample_grid_points_number=FLAGS.sdf_samples_each_iter, data_sources=FLAGS.sdf_data_sources, noise_stddev=FLAGS.noise_stddev)
 
  assert(len(data_list) == FLAGS.num_of_gpus*2+5)
  train_record_num = data_list[0]
  octree_all_gpu = data_list[1:(FLAGS.num_of_gpus+1)]
  gt_octree_all_gpu = data_list[(FLAGS.num_of_gpus+1):(FLAGS.num_of_gpus*2+1)]  
  points, filenames, sampled_points, sampled_sdf = data_list[(FLAGS.num_of_gpus*2+1):]
  
  points_all_gpu = []
  sampled_points_all_gpu = []
  sampled_sdf_all_gpu = []
  for i in range(FLAGS.num_of_gpus):
    points_all_gpu.append(points[i*train_batch_size_gpu:(i+1)*train_batch_size_gpu])
    sampled_points_all_gpu.append(sampled_points[i*train_batch_size_gpu:(i+1)*train_batch_size_gpu])
    sampled_sdf_all_gpu.append(sampled_sdf[i*train_batch_size_gpu:(i+1)*train_batch_size_gpu])
  return train_record_num, octree_all_gpu, gt_octree_all_gpu, points_all_gpu, sampled_points_all_gpu, sampled_sdf_all_gpu, filenames

def train_network():  
  train_record_num, octree_all_gpu, gt_octree_all_gpu, points_all_gpu, sampled_points_all_gpu, sampled_sdf_all_gpu, filenames = train_data_loader()    
  loss_all_gpu = []
  gradients_all_gpu=[]
  solver = tf.train.AdamOptimizer(learning_rate=lr_placeholder)
  all_loss_dict = []
  
  for g_id in range(FLAGS.num_of_gpus):
    with tf.device('/gpu:{}'.format(g_id)):
      with tf.name_scope('GPU_{}'.format(g_id)) as scope:
        if(g_id == 0):
          g_loss, predict_points, sampled_position_matrix, loss_dict = build_graph_gpu_octree_local(train_batch_size_gpu, octree_all_gpu[g_id], None if points_all_gpu is None else points_all_gpu[g_id], 
            sampled_points_all_gpu[g_id], sampled_sdf_all_gpu[g_id], reuse=None, gt_octree= gt_octree_all_gpu[g_id])
        else:
          g_loss, _, _, loss_dict = build_graph_gpu_octree_local(train_batch_size_gpu, octree_all_gpu[g_id], None if points_all_gpu is None else points_all_gpu[g_id], 
            sampled_points_all_gpu[g_id], sampled_sdf_all_gpu[g_id], reuse=True, gt_octree= gt_octree_all_gpu[g_id])
        loss_all_gpu.append(g_loss)
        gradients_all_gpu.append(solver.compute_gradients(g_loss))
        all_loss_dict.append(loss_dict)
        #use last tower statistics to update the moving mean/variance 
        if(g_id == 0):
          batchnorm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
    
  with tf.device('/cpu:0'):
    train_summary = tf_summary_from_dict(all_loss_dict, True)
  
  with tf.device('/gpu:0'):
    total_loss = sum(loss_all_gpu) / FLAGS.num_of_gpus
    with tf.control_dependencies(batchnorm_update_ops):
      if(FLAGS.num_of_gpus == 1):
        apply_gradient_op = solver.apply_gradients(gradients_all_gpu[0])
      else:
        grad_avg = average_gradient(gradients_all_gpu)
        apply_gradient_op = solver.apply_gradients(grad_avg)
  
  points_all_gpu[0] = tf.transpose(tf.reshape(points_all_gpu[0],[-1, 6, FLAGS.num_of_input_points]), perm=[0,2,1])
    
  return train_record_num, train_summary, total_loss, apply_gradient_op, predict_points, sampled_position_matrix, points_all_gpu[0], filenames

def octree_mls_points_squared_radius():
  return 0.25**(FLAGS.decoder_octree_depth - 1) / FLAGS.points_per_node

def build_graph_gpu_octree_local(input_batch_size, octree, points, grid_position, grid_sdf, reuse, is_training=True, gt_octree=None):
  assert(gt_octree is not None)
  
  split_loss, split_acc, mls_points, points_segment, octree_node_xyz_valid = \
    octree_network_unet(octree, FLAGS.depth, FLAGS.decoder_octree_depth, FLAGS.channel, FLAGS.points_per_node, training=is_training, reuse=reuse, node_receptive_field=FLAGS.node_receptive_field, predict_radius=not FLAGS.constant_radius, radius_range=FLAGS.radius_range, gt_octree=gt_octree)
  
  points_num = tf.cast(tf.segment_sum(tf.ones_like(points_segment, dtype=tf.float64), points_segment), tf.int32)
  constant_squared_radius = octree_mls_points_squared_radius()
  
  if(not FLAGS.constant_radius):
    per_point_squared_radius = constant_squared_radius* tf.math.square(mls_points[:,6])
    mls_points = mls_points[:,:6]
  else:
    per_point_squared_radius = constant_squared_radius* tf.ones(tf.shape(mls_points)[0])
  
  sampled_position_matrix = tf.reshape(grid_position, [-1, FLAGS.sdf_samples_each_iter, 3])
  grid_sdf = tf.reshape(grid_sdf, [-1, FLAGS.sdf_samples_each_iter, 4])
  
  sampled_position_matrix = tf.stop_gradient(sampled_position_matrix)
  grid_sdf = tf.stop_gradient(grid_sdf)  
  
  final_loss, loss_dict = network_loss(mls_points, points_segment, points_num, per_point_squared_radius, sampled_position_matrix, grid_sdf, name="final_geometry")
  
  loss_dict['split_accuracy'] = tf.add_n(split_acc) / len(split_acc)
  loss_dict['octree_split_loss'] = FLAGS.octree_split_loss_weighting*tf.add_n(split_loss)
  train_loss = final_loss + FLAGS.octree_split_loss_weighting*tf.add_n(split_loss)
  
  predict_points_list = [mls_points, points_num, per_point_squared_radius]
  loss_dict['final_geometry_total_loss'] = train_loss
  
  return train_loss, predict_points_list, sampled_position_matrix, loss_dict

def build_graph_gpu_octree_local_inference(octree, reuse=None, is_training=False, test_batch_size=1):
  assert(not is_training)
  print("inference using prediction octree")
  mls_points, points_segment, octree_node_xyz_valid = octree_network_unet_completion_decode_shape(octree, FLAGS.depth, FLAGS.decoder_octree_depth, FLAGS.channel, FLAGS.points_per_node, test_batch_size, training=is_training, reuse=reuse, node_receptive_field=FLAGS.node_receptive_field, predict_radius=not FLAGS.constant_radius, radius_range=FLAGS.radius_range)
  
  points_num = tf.cast(tf.segment_sum(tf.ones_like(points_segment, dtype=tf.float64), points_segment), tf.int32)
  constant_squared_radius = octree_mls_points_squared_radius()
  
  if(not FLAGS.constant_radius):
    per_point_squared_radius = constant_squared_radius* tf.math.square(mls_points[:,6])
    mls_points = mls_points[:,:6]
  else:
    per_point_squared_radius = constant_squared_radius* tf.ones(tf.shape(mls_points)[0])
    
  predict_points_list = [mls_points, points_num, per_point_squared_radius]
  
  return predict_points_list

def write_visualization_results(predict_points, sampled_position_matrix, input_points, dir, iter, filenames=None):
  final_prediction = predict_points[0]
  assert(len(predict_points) == 3)
  per_point_radius = np.reshape(predict_points[2], [-1])
  per_point_radius = np.split(per_point_radius, np.cumsum(predict_points[1]))
  per_point_radius.pop()
  final_prediction_ = np.split(np.reshape(final_prediction, [-1]), 6*np.cumsum(predict_points[1]))
  final_prediction = [np.reshape(item, [-1,6]) for item in final_prediction_]
  final_prediction.pop()
  batch_size = len(final_prediction)
  assert(len(per_point_radius) == batch_size)
  
  input_points = np.reshape(input_points, [-1, FLAGS.num_of_input_points, 6])
  assert(batch_size == input_points.shape[0])
  
  sampled_position_matrix = np.reshape(sampled_position_matrix, [batch_size, -1, 3])
    
  if(filenames is not None):
    filename_text_file = open(os.path.join(dir, "filenames_{:06d}.txt".format(iter)), "w")
    for train_batch_idx in range(batch_size):
      words_arr = filenames[train_batch_idx].decode("utf8")#.split("\\")
      #filename_text_file.write("{}\t{}\n".format(words_arr[1], words_arr[3][:-4]))
      filename_text_file.write("{}\n".format(words_arr))
    filename_text_file.close()
  
  for i in range(batch_size):
    np.savetxt(os.path.join(dir, 'input_{:06d}_{:04d}.xyz'.format(iter, i)), input_points[i], fmt='%0.4f')
    np.savetxt(os.path.join(dir, 'predict_pc_{:06d}_{:04d}.xyz'.format(iter, i)), final_prediction[i], fmt='%0.4f')
    if(iter == 0):
      cur_sample_position_matrix = sampled_position_matrix[i]
      cur_sample_position_matrix = cur_sample_position_matrix[np.where(cur_sample_position_matrix[:,0] > -50)[0]]
      np.savetxt(os.path.join(dir, 'sampled_{:06d}_{:04d}.xyz'.format(iter, i)), cur_sample_position_matrix)
    
    if(per_point_radius is not None):
      np.savetxt(os.path.join(dir, 'predict_pc_{:06d}_{:04d}_radius.txt'.format(iter, i)),   per_point_radius[i])

def inference_from_inputs():
  #placeholder for input pointcloud, should in shape [-1, 3]
  input_points = tf.placeholder(tf.float32)
  #convert input_points to octree
  input_points_str = points_new(input_points, [], input_points, [])
  input_octree = octree_batch([points2octree(input_points_str, depth=FLAGS.depth, full_depth=2, node_dis=False, split_label=True)])
  mls_points = build_graph_gpu_octree_local_inference(input_octree)
  
  #finish build the graph and next we load checkpoint
  gvars = tf.global_variables()
  print("{} global variables".format(len(gvars)))
  print("{} model parameters total".format(get_num_params()))
  
  ckpt = tf.train.latest_checkpoint(FLAGS.ckpt)
  start_iters = 0 if not ckpt else int(ckpt[ckpt.find('iter') + 4:-5]) + 1
  if(ckpt):
    tf_restore_saver = tf.train.Saver(var_list=gvars, max_to_keep=100)
  else:
    print("Cannot load checkpoint from {}".format(FLAGS.ckpt))
    raise NotImplementedError
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    # initialize
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    print("===========================")
    print("restore training from iteration %d: %s" %(start_iters, ckpt))
    print("===========================")
    tf_restore_saver.restore(sess, ckpt)
        
    if(True):
      #forward single ShapeNet object
      import plyfile
      #inference
      input_file = "examples/d0fa70e45dee680fa45b742ddc5add59.ply"
      assert(os.path.exists(input_file))
      plydata = plyfile.PlyData.read(input_file)
      noisy_points = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']], axis=1)
      assert(noisy_points.shape[1] == 3)
      #first normalize noisy points to [-0.95, 0.95]
      scale = 0.95 / np.abs(noisy_points).max()
      noisy_points *= scale
      assert(noisy_points.min() >= -1 and noisy_points.max() <= 1)
      mls_points_prediction = sess.run(mls_points, feed_dict={input_points:noisy_points})
      assert(len(mls_points_prediction) == 3)
      
      mls_points_geometry = mls_points_prediction[0]
      mls_points_radius = mls_points_prediction[2]
      
      mls_points_position = mls_points_geometry[:,:3]
      mls_points_normal = mls_points_geometry[:,3:] / np.linalg.norm(mls_points_geometry[:,3:], axis=1, keepdims=True)
      mls_points_geometry = np.concatenate([mls_points_position, mls_points_normal], axis=1)
      
      shape_scale = np.ones(1)*scale
      np.savetxt(input_file+".xyz", mls_points_geometry, fmt="%0.6f")
      np.savetxt(input_file+"_radius.txt", mls_points_radius, fmt="%0.6f")
      np.savetxt(input_file+"_scale.txt", shape_scale)        
      return
    
    #Shape Completion of ShapeNet:on ShapeNet 13 classes
    import plyfile
    if not os.path.exists(FLAGS.exp_folder):
      os.mkdir(FLAGS.exp_folder)
    test_output_dir = os.path.join(FLAGS.exp_folder, 'test')
    if not os.path.exists(test_output_dir): os.makedirs(test_output_dir, exist_ok=True)
    
    #where input pointcloud lies
    input_dir = "your_directory/convolutional_occupancy_networks-master/out/pointcloud/shapenet_3plane/generation_pretrained/input"    
    categories = os.listdir(input_dir)
    
    for cat in categories:
      cat_input_dir = os.path.join(input_dir, cat)
      filelist = os.listdir(cat_input_dir)
      cat_output_dir = os.path.join(test_output_dir, cat)
      if(not os.path.exists(cat_output_dir)): os.mkdir(cat_output_dir)
      for point_file in filelist:
        if(not os.path.isdir(point_file) and point_file.endswith(".ply")):
          sample_id = point_file.replace(".ply", "")
          print("{}:{}".format(cat, point_file))
          #do inference
          plydata = plyfile.PlyData.read(os.path.join(cat_input_dir, point_file))
          noisy_points = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']], axis=1)
          assert(noisy_points.shape[1] == 3)
          #first normalize noisy points to [-0.95, 0.95]
          scale = 0.95 / np.abs(noisy_points).max()
          noisy_points *= scale
          assert(noisy_points.min() >= -1 and noisy_points.max() <= 1)
          mls_points_prediction = sess.run(mls_points, feed_dict={input_points:noisy_points})
          assert(len(mls_points_prediction) == 3)
          
          mls_points_geometry = mls_points_prediction[0]
          mls_points_radius = mls_points_prediction[2]
          
          mls_points_position = mls_points_geometry[:,:3]
          mls_points_normal = mls_points_geometry[:,3:] / np.linalg.norm(mls_points_geometry[:,3:], axis=1, keepdims=True)
          mls_points_geometry = np.concatenate([mls_points_position, mls_points_normal], axis=1)
          
          shape_scale = np.ones(1)*scale
          np.savetxt(os.path.join(cat_output_dir, sample_id+".xyz"), mls_points_geometry, fmt="%0.6f")
          np.savetxt(os.path.join(cat_output_dir, sample_id+"_radius.txt"), mls_points_radius, fmt="%0.6f")
          np.savetxt(os.path.join(cat_output_dir, sample_id+"_scale.txt"), shape_scale)            

def training_pipeline():
  if not os.path.exists(FLAGS.exp_folder):
    os.mkdir(FLAGS.exp_folder)
  train_record_num, train_summary, total_loss, train_op, predict_points, sampled_position_matrix, input_points, train_filenames = train_network()
  
  # checkpoint
  ckpt = tf.train.latest_checkpoint(FLAGS.ckpt)
  start_iters = 0 if not ckpt else int(ckpt[ckpt.find('iter') + 4:-5]) + 1
  
  # saver
  gvars = tf.global_variables()
  print("{} global variables".format(len(gvars)))
  print("{} model parameters total".format(get_num_params()))
  
  save_vars = gvars
  save_vars_wo_adam  = [var for var in gvars if 'Adam' not in var.name]
  tf_saver_model = tf.train.Saver(var_list = save_vars_wo_adam, max_to_keep=100)
  restore_vars = save_vars_wo_adam
  
  print("restore {} vars".format(len(restore_vars)))
  print("save {} vars".format(len(save_vars)))
  
  tf_saver = tf.train.Saver(var_list=save_vars, max_to_keep=100)
  if ckpt:
    tf_restore_saver = tf.train.Saver(var_list=restore_vars, max_to_keep=100)
    
  #train_record_num = sum(1 for _ in tf.python_io.tf_record_iterator(FLAGS.train_data))
  print("train_record_num: {}".format(train_record_num))
  iter_100_epoch = int(100.0 * train_record_num / FLAGS.train_batch_size)
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    # initialize
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    
    if ckpt:
      print("===========================")
      print("restore training from iteration %d: %s" %(start_iters, ckpt))
      print("===========================")
      tf_restore_saver.restore(sess, ckpt)
    
    #folders for training outputs
    obj_dir = os.path.join(FLAGS.exp_folder, 'obj')
    if not os.path.exists(obj_dir): os.makedirs(obj_dir)
    
    #folders for checkpoint
    if not os.path.exists(os.path.join(FLAGS.exp_folder, "model")): os.makedirs(os.path.join(FLAGS.exp_folder, "model"), exist_ok=True)
    
    # tf summary
    summary_writer = tf.summary.FileWriter(FLAGS.exp_folder, sess.graph)
    lr_decay_step = int(FLAGS.lr_decay_epochs*iter_100_epoch/100.0)
    print("lr decay 20% per {} iterations({} epochs)".format(lr_decay_step, FLAGS.lr_decay_epochs))
    cur_lr = max(FLAGS.learning_rate_lower_bound, FLAGS.learning_rate*0.8**int(start_iters / lr_decay_step))
    print("initial lr setting to {}".format(cur_lr))
    
    max_iter = int(1.0 * FLAGS.max_training_epochs * train_record_num / FLAGS.train_batch_size) + 1
    print("max training iterations: {}".format(max_iter))
    last_loss = 0
    # start training
    for i in tqdm.tqdm(range(start_iters, max_iter)):
      #cur_epochs = 1.0 * i * FLAGS.train_batch_size / train_record_num
      if(i % lr_decay_step == 0):
        cur_lr = max(FLAGS.learning_rate_lower_bound, FLAGS.learning_rate*0.8**int(i / lr_decay_step))
        print("lr setting to {}".format(cur_lr))
      
      if(i % 5000 == 0):
        #write out visualization results
        _, train_summary_fetch, train_loss_eval, predict_points_eval, sampled_position_matrix_eval, input_points_fetch, train_filenames_fetch = \
          sess.run([train_op, train_summary, total_loss, predict_points, sampled_position_matrix, input_points, train_filenames], feed_dict={lr_placeholder: cur_lr})
        write_visualization_results(predict_points_eval, sampled_position_matrix_eval, input_points_fetch, obj_dir, i, filenames=train_filenames_fetch)
      else:
        _, train_summary_fetch, train_loss_eval = sess.run([train_op, train_summary, total_loss], feed_dict={lr_placeholder: cur_lr})
      
      #write summary to tfevents
      summary_writer.add_summary(train_summary_fetch, i)
      #save checkpoint
      if (i % 5000 == 0):
        tf_saver.save(sess, os.path.join(FLAGS.exp_folder, 'model/iter{:06d}.ckpt'.format(i)))
      #print logs to stdout
      if(i % 500 == 0):
        print("iteration {}: train loss = {}".format(i, train_loss_eval))
        sys.stdout.flush()
          
  print("Program terminated.")

def main():
  if(FLAGS.test):
    return inference_from_inputs()
  return training_pipeline()

if __name__ == '__main__':
  print('Program running...')
  main()