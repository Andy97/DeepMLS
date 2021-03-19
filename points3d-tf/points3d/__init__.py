import tensorflow as tf
import os
from tensorflow.python.framework import ops

_current_path = os.path.dirname(os.path.realpath(__file__))
_tf_points_module = tf.load_op_library(os.path.join(_current_path, 'libpoints3d.so'))

get_neighbor_spatial_grid_local_self = _tf_points_module.get_neighbor_spatial_grid_local_self
get_neighbor_spatial_grid_radius_v_voting = _tf_points_module.get_neighbor_spatial_grid_radius_v_voting

ops.NotDifferentiable("GetNeighborSpatialGridRadiusVVoting")
ops.NotDifferentiable("GetNeighborSpatialGridLocalSelf")