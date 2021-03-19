import tensorflow as tf
import sys
sys.path.append("Octree")
from ocnn import *

def preprocess_octree_data_from_unoriented_points(data, octree, depth):
  #subtract octree node mean position
  #print("=======================\ninput octree feature with shape:{}\n==========================".format(data.shape))
  data = tf.reshape(data, [3, -1])
  octree_node_xyz = tf.cast(octree_xyz(octree, depth), tf.int32) #we get nx4
  octree_node_center = tf.cast(octree_node_xyz[:,:3], dtype=tf.float32)*(0.5**(depth-1)) - 1 + (0.5**depth)
  #normalize to [-1, 1]
  data = (data - tf.transpose(octree_node_center))*(2**depth)
  
  #concat node occupancy indicator
  data = tf.concat([data, tf.ones([1, tf.shape(data)[1]], dtype=data.dtype)], axis=0)
  print("=======================\ninput octree feature with shape:{}\n==========================".format(data.shape))
  
  #mask empty nodes to zero
  input_node_mask = tf.reshape(tf.greater(octree_property(octree, property_name="split", dtype=tf.float32, depth=depth, channel=1), 0), [1, -1])
  input_node_mask = tf.cast(tf.tile(input_node_mask, multiples = [4,1]), data.dtype)
  data = tf.stop_gradient(data*input_node_mask)
  
  return tf.reshape(data, [1, 4, -1, 1])

def octree_network_unet(octree, input_octree_depth, output_octree_depth, octree_feature_channel, points_per_node, training, reuse=False, node_receptive_field=1.0, predict_radius=False, radius_range=2.0, gt_octree=None):
  #if gt_octree is None, this is ShapeAE Architecture else Shape Completion
  channels = [4, 64, 128, 128, 128, 64, 32, 16, 8]
  resblock_num = 3
  depth = input_octree_depth
  with tf.variable_scope('ocnn_encoder', reuse=reuse):
    with tf.variable_scope('signal_gt'):
      data = octree_property(octree, property_name="feature", dtype=tf.float32,
                            depth=depth, channel=octree_feature_channel)
      if(octree_feature_channel == 3):
        #which means input signal does not include normal
        data = preprocess_octree_data_from_unoriented_points(data, octree, depth)
      else:
        data = tf.reshape(data, [1, octree_feature_channel, -1, 1])
    
    with tf.variable_scope("front"):
      data = octree_conv_bn_relu(data, octree, depth, channels[depth], training)
        
    convd = [None]*10
    for d in range(depth, 1, -1):
      for i in range(0, resblock_num):
        with tf.variable_scope('resblock_%d_%d' % (d, i)):
          data = octree_resblock(data, octree, d, channels[d], 1, training)
      convd[d] = data #for skip connections
      if(d != 2):
        with tf.variable_scope('down_%d' % d):
          data = octree_conv_bn_relu(data, octree, d, channels[d-1], training,
                                    stride=2, kernel_size=[2])
    
    code = data
  
  #decoder
  depth = output_octree_depth
  with tf.variable_scope('ocnn_decoder', reuse=reuse):    
    data = code
    loss, accu = [], []
    for d in range(2, depth + 1):
      for i in range(0, resblock_num):
        with tf.variable_scope('resblock_%d_%d' % (d, i)):
          data = octree_resblock(data, octree if gt_octree is None else gt_octree, d, channels[d], 1, training)

      with tf.variable_scope('predict_%d' % d):
        logit, label = predict_label(data, 2, 32, training)
        logit = tf.transpose(tf.squeeze(logit, [0,3])) # (1, C, H, 1) -> (H, C)        

      with tf.variable_scope('octree_loss_%d' % d):
        with tf.variable_scope('label_gt'):
          label_gt = octree_property(octree if gt_octree is None else gt_octree, property_name="split", 
              dtype=tf.float32, depth=d, channel=1)
          label_gt = tf.reshape(tf.cast(label_gt, dtype=tf.int32), [-1])
        loss.append(softmax_loss(logit, label_gt, num_class=2))
        accu.append(label_accuracy(label, label_gt))
      
      signals_per_point = 6
      if(predict_radius):
        signals_per_point += 1
      
      if d == depth:
        with tf.variable_scope('regress_%d' % d):
          mls_points_local = predict_signal(data, points_per_node*signals_per_point, 128, training) #axis-angle
          #from signal to mls points, merge points
          mls_points_local = tf.reshape(tf.transpose(tf.squeeze(mls_points_local, [0,3])), [-1, signals_per_point]) # (1, C, H, 1) -> (H, C)
          position = tf.nn.tanh(mls_points_local[:,:3])*(0.5**depth)*node_receptive_field
          if(predict_radius):
            normal = mls_points_local[:,3:6]
            radius = tf.expand_dims(tf.math.pow(radius_range, tf.nn.tanh(mls_points_local[:,6])), axis=-1)
            normal = tf.concat([normal, radius], axis=-1)
          else:
            normal = mls_points_local[:,3:]
          
          octree_node_xyz = tf.cast(octree_xyz(octree if gt_octree is None else gt_octree, depth), tf.int32) #we get nx4
          octree_node_center = tf.cast(octree_node_xyz[:,:3], dtype=position.dtype)*(0.5**(depth-1)) - 1 + (0.5**depth)
          position += tf.reshape(tf.tile(tf.expand_dims(octree_node_center, axis=1), multiples = [1, points_per_node, 1]), [-1,3])
          
          mls_points = tf.concat([position, normal], axis=-1)
          #points_nums = tf.segment_sum(tf.ones_like(point_segment, dtype=tf.int32), point_segment) #tf.cumsum()
          
          #mask empty octree node
          if(True):
            #using groundtruth supervision in last depth
            node_mask = tf.greater(label_gt, 0)
          else:
            print("with real split in unet decoder output")
            #using the real split supervision
            node_mask = tf.greater(label, 0)
          octree_node_xyz_valid = tf.boolean_mask(octree_node_xyz, node_mask)
          point_segment =  octree_node_xyz_valid[:,3]
          
          #point_segment = tf.boolean_mask(octree_node_xyz[:,3], node_mask)
          point_segment = tf.cast(tf.reshape(tf.tile(tf.expand_dims(point_segment, axis=1), multiples=[1, points_per_node]), [-1]), tf.int32)
          points_mask = tf.reshape(tf.tile(tf.expand_dims(node_mask, axis=-1), multiples=[1, points_per_node]), [-1])
          mls_points = tf.boolean_mask(mls_points, points_mask)

      if d < depth:
        with tf.variable_scope('up_%d' % d):
          data = octree_deconv_bn_relu(data, octree if gt_octree is None else gt_octree, d, channels[d-1], training,
                                      stride=2, kernel_size=[2])
          #skip connections
          if(gt_octree is None):
            data = tf.concat([data, convd[d+1]], axis=1)
          else:
            skip, _ = octree_align(convd[d+1], octree, gt_octree, d+1)
            data = tf.concat([data, skip], axis=1)
  if(gt_octree is None):
    return mls_points, point_segment, octree_node_xyz_valid
  else:
    return loss, accu, mls_points, point_segment, octree_node_xyz_valid

def octree_network_unet_completion_decode_shape(octree, input_octree_depth, output_octree_depth, octree_feature_channel, points_per_node, shape_batch_size, training, reuse=False, node_receptive_field=1.0, predict_radius=False, radius_range=2.0):
  assert(not training)
  #Shape Completion decode shape at test phase
  channels = [4, 64, 128, 128, 128, 64, 32, 16, 8]
  resblock_num = 3
  depth = input_octree_depth
  with tf.variable_scope('ocnn_encoder', reuse=reuse):
    with tf.variable_scope('signal_gt'):
      data = octree_property(octree, property_name="feature", dtype=tf.float32,
                            depth=depth, channel=octree_feature_channel)
      if(octree_feature_channel == 3):
        #which means input signal does not include normal
        data = preprocess_octree_data_from_unoriented_points(data, octree, depth)
      else:
        data = tf.reshape(data, [1, octree_feature_channel, -1, 1])
    
    with tf.variable_scope("front"):
      data = octree_conv_bn_relu(data, octree, depth, channels[depth], training)
        
    convd = [None]*10
    input_split_label = [None]*10
    
    for d in range(depth, 1, -1):
      input_split_label[d] = octree_property(octree, property_name="split", dtype=tf.float32, depth=d, channel=1)
      print("input split label with shape in depth {}: {}".format(d, input_split_label[d].shape))
      for i in range(0, resblock_num):
        with tf.variable_scope('resblock_%d_%d' % (d, i)):
          data = octree_resblock(data, octree, d, channels[d], 1, training)
      convd[d] = data #for skip connections
      if(d != 2):
        with tf.variable_scope('down_%d' % d):
          data = octree_conv_bn_relu(data, octree, d, channels[d-1], training,
                                    stride=2, kernel_size=[2])
    
    code = data
  
  #decoder
  depth = output_octree_depth
  with tf.variable_scope('ocnn_decoder', reuse=reuse):
    # init the octree
    with tf.variable_scope('octree_0'):
      #dis = False if flags.channel < 4 else True
      octree_prediction = octree_new(batch_size=shape_batch_size, channel=4, has_displace=False)
    with tf.variable_scope('octree_1'):
      octree_prediction = octree_grow(octree_prediction, target_depth=1, full_octree=True)
    with tf.variable_scope('octree_2'):
      octree_prediction = octree_grow(octree_prediction, target_depth=2, full_octree=True)
    
    data = code
    for d in range(2, depth + 1):
      for i in range(0, resblock_num):
        with tf.variable_scope('resblock_%d_%d' % (d, i)):
          data = octree_resblock(data, octree_prediction, d, channels[d], 1, training)

      with tf.variable_scope('predict_%d' % d):
        logit, label = predict_label(data, 2, 32, training)
        logit = tf.transpose(tf.squeeze(logit, [0,3])) # (1, C, H, 1) -> (H, C)        
      
      #utilize input octree info
      #skip_label, _ = octree_align(tf.reshape(input_split_label[d], [1,1,-1,1]), octree, octree_prediction, d)
      #skip_label = tf.reshape(skip_label, [-1])
      #print("label shape {}".format(label.shape))
      #label = tf.cast(tf.greater(tf.cast(label, tf.float32)+skip_label, 0), tf.int32)
            
      with tf.variable_scope('octree_%d' % d):
          octree_prediction = octree_update(octree_prediction, label, depth=d, mask=1)
            
      if d == depth:
        signals_per_point = 6
        if(predict_radius):
          signals_per_point += 1
        with tf.variable_scope('regress_%d' % d):
          mls_points_local = predict_signal(data, points_per_node*signals_per_point, 128, training) #axis-angle
          #from signal to mls points, merge points
          mls_points_local = tf.reshape(tf.transpose(tf.squeeze(mls_points_local, [0,3])), [-1, signals_per_point]) # (1, C, H, 1) -> (H, C)
          position = tf.nn.tanh(mls_points_local[:,:3])*(0.5**depth)*node_receptive_field
          if(predict_radius):
            normal = mls_points_local[:,3:6]
            radius = tf.expand_dims(tf.math.pow(radius_range, tf.nn.tanh(mls_points_local[:,6])), axis=-1)
            normal = tf.concat([normal, radius], axis=-1)
          else:
            normal = mls_points_local[:,3:]
          
          octree_node_xyz = tf.cast(octree_xyz(octree_prediction, depth), tf.int32) #we get nx4
          octree_node_center = tf.cast(octree_node_xyz[:,:3], dtype=position.dtype)*(0.5**(depth-1)) - 1 + (0.5**depth)
          position += tf.reshape(tf.tile(tf.expand_dims(octree_node_center, axis=1), multiples = [1, points_per_node, 1]), [-1,3])
          
          mls_points = tf.concat([position, normal], axis=-1)
          #points_nums = tf.segment_sum(tf.ones_like(point_segment, dtype=tf.int32), point_segment) #tf.cumsum()
          
          #mask empty octree node
          node_mask = tf.greater(label, 0)
          octree_node_xyz_valid = tf.boolean_mask(octree_node_xyz, node_mask)
          point_segment =  octree_node_xyz_valid[:,3]
          
          #point_segment = tf.boolean_mask(octree_node_xyz[:,3], node_mask)
          point_segment = tf.cast(tf.reshape(tf.tile(tf.expand_dims(point_segment, axis=1), multiples=[1, points_per_node]), [-1]), tf.int32)
          points_mask = tf.reshape(tf.tile(tf.expand_dims(node_mask, axis=-1), multiples=[1, points_per_node]), [-1])
          mls_points = tf.boolean_mask(mls_points, points_mask)

      if d < depth:
        with tf.variable_scope('octree_%d' % (d+1)):
          octree_prediction = octree_grow(octree_prediction, target_depth=d+1, full_octree=False)
        with tf.variable_scope('up_%d' % d):
          data = octree_deconv_bn_relu(data, octree_prediction, d, channels[d-1], training,
                                      stride=2, kernel_size=[2])
          #skip connections
          skip, _ = octree_align(convd[d+1], octree, octree_prediction, d+1)
          data = tf.concat([data, skip], axis=1)
  
  return mls_points, point_segment, octree_node_xyz_valid
