import operator
import functools
import tensorflow as tf
import json

def config_reader(config_path):
	json_data=open(config_path).read()
	config = json.loads(json_data)
	return config

def get_num_params():
  num_params = 0
  for variable in tf.trainable_variables():
    shape = variable.get_shape()
    num_params += functools.reduce(operator.mul, [dim.value for dim in shape], 1)
  return num_params
  
def average_gradient(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, var in grad_and_vars:
      if g is not None:
        expanded_g = tf.expand_dims(g, 0)
        grads.append(expanded_g)
    if(len(grads) == 0):
      continue
    grad = tf.concat(grads, axis=0)
    grad = tf.reduce_mean(grad, axis=0)

    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def tf_summary_from_dict(loss_dict, is_training):
  if(is_training):
    summary_name = "train_summary"
  else:
    summary_name = "test_summary"
  #create summary
  with tf.name_scope(summary_name):
    summary_list = []
    gpu0_loss_dict = loss_dict[0]
    for item in gpu0_loss_dict:
      scalar_acc = 0
      for i in range(len(loss_dict)):
        scalar_acc += loss_dict[i][item]
      scalar_acc /= len(loss_dict)
      summary_item = tf.summary.scalar(item, scalar_acc)
      summary_list.append(summary_item)
  return tf.summary.merge(summary_list)

def rowwise_l2_norm_squared(feature):
  #assum input with size[n,f] out shape = [n]
  return tf.reduce_sum(tf.math.square(feature), axis=-1)