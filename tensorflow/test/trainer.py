# Copyright 2017 The Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
  A Trainer for DeepSpeech 2
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# for variable and default names
from config_helper import default_name

# for parse_args
import conf as CONF
import argparse
import os
ARGS = None



def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--batch_size',
    required=False,
    type=int,
    default=CONF.BATCH_SIZE,
    help='batch size for train and test.')
  parser.add_argument(
    '-d',
    '--use_dummy',
    required=False,
    type=bool,
    default=CONF.USE_DUMMY,
    help='If true, uses dummy(fake) data for unit testing.')
  parser.add_argument(
    '-m',
    '--max_iter',
    required=False,
    type=int,
    default=CONF.MAX_ITER,
    help='Number of iterations to run trainer.')
  parser.add_argument(
    '--learning_rate',
    type=float,
    default=CONF.LEARNING_RATE,
    help='Initial learning rate')

#  parser.add_argument(
#    '--dropout',
#    type=float,
#    default=0.9,
#    help='Keep probability for training dropout.')

  LOG_DIR = os.path.abspath(CONF.LOG_DIR)
  parser.add_argument(
    '--log_dir',
    type=str,
    default=LOG_DIR,
    help='Summaries log directory')

  args, unparsed = parser.parse_known_args()
  # TODO: handle unparsed

  return args, unparsed

def _variable_on_cpu(name, shape, initializer = None, use_fp16 = False):
  """Helper to create a Variable stored on cpu memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, dtype = dtype, initializer = initializer)
  return var

def weight_variable(name, shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(name, shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

@default_name("conv")
def op_conv(input,
                input_channels,
                num_kernels,
                kernel_size,
                stride_size,
                data_format="NHWC",
                padding='VALID',
                with_bias=True,
                name=None):
            
  """op_conv returns a 2d convolution layer.

  param:
    input_channels, num_kernels and kernel_size are required
    size of kernel and stride are list as: [height, width]
    data_format: "NHWC" or "NCHW"
    padding refer to tf.nn.conv2d padding
  """

  assert input_channels is not None and num_kernels is not None
  assert isinstance(kernel_size, list) and len(kernel_size) == 2
  assert isinstance(stride_size, list) and len(stride_size) == 2
  assert isinstance(name, str)
  assert data_format in ["NHWC", "NCHW"]

  ic = input_channels
  # outpout channel
  oc = num_kernels
  kh, kw = kernel_size
  sh, sw = stride_size
  if data_format == "NHWC":
    stride = [1, sh, sw, 1]
  else:
    stride = [1, 1, sh, sw]
  print(name)
  with tf.name_scope(name):
    wgt = weight_variable('param_weight', [kh, kw, ic, oc])
    conv = tf.nn.conv2d(input, wgt, stride, padding,
                        data_format=data_format, name=name)
    if with_bias:
      bias = bias_variable('param_bias', [oc])
      conv = tf.nn.bias_add(conv, bias, data_format, 'add_bias')
    return conv

@default_name("fc")
def op_fc(input, dim_out, dim_in=None, with_bias=True, name=None):
  '''op_fc returns a full connected layer
     Wx + b
     param:
      if dim_in is None, will set as dim_out
  '''
  print(name)
  if dim_in is None:
    dim_in = dim_out

  with tf.name_scope(name):
    x = tf.reshape(input, [-1, dim_in])
    wgt = weight_variable('weight', [dim_in, dim_out])
    fc_ = tf.matmul(x, wgt)
    if with_bias:
      bias = bias_variable('bias', [dim_out])
      fc_ = tf.add(fc_, bias, 'add_bias')
    return fc_


      
def train():
  img_h = 28
  img_w = 28
  num_classes = 10
  data_format = CONF.DATA_FORMAT
  assert data_format in ["NCHW", "NHWC"]
  
  # data
  input = tf.placeholder(tf.float32, [None, img_h*img_w])

  # Define loss and optimizer
  label = tf.placeholder(tf.float32, [None, num_classes])

  ic = 1
  oc = 32
  kh = 5
  kw = 5
  sh = 1
  sw = 1

  if data_format == "NHWC":
    img = tf.reshape(input, [-1, img_h, img_w, ic])
  else:
    img = tf.reshape(input, [-1, ic, img_h, img_w])

  conv = op_conv(img, ic, oc, [kh, kw], [sh, sw], data_format=data_format)

  conv = op_conv(conv, oc, oc, [kh, kw], [sh, sw], data_format=data_format)

  out = op_fc(conv, num_classes, dim_in=20*20*oc)
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=out))

  #optimizer = tf.train.AdamOptimizer(ARGS.learning_rate)
  train_step = tf.train.AdamOptimizer(ARGS.learning_rate).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(label, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
  # Import data
  mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(ARGS.max_iter):
      batch = mnist.train.next_batch(ARGS.batch_size)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={input: batch[0], label: batch[1]})
        print('step %d, training accuracy %g' % (i, train_accuracy))

      train_step.run(feed_dict={input: batch[0], label: batch[1]})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        input: mnist.test.images, label: mnist.test.labels}))


def main(_):
  if tf.gfile.Exists(ARGS.log_dir):
    tf.gfile.DeleteRecursively(ARGS.log_dir)
  tf.gfile.MakeDirs(ARGS.log_dir)
  train()

if __name__ == '__main__':
  ARGS, unparsed = parse_args()
  print(ARGS)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
