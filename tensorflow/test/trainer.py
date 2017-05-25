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


from config_helper import default_name
from config_helper import logger
import logging

from ds2_dataset import Dataset as dataset

# for parse_args
import conf as CONF
import argparse
import os
ARGS = None

#


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
  parser.add_argument(
    '--debug',
    type=bool,
    default=CONF.DEBUG,
    help='If true will in debug mode and logging debug')
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


def DEBUG(op):
  if ARGS.debug:
    op
  else:
   pass

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

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
    
@default_name("conv")
def conv_layer(input_tensor,
                input_channels,
                num_kernels,
                kernel_size,
                stride_size,
                data_format="NHWC",
                padding='VALID',
                with_bias=True,
                name=None):
            
  """conv_layer returns a 2d convolution layer.

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

  logger.debug(name)
  with tf.name_scope(name):
    wgt = weight_variable('param_weight', [kh, kw, ic, oc])
    DEBUG(variable_summaries(wgt))
    DEBUG(tf.summary.histogram('pre_' + name, input_tensor))
    conv = tf.nn.conv2d(input_tensor, wgt, stride, padding,
                        data_format=data_format, name=name)
    DEBUG(tf.summary.histogram('post_' + name, conv))
    if with_bias:
      bias = bias_variable('param_bias', [oc])
      DEBUG(variable_summaries(bias))
      conv = tf.nn.bias_add(conv, bias, data_format, 'add_bias')
      DEBUG(tf.summary.histogram('post_bias', conv))
    return conv

@default_name("fc")
def fc_layer(input_tensor, dim_out, dim_in=None, with_bias=True, name=None):
  '''fc_layer returns a full connected layer
     Wx + b
     param:
      if dim_in is None, will set as dim_out
  '''
  logger.debug(name)
  if dim_in is None:
    dim_in = dim_out

  with tf.name_scope(name):
    DEBUG(tf.summary.histogram('pre_' + name, input_tensor))
    x = tf.reshape(input_tensor, [-1, dim_in])
    wgt = weight_variable('weight', [dim_in, dim_out])
    DEBUG(variable_summaries(wgt))
    fc_ = tf.matmul(x, wgt)
    DEBUG(tf.summary.histogram('post_' + name, fc_))
    if with_bias:
      bias = bias_variable('bias', [dim_out])
      DEBUG(variable_summaries(bias))
      fc_ = tf.add(fc_, bias, 'add_bias')
      DEBUG(tf.summary.histogram('post_bias' + name, fc_))
    return fc_


      
def train():
  img_h = 28
  img_w = 28
  num_classes = 10
  data_format = CONF.DATA_FORMAT
  assert data_format in ["NCHW", "NHWC"]
  
  # data
  with tf.name_scope('input'):
    input = tf.placeholder(tf.float32, [None, img_h*img_w], name='data')

    # Define loss and optimizer
    label = tf.placeholder(tf.float32, [None, num_classes], name='label')

  ic = 1
  oc = 16
  kh = 3
  kw = 3
  sh = 1
  sw = 1

  if data_format == "NHWC":
    img = tf.reshape(input, [-1, img_h, img_w, ic])
  else:
    img = tf.reshape(input, [-1, ic, img_h, img_w])

  conv = conv_layer(img, ic, oc, [kh, kw], [sh, sw], data_format=data_format)

  conv = conv_layer(conv, oc, oc, [kh, kw], [sh, sw], data_format=data_format)

  out = fc_layer(conv, num_classes, dim_in=24*24*oc)

  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=out)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  DEBUG(tf.summary.scalar('cross_entropy', cross_entropy))

  #optimizer = tf.train.AdamOptimizer(ARGS.learning_rate)
  with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(ARGS.learning_rate).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(label, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  DEBUG(tf.summary.scalar('accuracy', accuracy))

 
  
  # Import data
  mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

  # Create a saver for writing training checkpoints.
  #saver = tf.train.Saver()

  def feed_dict(is_train = True):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if is_train:
      xs, ys = mnist.train.next_batch(ARGS.batch_size) #, fake_data=ARGS.fake_data)
    else:
      xs, ys = mnist.test.images, mnist.test.labels
    return {input: xs, label: ys}

  with tf.Session() as sess:
    # Merge all the summaries and write them out to ARGS.log_dir
    if ARGS.debug:
      summary_op = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(ARGS.log_dir + '/train', sess.graph)
      test_writer = tf.summary.FileWriter(ARGS.log_dir + '/test') #, sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(ARGS.max_iter):
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      sess.run(train_op, feed_dict=feed_dict(),
               options=run_options, run_metadata=run_metadata)
      if i % CONF.LOSS_ITER == 0:
        train_acc = sess.run(accuracy, feed_dict=feed_dict())
        logger.info('iter %d, training accuracy %g' % (i, train_acc))
        #summary, train_acc = sess.run([summary_op, train_op],
        #                      feed_dict={input: batch[0], label: batch[1]},
        #                      options=run_options,
        #                      run_metadata=run_metadata)
        if ARGS.debug:
          summary = sess.run(summary_op, feed_dict=feed_dict(),
                             options=run_options, run_metadata=run_metadata)
        DEBUG(train_writer.add_run_metadata(run_metadata, 'iter%03d' % i))
        DEBUG(train_writer.add_summary(summary, i))

      if i % CONF.TEST_INTERVAL == 0:
        test_acc = sess.run(accuracy, feed_dict=feed_dict(False))
        logger.info('iter %d, test accuracy %g' % (i, test_acc))
        if ARGS.debug:
          summary = sess.run(summary_op, feed_dict=feed_dict(False))
        DEBUG(test_writer.add_summary(summary, i))

    DEBUG(train_writer.close())
    DEBUG(test_writer.close())


def main(_):
  logger.setLevel(logging.INFO)
  if ARGS.debug:
    logger.setLevel(logging.DEBUG)
  logger.debug(ARGS)

  if tf.gfile.Exists(ARGS.log_dir):
    tf.gfile.DeleteRecursively(ARGS.log_dir)
  tf.gfile.MakeDirs(ARGS.log_dir)

  train()

if __name__ == '__main__':
  ARGS, unparsed = parse_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
