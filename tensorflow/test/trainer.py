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

import tensorflow as tf

from config_helper import default_name
from config_helper import logger
import logging

from ds2_dataset import Dataset

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


def _nchw(input_data):
  '''use nchw data format
    return seq, bs, dim
  '''
  # transpose to [bs, 161, seq]
  trans = tf.transpose(input_data, perm = [0, 2, 1])
  # to shape: [bs, 1, 161, seq]
  feat = tf.expand_dims(trans, 1)

  # ic = 1, oc = 32, kh = 5, kw = 20, sh = 2, sw = 2
  feat = conv_layer(feat, 1, 32, [5, 20], [2, 2], data_format='NCHW')

  # ic = 32, oc = 32, kh = 5, kw = 10, sh = 1, sw = 2
  feat = conv_layer(feat, 32, 32, [5, 10], [1, 2], data_format='NCHW')

  feat_shape = tf.shape(feat)
  
  dim = feat_shape[1] * feat_shape[2]
  seq = feat_shape[3]
  feat = tf.reshape(feat, [-1, dim, seq])
  # [bs, dim , seq] to [seq, bs, dim]
  out = tf.transpose(feat, [2, 0, 1])
  return out
  
def _nhwc(input_data):
  '''use nhwc data format
    return seq, bs, dim
  '''
  # to shape: [bs, 1, 161, seq]
  feat = tf.expand_dims(input_data, -1)

  # ic = 1, oc = 32, kh = 20, kw = 5, sh = 2, sw = 2
  feat = conv_layer(feat, 1, 32, [20, 5], [2, 2], data_format='NHWC')

  # ic = 32, oc = 32, kh = 10, kw = 5, sh = 2, sw = 1
  feat = conv_layer(feat, 32, 32, [10, 5], [2, 1], data_format='NHWC')

  # [bs, seq, 75, 32] to [seq, bs, 32, 75]
  feat = tf.transpose(feat, [1, 0, 3, 2])
  feat_shape = tf.shape(feat)

  seq = feat_shape[0]
  dim = feat_shape[2] * feat_shape[3]
  out = tf.reshape(feat, [seq, -1, dim])
  return out
  
def get_logits(input_data, num_classes, data_format):
  if data_format not in ["NCHW", "NHWC"]:
    logger.fatal("only support nchw or nhwc yet")

  if data_format == 'NCHW':
    feat = _nchw(input_data)
  else:
    feat = _nhwc(input_data)

  rnn_out = feat

  rnn_shape = tf.shape(rnn_out)
  #rnn_shape = rnn_out.get_shape().as_list()

  seq = rnn_shape[0]
  
  logits = fc_layer(rnn_out, num_classes, dim_in=32 * 75)
  logits = tf.reshape(logits, [seq, -1, num_classes])
  return logits


def get_seq_lens(input_utt_lens):
  '''
  according to len = floor((size - filter_len) / stride) + 1 
  '''
  width = [20, 10]
  stride = 2
  seq_lens = input_utt_lens
  for i in xrange(2):
    seq_lens = tf.div(seq_lens, stride)
    seq_lens = tf.subtract(seq_lens, int(width[i]/2 - 1))
  #seq_lens = tf.Print(seq_lens, [seq_lens], "Conved seq len: ")
  return seq_lens

def train():
  num_classes = Dataset.char_num + 1
  data_format = CONF.DATA_FORMAT
  
  # inputs
  with tf.name_scope('inputs'):
    # defulat input data shape is [bs, seq, 161]
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, None, Dataset.freq_bins])
    input_label_indice = tf.placeholder(dtype=tf.int64, shape=[None, None])
    input_label_value  = tf.placeholder(dtype=tf.int32, shape=[None])
    input_label_shape  = tf.placeholder(dtype=tf.int64, shape=[2])
    input_utt_lens = tf.placeholder(dtype=tf.int32, shape=[None])

  logits = get_logits(input_data, num_classes, data_format)


  # Calculate the average ctc loss across the batch.
  input_label = tf.SparseTensor(indices = input_label_indice,
                                values = input_label_value,
                                dense_shape = input_label_shape)

  #seq_len = tf.shape(logits)[0]
  #bs = tf.shape(logits)[1]
  #seq = tf.TensorArray(tf.int32, size=bs)
  #seq.write(0, seq_len)
  #input_utt_lens.write(0, seq_len)
  seq_lens = get_seq_lens(input_utt_lens)
  ctc_loss = tf.nn.ctc_loss(labels=input_label,
                        inputs = tf.cast(logits, tf.float32),
                        sequence_length = seq_lens,
                        preprocess_collapse_repeated = True,
                        time_major = True)  # use shape [seq, bs, dim]

  loss_op = tf.reduce_mean(ctc_loss, name = 'ctc_loss_mean')
  DEBUG(tf.summary.scalar('ctc_loss_mean', loss_op))

  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(ARGS.learning_rate)
    train_op = optimizer.minimize(loss_op)

  # Create a saver for writing training checkpoints.
  #saver = tf.train.Saver()

  dataset = Dataset(use_dummy=True)
  def feed_dict():
    dat, lbl_ind, lbl_val, lbl_shp, utt = dataset.next_batch(ARGS.batch_size)
    return {input_data: dat,
            input_label_indice: lbl_ind,
            input_label_value: lbl_val,
            input_label_shape: lbl_shp,
            input_utt_lens: utt}

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    run_options = None
    run_metadata = None
    # Merge all the summaries and write them out to ARGS.log_dir
    if ARGS.debug:
      summary_op = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(ARGS.log_dir + '/train', sess.graph)
      test_writer = tf.summary.FileWriter(ARGS.log_dir + '/test') #, sess.graph)
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()

    for i in range(ARGS.max_iter):
      train_loss, _ = sess.run([loss_op, train_op], feed_dict=feed_dict(),
               options=run_options, run_metadata=run_metadata)
      if i % CONF.LOSS_ITER == 0:
        logger.info('iter %d, training loss %g' % (i, train_loss))
        #summary, train_acc = sess.run([summary_op, train_op],
        #                      feed_dict={input: batch[0], label: batch[1]},
        #                      options=run_options,
        #                      run_metadata=run_metadata)
        if ARGS.debug:
          summary = sess.run(summary_op, feed_dict=feed_dict(),
                             options=run_options, run_metadata=run_metadata)
        DEBUG(train_writer.add_run_metadata(run_metadata, 'iter%03d' % i))
        DEBUG(train_writer.add_summary(summary, i))

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
