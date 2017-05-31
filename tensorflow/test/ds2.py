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
  DeepSpeech 2 Model
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from config_helper import logger, default_name, DEBUG
from ds2_dataset   import Dataset


class ConvParam(object):
  '''conv params
  size of kernel and stride
  input and output channel number
  '''
  def __init__(self, ic, oc, kh, kw, sh, sw):
    self.input_channels = ic
    self.num_kernels    = oc
    self.kernel_height  = kh
    self.kernel_width   = kw
    self.stride_height  = sh
    self.stride_width   = sw

CONVS = list()
# params saved as nchw format
CONVS.append(ConvParam(1,  32, 5, 20, 2, 2))
CONVS.append(ConvParam(32, 32, 5, 10, 1, 2)) 

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
                data_format,
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
    return shape [seq, bs, dim]
  '''
  # transpose from [bs, utt, 161] to [bs, 161, utt]
  trans = tf.transpose(input_data, perm = [0, 2, 1])
  # to shape: [bs, 1, 161, utt]
  feat = tf.expand_dims(trans, 1)

  for i in xrange(len(CONVS)):
    feat = conv_layer(feat,
                      CONVS[i].input_channels,
                      CONVS[i].num_kernels,
                      [CONVS[i].kernel_height, CONVS[i].kernel_width],
                      [CONVS[i].stride_height,CONVS[i].stride_width],
                      data_format='NCHW')

  # feat shape [bs, 32, 75, seq]
  feat_shape = tf.shape(feat)
  dim = feat_shape[1] * feat_shape[2]
  seq = feat_shape[3]

  # reshape to [bs, dim , seq]
  feat = tf.reshape(feat, [-1, dim, seq])

  # from [bs, dim , seq] to [seq, bs, dim]
  out = tf.transpose(feat, [2, 0, 1])
  return out
  
def _nhwc(input_data):
  '''use nhwc data format
    return shape [seq, bs, dim]
  '''
  # to shape: [bs, utt, 161, 1]
  feat = tf.expand_dims(input_data, -1)

  for i in xrange(len(CONVS)):
    feat = conv_layer(feat,
                      CONVS[i].input_channels,
                      CONVS[i].num_kernels,
                      [CONVS[i].kernel_width, CONVS[i].kernel_height],
                      [CONVS[i].stride_width,CONVS[i].stride_height],
                      data_format='NHWC')

  # transpose from [bs, seq, 75, 32] to [seq, bs, 32, 75]
  feat = tf.transpose(feat, [1, 0, 3, 2])
  feat_shape = tf.shape(feat)

  # reshape to [seq, bs, dim]
  seq = feat_shape[0]
  dim = feat_shape[2] * feat_shape[3]
  out = tf.reshape(feat, [seq, -1, dim])
  return out


def get_seq_lens(input_utt_lens):
  '''according to conv output len: floor((size - filter_len) / stride) + 1 
  return after convs seq_lens
  '''
  seq_lens = input_utt_lens
  for i in xrange(len(CONVS)):
    seq_lens = tf.div(seq_lens, CONVS[i].stride_width)
    seq_lens = tf.subtract(seq_lens, int(CONVS[i].kernel_width/CONVS[i].stride_width - 1))
  #seq_lens = tf.Print(seq_lens, [seq_lens], "Conved seq len: ")
  return seq_lens


def cbr_combine(input_tensor, data_format):
  '''combine of conv, batch_norm and cliiped relu
  return feature shape as [seq, bs, dim]
  '''
  # TODO: batchnorm relu
  if data_format == 'NCHW':
    feat = _nchw(input_tensor)
  else:
    feat = _nhwc(input_tensor)
  return feat

def rnn_layers(input_tensor, layer_nums=7):
  '''return feature shape as [seq, bs, dim]
  '''
  # TODO:...
  return input_tensor


def get_loss(input_data, input_label, input_utt_lens, data_format):
  '''get loss
  input params
    data    : tf.placeholder
    label   : tf.SparseTensor
    utt_lens: tf.placeholder

    data_format: only support NCHW and NHWC
  
  return the tf loss operation
  '''
  if data_format not in ["NCHW", "NHWC"]:
    logger.fatal("only support nchw or nhwc yet")

  feat = cbr_combine(input_data, data_format)

  rnn_out = rnn_layers(feat)

  # ctc needs 1 more class for blank
  num_classes = Dataset.char_num + 1
  fc_out = fc_layer(rnn_out, num_classes, dim_in=32 * 75) # todo: magic number

  # shape from rnn output is 3-D: [seq, bs, dim], keep 3-D since ctc needs
  rnn_shape = tf.shape(rnn_out)
  seq = rnn_shape[0]
  logits = tf.reshape(fc_out, [seq, -1, num_classes])

  # actually this seq_lens is equal [seq, seq, ..., seq] of len batch_size
  # since ctc loss need them as specific number so only can cal from input 
  seq_lens = get_seq_lens(input_utt_lens)
  
  ctc_loss = tf.nn.ctc_loss(labels=input_label,
                        inputs = tf.cast(logits, tf.float32),
                        sequence_length = seq_lens,
                        preprocess_collapse_repeated = True,
                        time_major = True)  # use shape [seq, bs, dim]

  # Calculate the average ctc loss across the batch.
  ctc_loss_mean = tf.reduce_mean(ctc_loss, name = 'ctc_loss_mean')
  DEBUG(tf.summary.scalar('ctc_loss_mean', ctc_loss_mean))
  return ctc_loss_mean

  

