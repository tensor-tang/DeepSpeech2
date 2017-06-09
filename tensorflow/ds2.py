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

from ds2_config_helper import logger, default_name, DEBUG
from ds2_dataset       import Dataset


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

def _variable_on_cpu(name, shape, initializer=None, use_fp16=False):
  """Helper to create a Variable stored on cpu memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable. Such as:
      tf.constant_initializer(-0.05),
      tf.zeros_initializer()
      tf.ones_initializer()
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, dtype=dtype, initializer=initializer)
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
def conv_layer(x,
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
    DEBUG(tf.summary.histogram('pre_' + name, x))
    conv = tf.nn.conv2d(x, wgt, stride, padding,
                        data_format=data_format, name=name)
    DEBUG(tf.summary.histogram('post_' + name, conv))
    if with_bias:
      bias = bias_variable('param_bias', [oc])
      DEBUG(variable_summaries(bias))
      conv = tf.nn.bias_add(conv, bias, data_format, 'add_bias')
      DEBUG(tf.summary.histogram('post_bias', conv))
    return conv


@default_name("bn")
def bn_layer(x,
                  is_training=True,
                  data_format='NHWC',
                  eps=1e-5,
                  name=None):
  """batch normalization layer
  spatial or sequential
  currently only work on NHWC
  """

  input_shape = x.get_shape()
  #with tf.name_scope(name):
  with tf.variable_scope(name, reuse = False):
    axis = list(range(len(input_shape) - 1))
    if data_format == 'NCHW':
      shape = input_shape[1]
    else:
      shape = input_shape[-1]

    shift = _variable_on_cpu('shift', shape, initializer=tf.zeros_initializer())
    scale = _variable_on_cpu('scale', shape, initializer=tf.ones_initializer())

    bn, _, _ = tf.nn.fused_batch_norm(
        x, scale, shift, mean=None, variance=None, epsilon=eps,
        data_format=data_format, is_training=is_training, name=name)
    bn.set_shape(input_shape)
    return bn


@default_name("crelu")
def relu_layer(x, capping=None, name=None):
  """ReLU layer, clipped if set capping
  y = min(max(0, x), capping)
  """
  with tf.name_scope(name):
    y = tf.nn.relu(x)
    if capping is not None:
      y = tf.minimum(y, capping)
    return y


@default_name("fc")
def fc_layer(x, dim_out, dim_in=None, with_bias=True, name=None):
  '''fc_layer returns a full connected layer
     Wx + b
     param:
      if dim_in is None, will set as dim_out
  '''
  logger.debug(name)
  if dim_in is None:
    dim_in = dim_out

  with tf.name_scope(name):
    DEBUG(tf.summary.histogram('pre_' + name, x))
    x = tf.reshape(x, [-1, dim_in])
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

  data_format = 'NCHW'
  for i in xrange(len(CONVS)):
    with tf.variable_scope(('CBR_%d' % i), reuse = False):
      feat = conv_layer(feat,
                        CONVS[i].input_channels,
                        CONVS[i].num_kernels,
                        [CONVS[i].kernel_height, CONVS[i].kernel_width],
                        [CONVS[i].stride_height, CONVS[i].stride_width],
                        data_format=data_format)
      feat = bn_layer(feat, data_format=data_format)
      feat = relu_layer(feat, capping=20)

  with tf.name_scope('reshape_trans'):
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

  data_format = 'NHWC'
  for i in xrange(len(CONVS)):
    with tf.variable_scope(('CBR_%d' % i), reuse = False):
      feat = conv_layer(feat,
                        CONVS[i].input_channels,
                        CONVS[i].num_kernels,
                        [CONVS[i].kernel_width, CONVS[i].kernel_height],
                        [CONVS[i].stride_width, CONVS[i].stride_height],
                        data_format=data_format)
      feat = bn_layer(feat, data_format=data_format)
      feat = relu_layer(feat, capping=20)

  with tf.name_scope('trans_reshape'):
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


def cbr_combine(x, data_format):
  '''combine of conv, batch_norm and cliiped relu
  return feature shape as [seq, bs, dim]
  '''
  # TODO: batchnorm relu
  if data_format == 'NCHW':
    feat = _nchw(x)
  else:
    feat = _nhwc(x)
  return feat


@default_name("seq_bn")
def seq_batch_norm_old(x, eps=1e-5, name = None, is_train = True):
  '''sequence batch normalization
  input is [seq, bs, dim]
  '''
  assert len(x.get_shape()) == 3, 'input should be 3-D'
  with tf.variable_scope(name, reuse=True):
    inputs_shape = tf.shape(x) #x.get_shape()
    
    seq_len = inputs_shape[0]
    dim = inputs_shape[2]
    # split 
    seq_bn = tf.split(x, num_or_size_splits=seq_len, axis=0)
    print('seq len %d' % len(seq_bn))
    shift = _variable_on_cpu('shift', dim, initializer = tf.zeros_initializer())
    scale = _variable_on_cpu('scale', dim, initializer = tf.ones_initializer())
    batch_mean, batch_var = tf.nn.moments(x, [0], name = 'moments')
    ema = tf.train.ExponentialMovingAverage(decay = 0.5)
    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    if is_train:
      mean, var = mean_var_with_update()
    else:
      mean, var = lambda : (ema.average(batch_mean), ema.average(batch_var))
    for i in xrange(len(seq_bn)):
      bn = tf.nn.batch_normalization(seq_bn[i], mean, var, shift, scale, eps)
  return bn



INFERENCE_DTYPE = tf.float32 # this global not work, need check


class SeqBNCell(tf.contrib.rnn.BasicRNNCell):
  """ This is cell is only used to seq batch norm,
  not directly a rnn cell, just use the rnn shell.
  """

  @default_name("seqBNCell")
  def __init__(self, num_units, activation = tf.nn.relu6, is_train=True, eps=1e-5, use_fp16=False, name=None):
    self._num_units = num_units
    self.use_fp16 = use_fp16
    self.eps = eps
    self.name = name
    self.is_train = is_train
    logger.debug(self.name)

  def __call__(self, inputs, state):
    """Most basic RNN:
    input is [bs, dim]
    output is [bs, dim]
    
    """
    logger.debug(self.name)
    with tf.variable_scope(self.name) as scope:
      dim = inputs.get_shape()[1]
      shift = _variable_on_cpu('shift', dim, initializer=tf.zeros_initializer())
      scale = _variable_on_cpu('scale', dim, initializer=tf.ones_initializer())
      logger.debug(shift.name)
      batch_mean, batch_var = tf.nn.moments(inputs, [0], name = 'moments')
      ema = tf.train.ExponentialMovingAverage(decay = 0.5)
      if self.is_train:
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
          mean = tf.identity(batch_mean)
          var =  tf.identity(batch_var)
      else:
        mean, var = lambda : (ema.average(batch_mean), ema.average(batch_var))
      scope.reuse_variables()
      bn = tf.nn.batch_normalization(inputs, mean, var, shift, scale, self.eps)
      return bn, bn

@default_name("seq_bn")
def seq_batch_norm(x, dim, seq_lens, eps=1e-5, name = None, is_train = True):
  '''sequence batch normalization
  input is [seq, bs, dim]
  '''
  assert len(x.get_shape()) == 3, 'input should be 3-D'
  dtype = INFERENCE_DTYPE
  logger.debug(dtype)
  with tf.variable_scope(name) as scope:
    bncell = SeqBNCell(num_units=dim)
    #cell = tf.contrib.rnn.CompiledWrapper(bncell)
    #cell = tf.contrib.rnn.RNNCell(bncell)
    cell = tf.contrib.rnn.MultiRNNCell([bncell])
    seqbn_output, _ = tf.nn.dynamic_rnn(
                        cell, x,
                        sequence_length = seq_lens,
                        dtype = dtype,
                        time_major = True,  # input is [seq, bs, dim]
                        swap_memory = True)
    return seqbn_output


@default_name("rnn_op")
def rnn_op(x, seq_lens, dim_out, dim_in=None, use_bi_dir=True, name=None):
  '''
  input is [seq, bs, dim_in]
  rnn_op: fc (without bias) -> seq bn -> rnn (skip_input)
  output is [seq, bs, dim_out]
  '''
  dtype = INFERENCE_DTYPE
  assert len(x.get_shape()) == 3, 'input should be 3-D'
  if dim_in is None:
    dim_in = dim_out

  input_shape = tf.shape(x)
  x = fc_layer(x, dim_out=dim_out, dim_in=dim_in, with_bias=False)
  x = tf.reshape(x, [input_shape[0], input_shape[1], dim_out])
  #print('out shape of fc', x.get_shape())
  x = seq_batch_norm(x, dim=dim_out, seq_lens=seq_lens)
#  rnn_input = seq_batch_norm(x)
  '''
  rnn_cell = RNNCellSkipInput(dim_out, use_fp16 = use_fp16)
  if use_bi_dir:
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                            rnn_cell, rnn_cell, rnn_input,
                            sequence_length=rnn_seq_lens, dtype = dtype,
                            time_major = True, scope = scope.name,
                            swap_memory = False)
    outputs_fw, outputs_bw = outputs
    rnn_outputs = tf.add(outputs_fw, outputs_bw)
  else:
    rnn_outputs, _ = tf.nn.dynamic_rnn(
                            rnn_cell, rnn_input,
                            sequence_length = rnn_seq_lens,
                            dtype = dtype, time_major = True,
                            scope = scope.name,
                            swap_memory = False)
  '''

  return x






def rnn_layers(x, seq_lens, layer_nums=7):
  '''return feature shape as [seq, bs, dim]
  '''
  assert layer_nums > 1, 'at least 1 rnn layer'
  assert len(x.get_shape()) == 3, 'input should be 3-D'
  dim_in = 32 * 75
  dim_out = 1760
  with tf.variable_scope('BRNN_0') as scope:
    x = rnn_op(x, seq_lens=seq_lens, dim_out=dim_out, dim_in=dim_in, name=scope.name)
  for i in range(1, layer_nums):  # range: [1, layer_nums)
    with tf.variable_scope('BRNN_%d' % i) as scope:
      x = rnn_op(x, seq_lens=seq_lens, dim_out=dim_out, name=scope.name)
  #x = tf.Print(x, [x.get_shape()], "out shape of rnn", 1760)
  print('out shape of rnn', x.get_shape())
  return x



def inference(input_data, data_format, input_utt_lens, dtype=tf.float32):
  '''inference return logits, seq_lens
  '''
  assert data_format in ["NCHW", "NHWC"], "only support nchw or nhwc yet"

  INFERENCE_DTYPE = dtype # tf.float16, tf.float32
  feat = cbr_combine(input_data, data_format)

  # actually this seq_lens is equal [seq, seq, ..., seq] of len batch_size
  # since ctc loss need them as specific number so only can cal from input 
  seq_lens = get_seq_lens(input_utt_lens)

  rnn_out = rnn_layers(feat, seq_lens)
  # shape from rnn output is 3-D: [seq, bs, dim]
  rnn_shape = tf.shape(rnn_out)
  seq = rnn_shape[0]
  bs  = rnn_shape[1]

  # ctc needs 1 more class for blank
  num_classes = Dataset.char_num + 1
  fc_out = fc_layer(rnn_out, dim_out=num_classes, dim_in=1760) # dim# todo: magic number

  # keep 3-D shape since ctc needs
  logits = tf.reshape(fc_out, [seq, bs, num_classes])

  return logits, seq_lens

def get_loss(input_data, input_label, input_utt_lens, data_format):
  '''get loss
  input params
    data    : tf.placeholder
    label   : tf.SparseTensor
    utt_lens: tf.placeholder

    data_format: only support NCHW and NHWC
  
  return the tf loss operation
  '''
  
  logits, seq_lens = inference(input_data, data_format, input_utt_lens)

  # Reuse variables
  #tf.get_variable_scope().reuse_variables()
  
  ctc_loss = tf.nn.ctc_loss(labels=input_label,
                        inputs = tf.cast(logits, tf.float32),
                        sequence_length = seq_lens,
                        preprocess_collapse_repeated = True,
                        time_major = True)  # use shape [seq, bs, dim]

  # Calculate the average ctc loss across the batch.
  ctc_loss_mean = tf.reduce_mean(ctc_loss, name = 'ctc_loss_mean')
  DEBUG(tf.summary.scalar('ctc_loss_mean', ctc_loss_mean))
  return ctc_loss_mean

  

