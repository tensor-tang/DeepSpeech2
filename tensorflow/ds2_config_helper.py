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
  config helper for deepspeech 2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import ds2_conf as CONF

import tensorflow as tf

# timeline and tfprof for profiling
from tensorflow.python.client import timeline
from tensorflow.contrib       import tfprof

# logger
import logging
logging.basicConfig(
  format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s', )
logger = logging.getLogger('ds2')
logger.setLevel(logging.INFO)

# for parse_args
import os
import argparse

DEBUG = False


def parse_args():
  '''parse arguments
  return [parsed, unparsed] args
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--batch_size',
    required=False,
    type=int,
    default=CONF.BATCH_SIZE,
    help='batch size for train and test.')
  parser.add_argument(
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
  #parser.add_argument(
    #'--debug',
    #action='store_true',
    #help='run in debug mode and logging debug')

  # choose debug mode
  debug_parser = parser.add_mutually_exclusive_group(required=False)
  debug_parser.add_argument('--debug', dest='debug', action='store_true',
    help='run in debug mode and logging debug')
  debug_parser.add_argument('--no_debug', dest='debug', action='store_false',
    help='run in no debug mode and do not logging debug')
  parser.set_defaults(debug=True if CONF.DEBUG else False)

  parser.add_argument(
    '--profil_iter',
    type=int,
    default=CONF.PROFIL_ITER,
    help='save profiling data(timeline) at one iteration')
  parser.add_argument(
    '--checkpoint_dir',
    type=str,
    default=CONF.CHECKPOINT_DIR,
    help='path to load and save checkpoint')
  parser.add_argument(
    '--checkpoint_iter',
    type=int,
    default=CONF.CHECKPOINT_ITER,
    help='at intervals of how many iterations to save checkpoint')
  parser.add_argument(
    '--data_format',
    type=str,
    default=CONF.DATA_FORMAT,
    help='data format only support NCHW or NHWC')
  parser.add_argument(
    '--loss_iter_interval',
    type=int,
    default=CONF.LOSS_ITER,
    help='at intervals of how many iterations to print the loss')
  parser.add_argument(
    '--log_dir',
    type=str,
    default=CONF.LOG_DIR,
    help='Summaries log directory')

  args, unparsed = parser.parse_known_args()

  args.log_dir = os.path.abspath(args.log_dir)
  args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
  args.data_format = args.data_format.upper()
  # TODO: handle unparsed
  return args, unparsed

def set_debug_mode(yes):
  '''
  set use debug mode or not
  '''
  DEBUG = yes
  if yes:
    logger.setLevel(logging.DEBUG)
  else:
    logger.setLevel(logging.INFO)

def DEBUG(op):
  '''
  only operate in debug mode
  '''
  if DEBUG:
    op
  else:
    pass
    

class DefaultNameFactory(object):
  def __init__(self, name_prefix):
    self.__counter__ = 0
    self.__name_prefix__ = name_prefix

  def __call__(self, func):
    if self.__name_prefix__ is None:
        self.__name_prefix__ = func.__name__
    name = "%s_%d" % (self.__name_prefix__, self.__counter__)
    self.__counter__ += 1
    return name

def default_name(name_prefix=None):
  """
  Decorator to set "name" arguments default to "{name_prefix}_{invoke_count}".

  ..  code:: python

    @default_name("some_name")
    def func(name=None):
      print name    # name will never be None. 
                    # If name is not set, name will be "some_name_%d"

  :param name_prefix: name prefix.
  :type name_prefix: basestring
  :return: a decorator to set default name
  :rtype: callable
  """
  assert name_prefix is None or isinstance(name_prefix, basestring)
  
  name_factory = DefaultNameFactory(name_prefix)
  
  def __impl__(func):
    @functools.wraps(func)
    def __wrapper__(*args, **kwargs):
      def check_args():
        if len(args) != 0:
          argspec = inspect.getargspec(func)
        #print(argspec)
        num_positional = len(argspec.args)
        if argspec.defaults:
          num_positional -= len(argspec.defaults)
        if not argspec.varargs and len(args) > num_positional:
          logger.warning("Should use keyword arguments for non-positional args")
      key = 'name'
      check_args()
      if key not in kwargs or kwargs[key] is None:
        kwargs[key] = name_factory(func)
      return func(*args, **kwargs)
    return __wrapper__
  return __impl__


def save_timeline(filename, run_metadata):
  '''save profiling timeline data with tf.RunMetadata()
  run_metadata: generate in sess.run()
  '''
  if run_metadata is None:
    logger.warning("run_metadata is none, skip save timeline")
    pass
  assert isinstance(filename, str), 'filename should be an string!'
  with open(filename, 'w') as f:
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    f.write(ctf)


def save_tfprof(prefix_path, graph, run_metadata=None):
  '''save TFprof all files
  includes: 
    params.log        - trainable params
    flops.log         - float_ops
    timing_memory.log - timeing memory
    device.log        - params on device
  if graph is none will get default graph from tf
  '''
  # Print trainable variable parameter statistics to stdout
  if graph is None:
    logger.warning("input graph is none, use tf.get_default_graph() instead")
    graph = tf.get_default_graph()
  assert isinstance(prefix_path, str), 'filename should be an string!'

  analyzer = tfprof.model_analyzer

  prof_options = analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
  prof_options['output'] = 'file:outfile=' + prefix_path + "/params.log"
  analyzer.print_model_analysis(graph, run_meta = run_metadata, tfprof_options = prof_options)

  prof_options = analyzer.FLOAT_OPS_OPTIONS
  prof_options['output'] = 'file:outfile=' + prefix_path + "/flops.log" 
  analyzer.print_model_analysis(graph, run_meta = run_metadata, tfprof_options = prof_options)

  prof_options = analyzer.PRINT_ALL_TIMING_MEMORY
  prof_options['output'] = 'file:outfile=' + prefix_path + "/timing_memory.log"  
  analyzer.print_model_analysis(graph, run_meta = run_metadata, tfprof_options = prof_options)

  prof_options = analyzer.PRINT_PARAMS_ON_DEVICE
  prof_options['output'] = 'file:outfile=' + prefix_path + "/device.log" 
  analyzer.print_model_analysis(graph, run_meta = run_metadata, tfprof_options = prof_options)


def init_variables(sess, saver=None, checkpoint_dir=None):
  """ Initialize variables of the graph
  will init from checkpoint if provided
  return the last iter number
  """
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  assert sess is not None, 'session should not be none, use tf.Session()'

  # logger.debug(ckpt)
  if ckpt is None or saver is None or ckpt.model_checkpoint_path is None:
    sess.run(tf.global_variables_initializer())
    return 0

  if ckpt.model_checkpoint_path:
    # Restores from checkpoint
    checkpoint_path = ckpt.model_checkpoint_path
    saver.restore(sess, checkpoint_path)
    # Assuming model_checkpoint_path looks something like:
    #   /my-path/model-1000,
    # extract last_iter from it.
    last_iter = checkpoint_path.split('/')[-1].split('-')[-1]
    logger.info("Init from checkpoint " + checkpoint_path)
    return int(last_iter)