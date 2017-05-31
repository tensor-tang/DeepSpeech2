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
import conf as CONF

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
  parser.add_argument(
    '--debug',
    type=bool,
    default=CONF.DEBUG,
    help='If true will in debug mode and logging debug')
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
  LOG_DIR = os.path.abspath(CONF.LOG_DIR)
  parser.add_argument(
    '--log_dir',
    type=str,
    default=LOG_DIR,
    help='Summaries log directory')

  args, unparsed = parser.parse_known_args()

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

