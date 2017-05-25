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

import logging
logging.basicConfig(
  format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s', )
logger = logging.getLogger('ds2')
logger.setLevel(logging.INFO)

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

