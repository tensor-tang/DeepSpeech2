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
  Configure for DeepSpeech 2
  The configures are just defualt settings
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
workspace configuration
'''
# logs dir for logs and models
LOG_DIR = "./logs"

#environments setting for all processes in cluster job
LD_LIBRARY_PATH = "/usr/lib64:/usr/local/lib:/usr/local/cuda/lib64"

DEBUG = True

'''
network configuration
'''
DATA_FORMAT = "NHWC"  #"NCHW"

# batch size
BATCH_SIZE = 32

# use dummy data
USE_DUMMY = True

# max iterations
MAX_ITER = 1000

# every 15 iteration test accuracy
TEST_INTERVAL = 15

# every 10 iteration print the loss
LOSS_ITER = 10

# learning rate
LEARNING_RATE = 0.001






