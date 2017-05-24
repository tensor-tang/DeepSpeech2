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
  unit test
"""

from config_helper import default_name


## test config helper
@default_name("abc")
def conv(n,
         a=1,
         b=2,
         c="xxxxxx",name=None):
  print(n, a, b, c, name)
  
  
@default_name()
def bn(n,
         a=1,
         b=2,
         c="xxxxxx",name=None):
  print(n, a, b, c, name)

  
conv(0)
conv(1)
conv(0, name="bbbb")
conv(1, name="bbbb")
bn(0)
bn(1)
bn(0, name="ccc")
bn(1, name="ccc")