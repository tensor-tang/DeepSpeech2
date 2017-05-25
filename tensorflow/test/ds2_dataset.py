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
  A dataset for DeepSpeech 2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from config_helper import logger


class Dataset(object):
  utt_lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
  counts = [3, 10, 11, 13, 14, 13, 9, 8, 5, 4, 3, 2, 2, 2, 1]
  lbl_lengths = [7, 17, 35, 48, 62, 78, 93, 107, 120, 134, 148, 163, 178, 193, 209]
  freq_bins = 161
  char_num = 29  # 29 chars

  # scale the counts to increase the total dataset
  scale_factor = 1024

  # extra length for more space of random table
  extra_len = 1000

  def __init__(self, batch_size=None, use_dummy=True, seed=None):
    """Construct a DataSet.
    if batch size is none, default use 64
    """
    if not use_dummy:
      logger.fatal("only support dummy data yet")

    self.utt_counts = list()
    for i in xrange(len(self.counts)):
      self.utt_counts.append(self.counts[i] * self.scale_factor)
    logger.debug(self.utt_counts);

    assert len(self.utt_lengths) == len(self.lbl_lengths)
    assert len(self.lbl_lengths) == len(self.counts)

    self.utt_cur_idx = 0
    self.remain_cnt = self.utt_counts[self.utt_cur_idx]
    self.seed = 0 if seed is None else seed
    self.batch_size = batch_size if batch_size is not None else 64

    self.reset_random_table(self.batch_size)


  def reset_random_table(self, batch_size=None, seed=None):
    '''reset random data table and label table
    if batch size is none will use self.batch_size(default is 64)
    '''
    np.random.seed(self.seed if seed is None else seed)

    if batch_size is None:
      batch_size = self.batch_size
    assert batch_size > 0
    
    self.dat_table_h = (max(self.utt_lengths) + self.extra_len) * batch_size
    dat_table = np.random.rand(self.dat_table_h, self.freq_bins)  # ((1500+1000)*bs, 161)

    self.lbl_table_h = (max(self.lbl_lengths) + self.extra_len) * batch_size
    lbl_table = np.random.random_integers(0, self.char_num - 1, self.lbl_table_h)

    self.dat_table = dat_table.astype('float32').tolist()
    self.lbl_table = lbl_table.astype('int').tolist()


  def get_from_table(self, batch_size):
    assert batch_size > 0
    idx = self.utt_cur_idx
    dat_h = self.utt_lengths[idx] * batch_size
    lbl_h = self.lbl_lengths[idx] * batch_size

    dat_start = np.random.randint(0, self.dat_table_h - dat_h)  #[0, len)
    lbl_start = np.random.randint(0, self.lbl_table_h - lbl_h)

    dat = self.dat_table[dat_start : dat_start + dat_h][:] 
    lbl = self.lbl_table[lbl_start : lbl_start + lbl_h]

    return dat, lbl
  
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    if batch_size > self.batch_size:
      self.batch_size = batch_size
      self.reset_random_table(batch_size)

    if self.remain_cnt <= 0:
      self.utt_cur_idx += 1
      if self.utt_cur_idx >= len(self.utt_counts):
        logger.info("one epoch done, restart dataset")
        self.utt_cur_idx = 0
      self.remain_cnt = self.utt_counts[self.utt_cur_idx]

    if self.remain_cnt < batch_size:
      batch_size = self.remain_cnt
    
    self.remain_cnt -= batch_size
    return self.get_from_table(batch_size)

   
