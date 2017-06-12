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
import time
import tensorflow as tf
import ds2

from ds2_config_helper import logger, parse_args, set_debug_mode, init_variables
from ds2_config_helper import save_timeline, save_tfprof
from ds2_dataset       import Dataset

import os

ARGS = None


def train_loop():
  '''ds2 train loop
  '''
  def feed_dict():
    '''feed inputs for ds2
    '''
    start_time = time.time()
    dat, lbl_ind, lbl_val, lbl_shp, utt = dataset.next_batch(ARGS.batch_size)
    duration = (time.time() - start_time) * 1000
    logger.debug('reading data %.3f ms' % duration)
    return {input_data:         dat,
            input_label_indice: lbl_ind,
            input_label_value:  lbl_val,
            input_label_shape:  lbl_shp,
            input_utt_lens:     utt}

  with tf.name_scope('inputs'):
    # defulat input data shape is [bs, seq, 161]
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, None, Dataset.freq_bins])
    input_label_indice = tf.placeholder(dtype=tf.int64, shape=[None, None])
    input_label_value  = tf.placeholder(dtype=tf.int32, shape=[None])
    input_label_shape  = tf.placeholder(dtype=tf.int64, shape=[2])
    input_utt_lens = tf.placeholder(dtype=tf.int32, shape=[None])
    input_label = tf.SparseTensor(indices = input_label_indice,
                                  values = input_label_value,
                                  dense_shape = input_label_shape)

  #with tf.name_scope('ds2'):
  with tf.variable_scope('ds2'):
    loss_op = ds2.get_loss(input_data, input_label, input_utt_lens, ARGS.data_format)

  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(ARGS.learning_rate)
    train_op = optimizer.minimize(loss_op)

  dataset = Dataset(use_dummy=ARGS.use_dummy)

  # Create a saver. TODO: how about only inference
  saver = tf.train.Saver(tf.global_variables())

  sess = tf.Session(config = tf.ConfigProto(
                      allow_soft_placement = False, #True,
                      log_device_placement = False,
                      inter_op_parallelism_threads = 8,
                      intra_op_parallelism_threads = 32)
                    ) 
  last_iter = init_variables(sess, saver, ARGS.checkpoint_dir)
  run_options = None
  run_metadata = None
  # Merge all the summaries and write them out to ARGS.log_dir
  if ARGS.debug:
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(ARGS.log_dir, sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

  #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
  sum_time = 0
  min_time = float('inf')
  for i in range(ARGS.max_iter):
    start_time = time.time()
    train_loss, _ = sess.run([loss_op, train_op], feed_dict=feed_dict(),
                             options=run_options, run_metadata=run_metadata)
    duration = time.time() - start_time
    sum_time += duration
    if min_time > duration:
      min_time = duration
    # Save the model checkpoint periodically
    if i % ARGS.checkpoint_iter == 0 or (i + 1) == ARGS.max_iter:
      checkpoint_path = ARGS.checkpoint_dir + '/ds2_model'
      global_step = last_iter + i
      saver.save(sess, checkpoint_path, global_step = global_step)
      logger.info('save checkpoint to %s-%d' % (ARGS.checkpoint_dir, global_step))
        
    if i == ARGS.profil_iter:
      # save timeline to a json and only save once 
      filename = ARGS.log_dir + ('/timeline_iter%d' % i) + '.json'
      logger.debug('save timeline to: ' + filename)
      save_timeline(filename, run_metadata)
      # tfprof
      logger.debug('save TFprof to: ' + ARGS.log_dir)
      save_tfprof(ARGS.log_dir, sess.graph, run_metadata)
      
    if i % ARGS.loss_iter_interval == 0:
      format_str = 'iter %d, training loss = %.2f (%.3f sec/iter)'
      log_content = (i, train_loss, duration)
      if ARGS.debug:
        s = time.time()
        summary = sess.run(summary_op, feed_dict=feed_dict(),
                           options=run_options, run_metadata=run_metadata)
        summary_writer.add_run_metadata(run_metadata, 'iter%03d' % i)
        summary_writer.add_summary(summary, i)
        d = time.time() - s
        format_str += ', summary %.3g sec/iter'
        log_content += (d,)
      logger.info(format_str % log_content)

    if ARGS.debug:
      summary_writer.close()
  logger.info('avg time: %.3f ms, min_time: %.3f ms' % (sum_time*1000/ARGS.max_iter, min_time*1000)) 
  sess.close()

def main(_):
  set_debug_mode(ARGS.debug)
  logger.debug(ARGS)

  if tf.gfile.Exists(ARGS.log_dir):
    tf.gfile.DeleteRecursively(ARGS.log_dir)
  tf.gfile.MakeDirs(ARGS.log_dir)

  if not tf.gfile.Exists(ARGS.checkpoint_dir):
    tf.gfile.MakeDirs(ARGS.checkpoint_dir)
  os.environ["KMP_BLOCKTIME"] = "1"
  os.environ["KMP_SETTINGS"] = "1"
  os.environ["OMP_NUM_THREADS"] = "32"
  os.environ["MKL_NUM_THREADS"] = "32" 
  os.environ["OMP_DYNAMIC"] = "false"
  os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
  train_loop()

if __name__ == '__main__':
  ARGS, unparsed = parse_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
