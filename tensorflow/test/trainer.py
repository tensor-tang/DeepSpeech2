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

from config_helper import logger, parse_args, set_debug_mode, DEBUG
from ds2_dataset   import Dataset
from ds2           import get_loss

ARGS = None


def train_loop():
  def feed_dict():
    dat, lbl_ind, lbl_val, lbl_shp, utt = dataset.next_batch(ARGS.batch_size)
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

  with tf.name_scope('loss'):
    loss_op = get_loss(input_data, input_label, input_utt_lens, ARGS.data_format)

  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(ARGS.learning_rate)
    train_op = optimizer.minimize(loss_op)

  # Create a saver for writing training checkpoints.
  #saver = tf.train.Saver()

  dataset = Dataset(use_dummy=ARGS.use_dummy)

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
      if i % ARGS.loss_iter_interval == 0:
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
  set_debug_mode(ARGS.debug)
  logger.debug(ARGS)

  if tf.gfile.Exists(ARGS.log_dir):
    tf.gfile.DeleteRecursively(ARGS.log_dir)
  tf.gfile.MakeDirs(ARGS.log_dir)

  train_loop()

if __name__ == '__main__':
  ARGS, unparsed = parse_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
