# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2018 Rutika Banakar
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

"""Script to train/evaluate/test the summarization model"""

import sys
import time

import tensorflow as tf
import batch_reader
# import data
import seq2seq_attention_decode
import seq2seq_attention_model
from helper.vocabulary import Vocabulary2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('article_key', 'article', 'tf.Example feature key for article.')
tf.app.flags.DEFINE_string('abstract_key', 'headline', 'tf.Example feature key for abstract.')
tf.app.flags.DEFINE_string('log_root', '', 'Directory for model root.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')
tf.app.flags.DEFINE_string('train_dir', '', 'Directory for train.')
tf.app.flags.DEFINE_string('eval_dir', '', 'Directory for eval.')
tf.app.flags.DEFINE_string('decode_dir', '', 'Directory for decode summaries.')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_integer('max_run_steps', 10000000, 'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('max_article_sentences', 2, 'Max number of first sentences to use from the article')
tf.app.flags.DEFINE_integer('max_abstract_sentences', 100, 'Max number of first sentences to use from the abstract')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 60, 'How often to checkpoint.')
tf.app.flags.DEFINE_bool('use_bucketing', False, 'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', False, 'Truncate inputs that are too long. If False, examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")


def _RunningAvgLoss(loss, running_avg_loss, summary_writer, step, decay=0.999):
  """Calculate the running average of losses."""
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)
  loss_sum = tf.Summary()
  loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)
  return running_avg_loss


def _Train(model, data_batcher):
  """Runs model training."""
  with tf.device('/cpu:0'):
    model.build_graph()
    saver = tf.train.Saver()
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
    sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=FLAGS.checkpoint_secs,
                             global_step=model.global_step)
    sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
        allow_soft_placement=True))
    running_avg_loss = 0
    step = 0
    while not sv.should_stop() and step < FLAGS.max_run_steps:
      (article_batch, abstract_batch, targets, article_lens, abstract_lens,
       loss_weights, _, _) = data_batcher.NextBatch()
      (_, summaries, loss, train_step) = model.run_train_step(
          sess, article_batch, abstract_batch, targets, article_lens,
          abstract_lens, loss_weights)

      summary_writer.add_summary(summaries, train_step)
      running_avg_loss = _RunningAvgLoss(
          running_avg_loss, loss, summary_writer, train_step)
      step += 1
      if step % 100 == 0:
        summary_writer.flush()
    sv.Stop()
    return running_avg_loss


def _Eval(model, data_batcher, vocab=None):
  """Runs model eval."""
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  running_avg_loss = 0
  step = 0
  while True:
    time.sleep(FLAGS.eval_interval_secs)
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.train_dir)
      continue

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    (article_batch, abstract_batch, targets, article_lens, abstract_lens,
     loss_weights, _, _) = data_batcher.NextBatch()
    (summaries, loss, train_step) = model.run_eval_step(
        sess, article_batch, abstract_batch, targets, article_lens,
        abstract_lens, loss_weights)
    tf.logging.info(
        'article:  %s',
        ' '.join(data.Ids2Words(article_batch[0][:].tolist(), vocab)))
    tf.logging.info(
        'abstract: %s',
        ' '.join(data.Ids2Words(abstract_batch[0][:].tolist(), vocab)))

    summary_writer.add_summary(summaries, train_step)
    running_avg_loss = _RunningAvgLoss(
        running_avg_loss, loss, summary_writer, train_step)
    if step % 100 == 0:
      summary_writer.flush()


def main(unused_argv):
  vocab = Vocabaluray2(FLAGS.vocab_path, 150000)
  # Check for presence of required special tokens.
  # assert vocab.CheckVocab(data.PAD_TOKEN) > 0
  # assert vocab.CheckVocab(data.UNKNOWN_TOKEN) >= 0
  # assert vocab.CheckVocab(data.SENTENCE_START) > 0
  # assert vocab.CheckVocab(data.SENTENCE_END) > 0

  batch_size = 4
  if FLAGS.mode == 'decode':
    batch_size = FLAGS.beam_size

  hps = seq2seq_attention_model.HParams(
      mode=FLAGS.mode,  # train, eval, decode
      min_lr=0.01,  # min learning rate.
      lr=0.15,  # learning rate
      batch_size=batch_size,
      enc_layers=4,
      enc_timesteps=FLAGS.max_enc_steps,
      dec_timesteps=FLAGS.max_dec_steps,
      min_input_len=2,  # discard articles/summaries < than this
      num_hidden=FLAGS.hidden_dim,  # for rnn cell
      emb_dim=FLAGS.emb_dim,  # If 0, don't use embedding
      max_grad_norm=2,
      num_softmax_samples=4096)  # If 0, no sampled softmax.

  batcher = batch_reader.Batcher(
      FLAGS.data_path, vocab, hps, FLAGS.article_key,
      FLAGS.abstract_key, FLAGS.max_article_sentences,
      FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,
      truncate_input=FLAGS.truncate_input)
  tf.set_random_seed(FLAGS.random_seed)

  if hps.mode == 'train':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    _Train(model, batcher)
  elif hps.mode == 'eval':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    _Eval(model, batcher, vocab=vocab)
  elif hps.mode == 'decode':
    decode_mdl_hps = hps
    # Only need to restore the 1st step and reuse it since
    # we keep and feed in state for each step's output.
    decode_mdl_hps = hps._replace(dec_timesteps=1)
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        decode_mdl_hps, vocab, num_gpus=FLAGS.num_gpus)
    decoder = seq2seq_attention_decode.BSDecoder(model, batcher, hps, vocab)
    decoder.DecodeLoop()


if __name__ == '__main__':
  tf.app.run()
