from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import convnet_tf
import cifar10_utils
import time

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer, # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer, # Adadelta
                  'adagrad': tf.train.AdagradOptimizer, # Adagrad
                  'ADAM': tf.train.AdamOptimizer, # Adam
                  'rmsprop': tf.train.RMSPropOptimizer # RMSprop
                  }


DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in range(steps_per_epoch):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
    images_feed = images_feed.reshape(FLAGS.batch_size, -1)
    feed_dict = {
      images_placeholder: images_feed,
      labels_placeholder: labels_feed,
    }
    true_count += np.sum(sess.run(eval_correct, feed_dict=feed_dict))
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def train():
  """
  Performs training and evaluation of ConvNet model.

  First define your graph using class ConvNet and its methods. Then define
  necessary operations such as savers and summarizers. Finally, initialize
  your model within a tf.Session and do the training.

  ---------------------------
  How to evaluate your model:
  ---------------------------
  Evaluation on test set should be conducted over full batch, i.e. 10k images,
  while it is alright to do it over minibatch for train set.

  ---------------------------------
  How often to evaluate your model:
  ---------------------------------
  - on training set every print_freq iterations
  - on test set every eval_freq iterations

  ------------------------
  Additional requirements:
  ------------------------
  Also you are supposed to take snapshots of your model state (i.e. graph,
  weights and etc.) every checkpoint_freq iterations. For this, you should
  study TensorFlow's tf.train.Saver class.
  """

  # Set the random seeds for reproducibility. DO NOT CHANGE.
  tf.set_random_seed(42)
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  ########################
  # Load data
  print('Get cifar data')
  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
  x, y = cifar10.train.next_batch(FLAGS.batch_size)
  x = x.reshape(FLAGS.batch_size, -1)
  input_dimensions = x.shape[1]
  n_classes = y.shape[1]
  print('Data obtained')


  # Create Graph and run Session
  with tf.Graph().as_default():
    # Initialize ConvNet instance
    is_training = tf.Variable([True], tf.bool)
    ConvNet    = convnet_tf.ConvNet(OPTIMIZER_DICT[OPTIMIZER_DEFAULT], is_training, n_classes)

    # Generate placeholders for the images and labels.
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, input_dimensions))
    labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, n_classes))

    # Build a Graph that computes predictions from the inference model.
    logits = ConvNet.inference(images_placeholder)

    # Add to the Graph the Ops for loss calculation.
    loss   = ConvNet.loss(logits,labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_step = ConvNet.train_step(loss, FLAGS)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = ConvNet.accuracy(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()
    # Add the variable initializer Op.
    init = tf.global_variables_initializer()
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    # Create a session for running Ops on the Graph.
    sess = tf.Session()
    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    # And then after everything is built:
    # Run the Op to initialize the variables.
    sess.run(init)

    # Training loop
    for step in range(FLAGS.max_steps):
      start_time = time.time()
      feed_dict = {
        images_placeholder: x,
        labels_placeholder: y,
      }
      _, loss_value = sess.run([train_step, loss], feed_dict=feed_dict)
      x, y = cifar10.train.next_batch(FLAGS.batch_size)
      x = x.reshape(FLAGS.batch_size, -1)
      if step % 100 == 0:
        duration = time.time() - start_time
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                cifar10.test)
  #raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  ########################

def initialize_folders():
  """
  Initializes all folders in FLAGS variable.
  """

  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)

  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  if not tf.gfile.Exists(FLAGS.checkpoint_dir):
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main(_):
  print_flags()

  initialize_folders()

  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
  parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
  parser.add_argument('--print_freq', type=int, default=PRINT_FREQ_DEFAULT,
                        help='Frequency of evaluation on the train set')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
  parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')
  parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
