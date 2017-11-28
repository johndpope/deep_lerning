"""
This module implements training and evaluation of a multi-layer perceptron in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import mlp_tf
import cifar10_utils
import time

import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DROPOUT_RATE_DEFAULT = 0.
DNN_HIDDEN_UNITS_DEFAULT = '100'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/cifar10'


# This is the list of options for command line arguments specified below using argparse.
# Make sure that all these options are available so we can automatically test your code
# through command line arguments.

# You can check the TensorFlow API at
# https://www.tensorflow.org/programmers_guide/variables
# https://www.tensorflow.org/api_guides/python/contrib.layers#Initializers
WEIGHT_INITIALIZATION_DICT = {'xavier': tf.contrib.layers.xavier_initializer(), # Xavier initialisation
                              'normal': tf.truncated_normal_initializer, # Initialization from a standard normal
                              'uniform': tf.random_uniform_initializer, # Initialization from a uniform distribution
                             }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/contrib.layers#Regularizers
WEIGHT_REGULARIZER_DICT = {'none': None, # No regularization
                           'l1': tf.contrib.layers.l1_regularizer(0.001), # L1 regularization
                           'l2': tf.contrib.layers.l2_regularizer(0.001) # L2 regularization
                          }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/nn
ACTIVATION_DICT = {'relu': tf.nn.relu, # ReLU
                   'elu': tf.nn.elu, # ELU
                   'tanh': tf.tanh, #Tanh
                   'sigmoid': tf.sigmoid} #Sigmoid

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/train
OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer, # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer, # Adadelta
                  'adagrad': tf.train.AdagradOptimizer, # Adagrad
                  'adam': tf.train.AdamOptimizer, # Adam
                  'rmsprop': tf.train.RMSPropOptimizer # RMSprop
                  }

FLAGS = None

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def train():
  """
  Performs training and evaluation of MLP model. Evaluate your model each 100 iterations
  as you did in the task 1 of this assignment. 
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  tf.set_random_seed(42)
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
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
    # Initialize MLP instance
    is_training = tf.Variable([True], tf.bool)
    MLP    = mlp_tf.MLP(dnn_hidden_units, n_classes, is_training, input_dimensions, FLAGS.weight_init_scale, FLAGS.weight_reg_strength,
                        ACTIVATION_DICT[FLAGS.activation], FLAGS.dropout_rate,
                        WEIGHT_INITIALIZATION_DICT[FLAGS.weight_init],
                        WEIGHT_REGULARIZER_DICT[FLAGS.weight_reg], OPTIMIZER_DICT[FLAGS.optimizer])

    # Generate placeholders for the images and labels.
    images_placeholder = tf.placeholder(tf.float32, shape=(None, input_dimensions))
    labels_placeholder = tf.placeholder(tf.int32, shape=(None, n_classes))

    # Build a Graph that computes predictions from the inference model.
    logits = MLP.inference(images_placeholder)

    # Add to the Graph the Ops for loss calculation.
    loss   = MLP.loss(logits,labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_step = MLP.train_step(loss, FLAGS)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = MLP.accuracy(logits, labels_placeholder)

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
      _, loss_value, [accuracy_value, _] = sess.run([train_step, loss, eval_correct], feed_dict=feed_dict)
      x, y = cifar10.train.next_batch(FLAGS.batch_size)
      x = x.reshape(FLAGS.batch_size, -1)
      if step % 100 == 0:
        duration = time.time() - start_time
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
      if (step + 1) == FLAGS.max_steps:
        # Evaluate against the test set.
        print('Test Data Eval:')
        '''do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                cifar10.test)'''
        images_feed, labels_feed = cifar10.test.images, cifar10.test.labels
        images_feed = images_feed.reshape(cifar10.test.num_examples, -1)
        feed_dict = {
          images_placeholder: images_feed,
          labels_placeholder: labels_feed,
        }
        accuracy, y_pred = sess.run(eval_correct, feed_dict=feed_dict)
        #true_count = np.sum(accuracy)
        num_examples = cifar10.test.num_examples
        true_count = round(accuracy * num_examples)
        #precision = float(true_count) / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, accuracy))

        # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        # Compute confusion matrix
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(cifar10.test.labels, axis=1)
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        import pickle
        with open(FLAGS.data_dir + '/batches.meta', 'rb') as fo:
          label_names = pickle.load(fo, encoding='bytes')
          label_names = label_names[b'label_names']
          label_names = [name.decode('UTF-8') for name in label_names]
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=label_names,
                      title='Confusion matrix, without normalization')
        plt.show()


  #raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main(_):
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  # Make directories if they do not exists yet
  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--weight_init', type = str, default = WEIGHT_INITIALIZATION_DEFAULT,
                      help='Weight initialization type [xavier, normal, uniform].')
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  parser.add_argument('--weight_reg', type = str, default = WEIGHT_REGULARIZER_DEFAULT,
                      help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')
  parser.add_argument('--dropout_rate', type = float, default = DROPOUT_RATE_DEFAULT,
                      help='Dropout rate.')
  parser.add_argument('--activation', type = str, default = ACTIVATION_DEFAULT,
                      help='Activation function [relu, elu, tanh, sigmoid].')
  parser.add_argument('--optimizer', type = str, default = OPTIMIZER_DEFAULT,
                      help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
