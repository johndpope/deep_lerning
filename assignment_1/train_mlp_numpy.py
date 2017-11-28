"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import mlp_numpy
import cifar10_utils

# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DNN_HIDDEN_UNITS_DEFAULT = '100'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def train():
  """
  Performs training and evaluation of MLP model. Evaluate your model on the whole test set each 100 iterations.
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
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
  batch_size = FLAGS.batch_size
  # Get cifar data
  print('Get cifar data')
  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
  x, y = cifar10.train.next_batch(batch_size)
  #x = x.flatten()
  x = x.reshape(batch_size, -1)
  input_dimensions = x.shape[1]
  n_classes = y.shape[1]
  print('Data obtained')

  # Initialize MLP instance
  MLP = mlp_numpy.MLP(dnn_hidden_units, n_classes, input_dimensions, batch_size, FLAGS.weight_reg_strength, FLAGS.weight_init_scale)

  # Train
  #FLAGS.max_steps = 7
  for step in range(FLAGS.max_steps):
    logits = MLP.inference(x)
    #print(logits)
    loss, logits   = MLP.loss(logits, y)
    #print(logits)
    MLP.train_step(loss, FLAGS, logits, y)
    
    if step%10 == 0:
      # Calculate accuracy
      print(str(step) + ": " + str(MLP.accuracy(logits,y)) + ",\tloss: " + str(loss))

    x, y = cifar10.train.next_batch(batch_size)
    x = x.reshape(batch_size, -1)
    
    if (step + 1) == FLAGS.max_steps:
      # Load test data
      test_x, test_y = cifar10.test.images, cifar10.test.labels
      test_x = test_x.reshape(test_x.shape[0], -1)
      logits = MLP.inference(test_x)
      print("Accuracy on test data: " + str(MLP.accuracy(logits,test_y)))

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

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

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
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
