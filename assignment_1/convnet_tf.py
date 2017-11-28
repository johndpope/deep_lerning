"""
This module implements a convolutional neural network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class ConvNet(object):
  """
  This class implements a convolutional neural network in TensorFlow.
  It incorporates a certain graph model to be trained and to be used
  in inference.
  """

  def __init__(self, optimizer, is_training, n_classes = 10):
    """
    Constructor for an ConvNet object. Default values should be used as hints for
    the usage of each parameter.
    Args:
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the ConvNet.
    """
    self.n_classes = n_classes
    self.optimizer = optimizer
    self.is_training = is_training

  def dense_batch_relu(self, x, units, scope):
    with tf.variable_scope(scope):
      h1 = tf.contrib.layers.fully_connected(x, units, 
                                             activation_fn=None,
                                             scope='dense')
      h2 = tf.contrib.layers.batch_norm(h1, 
                                        center=True, scale=True, 
                                        is_training=self.is_training,
                                        scope='bn')
      return tf.nn.relu(h2, 'relu')

  def inference(self, x):
    """
    Performs inference given an input tensor. This is the central portion
    of the network where we describe the computation graph. Here an input
    tensor undergoes a series of convolution, pooling and nonlinear operations
    as defined in this method. For the details of the model, please
    see assignment file.

    Here we recommend you to consider using variable and name scopes in order
    to make your graph more intelligible for later references in TensorBoard
    and so on. You can define a name scope for the whole model or for each
    operator group (e.g. conv+pool+relu) individually to group them by name.
    Variable scopes are essential components in TensorFlow for parameter sharing.
    Although the model(s) which are within the scope of this class do not require
    parameter sharing it is a good practice to use variable scope to encapsulate
    model.

    Args:
      x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

    Returns:
      logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
              the logits outputs (before softmax transformation) of the
              network. These logits can then be used with loss and accuracy
              to evaluate the model.
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    # Input Layer
    input_layer = tf.reshape(x, [-1, 32, 32, 3])

    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    # Flatten
    pool2_flat = tf.reshape(pool2, [128, -1])

    # Dense Layer #1
    dense1 = tf.layers.batch_normalization(tf.layers.dense(inputs=pool2_flat, units=384, activation=tf.nn.relu))
    #dense1 = self.dense_batch_relu(pool2_flat, 384, 'dense1')

    # Dense Layer #2
    dense2 = tf.layers.batch_normalization(tf.layers.dense(inputs=dense1, units=192, activation=tf.nn.relu))
    #dense2 = self.dense_batch_relu(dense1, 192, 'dense2')

    # Ouput Layer
    logits = tf.layers.dense(inputs=dense2, units=10)

    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################
    return logits

  def loss(self, logits, labels):
    """
    Calculates the multiclass cross-entropy loss from the logits predictions and
    the ground truth labels. The function will also add the regularization
    loss from network weights to the total loss that is return.

    In order to implement this function you should have a look at
    tf.nn.softmax_cross_entropy_with_logits.
    
    You can use tf.summary.scalar to save scalar summaries of
    cross-entropy loss, regularization loss, and full loss (both summed)
    for use with TensorBoard. This will be useful for compiling your report.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   sample in the batch.

    Returns:
      loss: scalar float Tensor, full loss = cross_entropy + reg_loss
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################

    return loss

  def train_step(self, loss, flags):
    """
    Implements a training step using a parameters in flags.

    Args:
      loss: scalar float Tensor.
      flags: contains necessary parameters for optimization.
    Returns:
      train_step: TensorFlow operation to perform one training step
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    tf.summary.scalar('loss', loss)
    optimizer   = self.optimizer(flags.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step  = optimizer.minimize(loss, global_step=global_step)

    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return train_step

  def accuracy(self, logits, labels):
    """
    Calculate the prediction accuracy, i.e. the average correct predictions
    of the network.
    As in self.loss above, you can use tf.scalar_summary to save
    scalar summaries of accuracy for later use with the TensorBoard.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                 with one-hot encoding. Ground truth labels for
                 each sample in the batch.

    Returns:
      accuracy: scalar float Tensor, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    accuracy = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################

    return accuracy

