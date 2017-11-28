"""
This module implements a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from copy import deepcopy

def RELUderivative(M):
  N=deepcopy(M)
  N[N>0]  = 1
  N[N<=0] = 0
  return N

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform inference, training and it
  can also be used for evaluating prediction performance.
  """

  def __init__(self, n_hidden, n_classes, input_dimensions, batch_size, weight_decay=0.0, weight_scale=0.0001):
    """
    Constructor for an MLP object. Default values should be used as hints for
    the usage of each parameter. Weights of the linear layers should be initialized
    using normal distribution with mean = 0 and std = weight_scale. Biases should be
    initialized with constant 0. All activation functions are ReLUs.

    Args:
      n_hidden: list of ints, specifies the number of units
                     in each hidden layer. If the list is empty, the MLP
                     will not have any hidden units, and the model
                     will simply perform a multinomial logistic regression.
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the MLP.
      weight_decay: L2 regularization parameter for the weights of linear layers.
      weight_scale: scale of normal distribution to initialize weights.

    """
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.weight_decay = weight_decay
    self.weight_scale = weight_scale
    self.weights = []
    self.bias    = []
    prev_size    = input_dimensions
    self.z_layer_results = [np.matrix(np.zeros((batch_size,prev_size)), dtype='float64')]
    for size in n_hidden:
      # Initialize Bias and Hidden units weights
      hidd_weights = np.random.normal(0,weight_scale,(prev_size,size))
      bias_weights = np.zeros((1,size))
      self.weights.append(np.matrix(hidd_weights, dtype='float64'))
      self.bias.append(np.matrix(bias_weights, dtype='float64'))
      self.z_layer_results.append(np.matrix(np.zeros((batch_size,size)), dtype='float64'))
      prev_size = size
    hidd_weights = np.random.normal(0,weight_scale,(prev_size,n_classes))
    bias_weights = np.zeros((1,n_classes))
    self.weights.append(np.matrix(hidd_weights, dtype='float64'))
    self.bias.append(np.matrix(bias_weights, dtype='float64'))

  def inference(self, x):
    """
    Performs inference given an input array. This is the central portion
    of the network. Here an input array is transformed through application
    of several hidden layer transformations (as defined in the constructor).
    We recommend you to iterate through the list self.n_hidden in order to
    perform the sequential transformations in the MLP. Do not forget to
    add a linear output layer (without non-linearity) as the last transformation.

    It can be useful to save some intermediate results for easier computation of
    gradients for backpropagation during training.
    Args:
      x: 2D float array of size [batch_size, input_dimensions]

    Returns:
      logits: 2D float array of size [batch_size, self.n_classes]. Returns
             the logits outputs (before softmax transformation) of the
             network. These logits can then be used with loss and accuracy
             to evaluate the model.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    logits = np.matrix(deepcopy(x), dtype='float64')
    self.z_layer_results[0] = logits

    # hidden layers
    for i,hidden_layer in enumerate(self.weights[:-1]):
      logits = np.dot(logits,hidden_layer)
      logits += self.bias[i]#Add Bias
      logits = np.maximum(0,logits)#RELU function
      self.z_layer_results[i+1] = logits
    
    # output layer
    logits = np.dot(logits,self.weights[-1])
    logits += self.bias[-1]#Add Bias
    
    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return logits

  def loss(self, logits, labels):
    """
    Computes the multiclass cross-entropy loss from the logits predictions and
    the ground truth labels. The function will also add the regularization
    loss from network weights to the total loss that is return.

    It can be useful to compute gradients of the loss for an easier computation of
    gradients for backpropagation during training.

    Args:
      logits: 2D float array of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int array of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   sample in the batch.
    Returns:
      loss: scalar float, full loss = cross_entropy + reg_loss
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # softmax exponential trick
    logits -= np.amax(logits, axis=1)
    logits = np.exp(logits)
    logits = logits/logits.sum(axis=1)

    # cross_entropy = -sum(p*log q)# p = true distribution# q = unnatural distribution
    true_classes  = np.where(labels==1)
    cross_entropy = -np.sum(np.log(logits[true_classes]))/logits.shape[0]

    # regularization loss
    reg_loss = 0.0
    if self.weight_decay != 0.0:
      reg_loss = (self.weight_decay/2)*np.sum([np.linalg.norm(hidden_layer)**2 for hidden_layer in self.weights])

    loss = cross_entropy + reg_loss

    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return loss, logits

  def train_step(self, loss, flags, logits, labels):
    """
    Implements a training step using a parameters in flags.
    Use mini-batch Gradient Descent to update the parameters of the MLP.

    Args:
      loss: scalar float.
      flags: contains necessary parameters for optimization.
    Returns:

    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    learning_rate        = flags.learning_rate
    delta                = logits-labels#-np.divide(1,logits,where=(logits!=0))#-1/logits
    # update hidden layers
    for i,hidden_layer_results in reversed(list(enumerate(self.z_layer_results))):
      if i != len(self.z_layer_results)-1:
        delta          = np.multiply(np.dot(delta,prev_weights.T),RELUderivative(self.z_layer_results[i+1]))
      dW               = np.dot(delta.T,hidden_layer_results)/logits.shape[0]
      prev_weights     = deepcopy(self.weights[i])
      self.weights[i] -= learning_rate*np.transpose(dW) + self.weight_decay*self.weights[i]
      self.bias[i]    -= learning_rate*np.sum(delta, axis=0)/logits.shape[0]# apply regularization in bias?

    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return

  def accuracy(self, logits, labels):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      logits: 2D float array of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int array of size [batch_size, self.n_classes]
                 with one-hot encoding. Ground truth labels for
                 each sample in the batch.
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    '''
    accuracy = 0.0
    for prediction, true_value in zip(logits, labels)
      if prediction.index(max(prediction)) == true_value.index(max(true_value)):
        accuracy += 1.0
    accuracy /= batch_size
    '''
    batch_size = logits.shape[0]
    prediction = np.argmax(logits, axis=1)
    true_value = np.argmax(np.matrix(labels), axis=1)
    accuracy = np.sum(prediction == true_value)/batch_size

    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy