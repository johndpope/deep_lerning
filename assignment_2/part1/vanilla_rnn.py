# MIT License
# 
# Copyright (c) 2017 Tom Runia
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2017-10-19

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

################################################################################

class VanillaRNN(object):

    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):

        self._input_length = input_length
        self._input_dim    = input_dim
        self._num_hidden   = num_hidden
        self._num_classes  = num_classes
        self._batch_size   = batch_size

        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)

        # Initialize the stuff you need
        with tf.variable_scope('weights'):
            # input-to-hidden
            self.Whx = tf.get_variable(shape=[self._input_dim, self._num_hidden], initializer=initializer_weights, name='Whx')
            # hidden-to-hidden
            self.Whh = tf.get_variable(shape=[self._num_hidden, self._num_hidden], initializer=initializer_weights, name='Whh')
            # hidden-to-output
            self.Woh = tf.get_variable(shape=[self._num_hidden, self._num_classes], initializer=initializer_weights, name='Woh')
        
        with tf.variable_scope('biases'):
            self.bias_h = tf.get_variable(shape=[self._num_hidden ], initializer=initializer_biases, name='bias_h')
            self.bias_o = tf.get_variable(shape=[self._num_classes], initializer=initializer_biases, name='bias_o')

        #self.h = tf.get_variable(shape=[self._num_hidden ], initializer=initializer_biases, name='h')
        #self.p = tf.get_variable(shape=[self._num_classes], initializer=initializer_biases, name='p')

    def _rnn_step(self, h_prev, x):
        # Single step through Vanilla RNN cell ...
        #raise NotImplementedError()
        aux1 = tf.matmul(x, self.Whx)
        aux2 = tf.matmul(h_prev, self.Whh)
        h = tf.tanh(aux1 + aux2 + self.bias_h )
        #h = tf.tanh(tf.matmul(self.Whx, tf.transpose(x)) + tf.matmul(self.Whh, h_prev) + self.bias_h )
        return h

    def compute_logits(self, x):
        # Implement the logits for predicting the last digit in the palindrome
        init_state = tf.zeros([self._batch_size, self._num_hidden], name= 'init_state')
        states = tf.scan(self._rnn_step, x, initializer=init_state)
        h = states[-1]
        p = tf.matmul(h, self.Woh) + self.bias_o
        logits = p
        return logits

    def compute_logits_test(self, x, test_size):
        # Implement the logits for predicting the last digit in the palindrome
        init_state = tf.zeros([test_size, self._num_hidden], name= 'init_state_test')
        states = tf.scan(self._rnn_step, x, initializer=init_state)
        h = states[-1]
        p = tf.matmul(h, self.Woh) + self.bias_o
        logits = p
        return logits

    def compute_loss(self, logits, labels):
        # Implement the cross-entropy loss for classification of the last digit
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        #tf.summary.scalar('loss', loss)
        return loss

    def accuracy(self, logits, labels):
        # Implement the accuracy of predicting the
        # last digit over the current batch ...
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), tf.argmax(labels,1)), 'float32'))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy
#CCHECK: HOW TO COMPUTE LOSS? HOW TO UPDATE GRADIENTS?