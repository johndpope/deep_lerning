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

import tensorflow as tf


class LSTM(object):

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
            self.Wgx = tf.get_variable(shape=[self._input_dim, self._num_hidden ], initializer=initializer_weights, name='Wgx')
            self.Wgh = tf.get_variable(shape=[self._num_hidden, self._num_hidden], initializer=initializer_weights, name='Wgh')
            self.Wix = tf.get_variable(shape=[self._input_dim, self._num_hidden ], initializer=initializer_weights, name='Wix')
            self.Wih = tf.get_variable(shape=[self._num_hidden, self._num_hidden], initializer=initializer_weights, name='Wih')
            self.Wfx = tf.get_variable(shape=[self._input_dim, self._num_hidden ], initializer=initializer_weights, name='Wfx')
            self.Wfh = tf.get_variable(shape=[self._num_hidden, self._num_hidden], initializer=initializer_weights, name='Wfh')
            self.Wox = tf.get_variable(shape=[self._input_dim, self._num_hidden ], initializer=initializer_weights, name='Wox')
            self.Woh = tf.get_variable(shape=[self._num_hidden, self._num_hidden], initializer=initializer_weights, name='Woh')
            # output
            self.Wout = tf.get_variable(shape=[self._num_hidden, self._num_classes], initializer=initializer_weights, name='Wout')
        
        with tf.variable_scope('biases'):
            self.bias_g = tf.get_variable(shape=[self._num_hidden ], initializer=initializer_biases, name='bias_g')
            self.bias_i = tf.get_variable(shape=[self._num_hidden ], initializer=initializer_biases, name='bias_i')
            self.bias_f = tf.get_variable(shape=[self._num_hidden ], initializer=initializer_biases, name='bias_f')
            self.bias_o = tf.get_variable(shape=[self._num_hidden ], initializer=initializer_biases, name='bias_o')
            self.bias_out = tf.get_variable(shape=[self._num_classes], initializer=initializer_biases, name='bias_out')

    def _lstm_step(self, lstm_state_tuple, x):
        # Single step through LSTM cell ...
        c_prev,h_prev = tf.unstack(lstm_state_tuple)
        g = tf.tanh(tf.matmul(x, self.Wgx) + tf.matmul(h_prev, self.Wgh) + self.bias_g )
        i = tf.sigmoid(tf.matmul(x, self.Wix) + tf.matmul(h_prev, self.Wih) + self.bias_i )
        f = tf.sigmoid(tf.matmul(x, self.Wfx) + tf.matmul(h_prev, self.Wfh) + self.bias_f )
        o = tf.sigmoid(tf.matmul(x, self.Wox) + tf.matmul(h_prev, self.Woh) + self.bias_o )
        c = tf.multiply(g,i) + tf.multiply(c_prev,f)
        h = tf.multiply(tf.tanh(c),o)
        return tf.stack([c,h])

    def compute_logits(self, x):
        # Implement the logits for predicting the last digit in the palindrome
        h_init = tf.zeros([self._batch_size, self._num_hidden], name='h_init')
        c_init = tf.zeros([self._batch_size, self._num_hidden], name='c_init')
        init_state = tf.stack([c_init,h_init])
        states = tf.scan(self._lstm_step, x, initializer=init_state)
        _,h = tf.unstack(states[-1])
        p = tf.matmul(h, self.Wout) + self.bias_out
        logits = p
        return logits

    def compute_logits_test(self, x, test_size):
        # Implement the logits for predicting the last digit in the palindrome
        h_init = tf.zeros([test_size, self._num_hidden], name='h_init')
        c_init = tf.zeros([test_size, self._num_hidden], name='c_init')
        init_state = tf.stack([c_init,h_init])
        states = tf.scan(self._lstm_step, x, initializer=init_state)
        _,h = tf.unstack(states[-1])
        p = tf.matmul(h, self.Wout) + self.bias_out
        logits = p
        return logits

    def compute_loss(self, logits, labels):
        # Implement the cross-entropy loss for classification of the last digit
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.summary.scalar('loss', loss)
        return loss

    def accuracy(self, logits, labels):
        # Implement the accuracy of predicting the
        # last digit over the current batch ...
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), tf.argmax(labels,1)), 'float32'))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy