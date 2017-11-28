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

class TextGenerationModel(object):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden, lstm_num_layers):

        self._seq_length = seq_length
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._vocab_size = vocabulary_size

        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)
        # Initialization:
        self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_num_hidden) for _ in range(lstm_num_layers)])
        self.state        = self.stacked_lstm.zero_state(batch_size, tf.float32)
        self.Wout         = tf.get_variable(shape=[lstm_num_hidden, vocabulary_size], initializer=initializer_weights, name='Wout')
        self.bias_out     = tf.get_variable(shape=[vocabulary_size], initializer=initializer_biases, name='bias_out')

    def _build_model(self, x):
        # Implement your model to return the logits per step of shape:
        #   [timesteps, batch_size, vocab_size]
        logits_per_step = []
        for step in range(x.shape[0]):
            output, self.state = self.stacked_lstm(x[step], self.state)
            logits = tf.matmul(output, self.Wout) + self.bias_out
            logits_per_step.append(logits)
        return logits_per_step

    def _compute_loss(self, logits_per_step, labels_per_step):
        # Cross-entropy loss, averaged over timestep and batch
        logits_reshaped = tf.reshape(logits_per_step, [-1])
        labels_reshaped = tf.reshape(labels_per_step, [-1])
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                            labels=labels_reshaped, logits=logits_reshaped, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def probabilities(self, logits_per_step):
        # Returns the normalized per-step probabilities
        probabilities = []
        for logits in logits_per_step:
            probabilities.append(tf.nn.softmax(logits))
        return probabilities

    def predictions(self, probabilities):
        # Returns the per-step predictions
        predictions = tf.argmax(tf.argmax(probabilities,0),1)
        return predictions