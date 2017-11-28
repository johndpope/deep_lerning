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

import os
import time
from datetime import datetime
import argparse

import numpy as np
import tensorflow as tf

from dataset import TextDataset
from model import TextGenerationModel

def one_hot(labels, num_classes):
    b = np.zeros((len(labels), num_classes))
    b[np.arange(len(labels)), labels] = 1
    return b

def get_batch(batch_size, seq_length, dataset):
    x,y = dataset.batch(batch_size, seq_length)
    x = np.transpose(x)
    #y = np.transpose(y)
    inputs  = np.zeros((x.shape[0], x.shape[1], dataset.vocab_size))
    #targets = np.zeros((y.shape[0], y.shape[1], dataset.vocab_size))
    for i in range(len(x)):
        inputs[i, :, :] = one_hot(x[i, :], dataset.vocab_size)
    #for i in range(len(y)):
    #    targets[i, :, :] = one_hot(y[i, :], dataset.vocab_size)
    return inputs,y

def train(config):

    # Initialize the text dataset
    dataset = TextDataset(config.txt_file)

    # Initialize the model
    model = TextGenerationModel(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        vocabulary_size=dataset.vocab_size,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers
    )

    ###########################################################################
    # Implement code here.
    ###########################################################################

    input_placeholder = tf.placeholder(tf.float32, [config.seq_length, config.batch_size, dataset.vocab_size])
    #label_placeholder = tf.placeholder(tf.float32, [config.seq_length, config.batch_size, dataset.vocab_size])
    #input_placeholder = tf.placeholder(tf.int32, [config.seq_length, config.batch_size])
    label_placeholder = tf.placeholder(tf.int32, [config.batch_size, config.seq_length])
    char_placeholder = tf.placeholder(tf.float32, [1,config.batch_size, dataset.vocab_size])

    # Transform to one hot
    #input_placeholder = tf.one_hot(input_placeholder, dataset.vocab_size)
    #label_placeholder = tf.one_hot(label_placeholder, dataset.vocab_size)

    # Compute logits
    logits_per_step = model._build_model(input_placeholder)

    # Define the optimizer
    optimizer = tf.train.RMSPropOptimizer(config.learning_rate)

    # Compute the gradients for each variable
    loss = model._compute_loss(logits_per_step, label_placeholder)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)#, global_step)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables))#, global_step=global_step)

    # Compute prediction of next character
    next_logits = model._build_model(char_placeholder)
    probabilities = model.probabilities(next_logits)
    predictions   = model.predictions(probabilities)

    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(config.summary_path, sess.graph)
    sess.run(init)
    ###########################################################################
    # Implement code here.
    ###########################################################################

    for train_step in range(int(config.train_steps)):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################################
        # Implement code here.
        #######################################################################

        # Load next sequence
        inputs, targets = get_batch(config.batch_size, config.seq_length, dataset)
        #inputs,targets = dataset.batch(config.batch_size, config.seq_length)
        #inputs = np.transpose(inputs)
        #targets = np.transpose(targets)
        feed_dict = {
            input_placeholder: inputs,
            label_placeholder: targets,
            char_placeholder:  np.zeros((1,config.batch_size,dataset.vocab_size))#np.expand_dims(one_hot([1], dataset.vocab_size),1)
        }
        _, loss_value = sess.run([apply_gradients_op, loss], feed_dict=feed_dict)

        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        # Output the training progress
        if train_step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {:.2f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), train_step+1,
                int(config.train_steps), config.batch_size, examples_per_second,
                loss_value
            ))
            # Update the events file.
            #summary_str = sess.run(summary, feed_dict=feed_dict)
            #summary_writer.add_summary(summary_str, train_step)
            #summary_writer.flush()

        if train_step % config.sample_every == 0:
            inputs  = np.zeros((config.seq_length, config.batch_size, dataset.vocab_size))
            targets = np.zeros((config.batch_size, config.seq_length))
            char    = [0]
            char_   = np.zeros((1, config.batch_size, dataset.vocab_size))
            final_string = dataset.convert_to_string(char)
            for _ in range(config.seq_length):
                for i in range(config.batch_size):
                    char_[:, i, :] = one_hot(char, dataset.vocab_size)
                feed_dict = {
                    input_placeholder: inputs,
                    label_placeholder: targets,
                    char_placeholder:  char_
                }
                predic = sess.run(predictions, feed_dict=feed_dict)
                char   = [predic[0]]
                final_string += dataset.convert_to_string(char)
                #print(predic.shape)
                #print(predic)
                #print(dataset.convert_to_string(predic))
            print('\n\n\n\n\n\n')
            print(final_string)

        if train_step % config.checkpoint_every == 0:
            saver.save(sess, save_path='./checkpoints/model.ckpt')



if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    #parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--txt_file', type=str, default='books/book_EN_grimms_fairy_tails.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5, help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    #parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--sample_every', type=int, default=200, help='How often to sample from the model')
    parser.add_argument('--checkpoint_every', type=int, default=500, help='How often to save the model')

    config = parser.parse_args()

    # Train the model
    train(config)