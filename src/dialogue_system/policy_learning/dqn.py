# -*- coding:utf-8 -*-
"""
网络模型，input是一个sequence，首先经过一个LSTM层，然后在输入给DQN。
"""

import numpy as np
import tensorflow as tf
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/policy_learning",""))




class MediumConfig(object):
    """Medium config."""
    lstm_hidden_layer_size_lsit = [10, 10]
    init_scale = 0.05
    learning_rate = 1.0
    max_time_step = 40
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 1
    lr_decay = 0.8
    batch_size = 1


class DQN(object):
    def __init__(self):
        self.state_rep_list = []
        config = MediumConfig()
        self.config = config

        # self.inputs = tf.placeholder(tf.float64,shape=[None, None, self.config.lstm_hidden_layer_size_lsit[0]])
        self.inputs = tf.placeholder(tf.float64,shape=[None, None, 173])
        self.lstm_cell = self._build_lstm_layer()
        # self.initial_state = self.lstm_cell.zero_state(None,tf.float64)

        initializer = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(initializer)

    def _build_lstm_layer(self):
        with tf.variable_scope("lstm_layer"):
            # tf.get_variable_scope().reuse_variables()
            lstm_layers = [tf.nn.rnn_cell.LSTMCell(size, forget_bias=1.0,state_is_tuple=True,reuse=None) for size in self.config.lstm_hidden_layer_size_lsit]
            print(type(lstm_layers))
            # tf.nn.rnn_cell.LSTMCell()
            if self.config.keep_prob < 1 :
                for index in range(0,len(lstm_layers)):
                    lstm_layers[index] = tf.nn.rnn_cell.DropoutWrapper(lstm_layers[index],input_keep_prob=self.config.keep_prob)
            self.multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_layers, state_is_tuple=True)
            print("output_size:", lstm_layers[-1].output_size)
            lstm = tf.nn.dynamic_rnn(cell=self.multi_lstm_cell,inputs=self.inputs,dtype=tf.float64)
        return lstm

    def get_state(self, state_rep):

        outputs, state = self.session.run(self.lstm_cell, feed_dict={self.inputs:[state_rep]})
        outputs, state2 = self.session.run(self.lstm_cell, feed_dict={self.inputs:[state_rep]})
        print("state1:", state[-1][-1])
        print("state2:", state2[-1][-1])

tf.Session()