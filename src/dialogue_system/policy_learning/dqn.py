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


class LSTM(object):
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


class DQN(object):
    def __init__(self, input_size, hidden_size, output_size, checkpoint_path):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.checkpoint_path = checkpoint_path
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder(dtype=tf.float64,shape=(None, self.input_size), name="input")
            self.target_value = tf.placeholder(dtype=tf.float64, shape=(None, None), name="target_value")
            # Target network
            with tf.variable_scope(name_or_scope="target_network"):
                self.target_weights = tf.get_variable(name="weights", shape=(self.input_size, output_size),dtype=tf.float64)
                self.target_bias = tf.get_variable(name="bias", shape=(self.output_size),dtype=tf.float64)
                self.target_output = tf.nn.relu(tf.add(tf.matmul(self.input, self.target_weights),self.target_bias), name="target_output")
            # Current network.
            with tf.variable_scope(name_or_scope="current_network"):
                self.current_weights = tf.get_variable(name="weights", shape=(self.input_size, output_size),dtype=tf.float64)
                self.current_bias = tf.get_variable(name="bias", shape=(self.output_size),dtype=tf.float64)
                self.current_output = tf.nn.relu(tf.add(tf.matmul(self.input,self.current_weights), self.current_bias), name="current_output")
            self.update_target_weights = tf.assign(self.target_weights, self.current_weights.value())
            self.update_target_bias = tf.assign(self.target_bias, self.current_bias.value())

            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.target_value - self.current_output)))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=self.loss)
            self.initializer = tf.global_variables_initializer()
            self.model_saver = tf.train.Saver()
        self.session = tf.Session(graph=self.graph)
        self.graph.finalize()
        self.session.run(self.initializer)

    def singleBatch(self, batch, params):
        # state, agent_action, reward, next_state, episode_over
        gamma = params.get('gamma', 0.9)
        Xs = []
        next_Xs = []
        for i,x in enumerate(batch):
            state_pre = x[0]
            next_state_pre = x[3]
            Xs.append(state_pre)
            next_Xs.append(next_state_pre)
        next_Ys = self._predict_target(Xs=next_Xs,params=params)[0]
        Ys_label = self.predict(Xs=Xs,params=params)[0] # numpy.ndarray

        for i, x in enumerate(batch):
            reward = x[2] # int
            episode_over = x[4] # bool
            action = x[1] # int
            next_max_y = np.max(next_Ys[i])
            target_y = reward
            if not episode_over:
                target_y = float(gamma * next_max_y + reward)
            Ys_label[i][action] = target_y

        feed_dict = {self.input:Xs, self.target_value:Ys_label}
        loss = self.session.run(self.loss,feed_dict=feed_dict) # For return
        self.session.run(self.optimizer, feed_dict=feed_dict)
        return {"loss": loss}

    def predict(self, Xs, **kwargs):
        feed_dict = {self.input:Xs}
        Ys = self.session.run(self.current_output,feed_dict=feed_dict)
        max_index = np.argmax(Ys, axis=1)
        return Ys, max_index[0]

    def _predict_target(self, Xs, params, **kwargs):
        feed_dict ={self.input:Xs}
        Ys = self.session.run(self.target_output,feed_dict=feed_dict)
        max_index = np.argmax(Ys, axis=1)
        return Ys, max_index[0]

    def update_target_network(self):
        self.session.run(self.update_target_weights)
        self.session.run(self.update_target_bias)

    def save_model(self, model_performance):
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        model_file_name = "model_s" + str(success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn) + ".ckpt"
        self.model_saver.save(sess=self.session,save_path=self.checkpoint_path + model_file_name)