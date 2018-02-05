# -*- coding:utf-8 -*-
"""
网络模型，input是一个sequence，首先经过一个LSTM层，然后在输入给DQN。
"""

import numpy as np
import math
import copy
import pickle
import tensorflow as tf
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/policy_learning",""))


class DQN0(object):
    """
    Initial DQN written by Qianlong, one layer.
    """
    def __init__(self, input_size, hidden_size, output_size, parameter):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.checkpoint_path = parameter.get("checkpoint_path")
        self.log_dir = parameter.get("log_dir")
        self.parameter = parameter
        self.learning_rate = parameter.get("dqn_learning_rate")
        device = parameter.get("device_for_tf")
        with tf.device(device):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.input = tf.placeholder(dtype=tf.float64, shape=(None, self.input_size), name="input")
                self.target_value = tf.placeholder(dtype=tf.float64, shape=(None, None), name="target_value")
                # Target network
                with tf.variable_scope(name_or_scope="target_network"):
                    self.target_weights = tf.get_variable(name="weights", shape=(self.input_size, output_size),dtype=tf.float64)
                    self.target_bias = tf.get_variable(name="bias", shape=(self.output_size), dtype=tf.float64)
                    self.target_output = tf.nn.relu(tf.add(tf.matmul(self.input, self.target_weights), self.target_bias), name="target_output")

                    tf.summary.scalar("target_weight", self.target_weights)
                    tf.summary.scalar("target_bias", self.target_bias)
                # Current network.
                with tf.variable_scope(name_or_scope="current_network"):
                    self.current_weights = tf.get_variable(name="weights", shape=(self.input_size, output_size),dtype=tf.float64)
                    self.current_bias = tf.get_variable(name="bias", shape=(self.output_size), dtype=tf.float64)
                    self.current_output = tf.nn.relu(tf.add(tf.matmul(self.input, self.current_weights), self.current_bias), name="current_output")

                    tf.summary.scalar("current_weight", self.current_weights)
                    tf.summary.scalar("current_bias", self.current_bias)
                # Updating target network.

                self.update_target_weights = tf.assign(self.target_weights, self.current_weights.value())
                self.update_target_bias = tf.assign(self.target_bias, self.current_bias.value())

                # Optimization.
                self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.target_value - self.current_output),axis=1),name="loss")
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss)
                self.initializer = tf.global_variables_initializer()
                self.model_saver = tf.train.Saver()
            self.graph.finalize()

        # Visualizing learning.
        merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(logdir=self.log_dir + "train", graph=self.graph)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.session = tf.Session(graph=self.graph,config=config)
        self.session.run(self.initializer)

        if self.parameter.get("train_mode") != 1:
            self.restore_model(self.parameter.get("saved_model"))

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

    def save_model(self, model_performance,episodes_index, checkpoint_path = None):
        if checkpoint_path == None: checkpoint_path = self.checkpoint_path
        agent_id = self.parameter.get("agent_id")
        dqn_id = self.parameter.get("dqn_id")
        disease_number = self.parameter.get("disease_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_wrong_disease = model_performance["average_wrong_disease"]
        model_file_name = "model_d" + str(disease_number) + "_agent" + str(agent_id) + "_dqn" + str(dqn_id) + "_s" + str(success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn) + "_wd" + str(average_wrong_disease) + "_e" + str(episodes_index) + ".ckpt"
        self.model_saver.save(sess=self.session,save_path=checkpoint_path + model_file_name)

    def restore_model(self, saved_model):
        print("loading trained model")
        self.model_saver.restore(sess=self.session,save_path=saved_model)


class DQN1(DQN0):
    """
    One layer. Written by Qianlong with tensorflow.
    """
    def __init__(self, input_size, hidden_size, output_size, parameter):
        super(DQN1, self).__init__(input_size,hidden_size,output_size,parameter)
        self.update_target_network_operations = []
        self.__build_model()

    def __build_model(self):
        device = self.parameter.get("device_for_tf")
        with tf.device(device):
            self.graph = tf.Graph()

            with self.graph.as_default():
                self.input = tf.placeholder(dtype=tf.float64, shape=(None, self.input_size), name="input")
                self.target_value = tf.placeholder(dtype=tf.float64, shape=(None, self.output_size), name="target_value")
                # Target network
                self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

                with tf.variable_scope(name_or_scope="target_network"):
                    self.target_network_variables = {}
                    self.target_output = self._build_layer(variables_dict=self.target_network_variables,
                                                       input=self.input,
                                                       input_size=self.input_size,
                                                       output_size=self.output_size,
                                                       weights_key="w1",
                                                       bias_key="b1")

                # Current network.
                with tf.variable_scope(name_or_scope="current_network"):
                    self.current_network_variables = {}
                    self.current_output = self._build_layer(variables_dict=self.current_network_variables,
                                                       input=self.input,
                                                       input_size=self.input_size,
                                                       output_size=self.output_size,
                                                       weights_key="w1",
                                                       bias_key="b1")

                    # Regularization.
                    for key, value in self.current_network_variables.items():
                        if "w" in key: tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value)
                # Updating target network.
                with tf.name_scope(name="ops_of_updating_target_network"):
                    self.update_target_network_operations = self._update_target_network_operations()

                # Optimization.
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.reg_loss = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
                self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.target_value - self.current_output),axis=1),name="loss") \
                            + self.reg_loss

                tf.summary.scalar("loss", self.loss)
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss= self.loss)
                self.initializer = tf.global_variables_initializer()
                self.model_saver = tf.train.Saver()
                # self.merged_summary = tf.summary.merge_all()
            self.graph.finalize()

        # Visualizing graph.
        self.summary_writer = tf.summary.FileWriter(logdir=self.log_dir + "train", graph=self.graph)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.session = tf.Session(graph=self.graph,config=config)
        self.session.run(self.initializer)

        if self.parameter.get("train_mode") != 1:
            self.restore_model(self.parameter.get("saved_model"))

    def update_target_network(self):
        self.session.run(fetches=self.update_target_network_operations)

    def _update_target_network_operations(self):
        update_target_network_operations = []
        for key in self.current_network_variables.keys():
            update = tf.assign(ref=self.target_network_variables[key],value=self.current_network_variables[key].value())
            update_target_network_operations.append(update)
        return update_target_network_operations

    def _build_layer(self, variables_dict, input, input_size, output_size, weights_key, bias_key):
        weights = tf.get_variable(name=weights_key, shape=(input_size, output_size), dtype=tf.float64)
        bias = tf.get_variable(name=bias_key, shape=(output_size), dtype=tf.float64)
        variables_dict[weights_key] = weights
        variables_dict[bias_key] = bias
        tf.summary.scalar(name=weights_key, tensor=weights)
        tf.summary.scalar(name=bias_key,tensor=bias)
        output = tf.nn.relu(tf.add(tf.matmul(input, weights), bias),name="output")
        return output


class DQN2(DQN1):
    """
    Two layers. Written by Qianlong with tensorflow.
    """
    def __init__(self, input_size, hidden_size, output_size, parameter):
        super(DQN2, self).__init__(input_size,hidden_size,output_size,parameter)
        self.__build_model()

    def __build_model(self):
        device = self.parameter.get("device_for_tf")
        with tf.device(device):
            self.graph = tf.Graph()

            with self.graph.as_default():
                self.input = tf.placeholder(dtype=tf.float64, shape=(None, self.input_size), name="input")
                self.target_value = tf.placeholder(dtype=tf.float64, shape=(None, self.output_size),
                                                   name="target_value")
                # Target network
                # from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
                self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

                with tf.variable_scope(name_or_scope="target_network"):
                    self.target_network_variables = {}
                    target_hidden_output = self._build_layer(variables_dict=self.target_network_variables,
                                                              input=self.input,
                                                              input_size=self.input_size,
                                                              output_size=self.hidden_size,
                                                              weights_key="w1",
                                                              bias_key="b1")

                    self.target_output = self._build_layer(variables_dict=self.target_network_variables,
                                                            input=target_hidden_output,
                                                            input_size=self.hidden_size,
                                                            output_size=self.output_size,
                                                            weights_key="w2",
                                                            bias_key="b2")

                # Current network.
                with tf.variable_scope(name_or_scope="current_network"):
                    self.current_network_variables = {}
                    current_hidden_output = self._build_layer(variables_dict=self.current_network_variables,
                                                               input=self.input,
                                                               input_size=self.input_size,
                                                               output_size=self.hidden_size,
                                                               weights_key="w1",
                                                               bias_key="b1")

                    self.current_output = self._build_layer(variables_dict=self.current_network_variables,
                                                             input=current_hidden_output,
                                                             input_size=self.hidden_size,
                                                             output_size=self.output_size,
                                                             weights_key="w2",
                                                             bias_key="b2")

                    # Regularization.
                    for key, value in self.current_network_variables.items():
                        if "w" in key: tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value)
                # Updating target network.
                with tf.name_scope(name="ops_of_updating_target_network"):
                    self.update_target_network_operations = self._update_target_network_operations()

                # Optimization.
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.reg_loss = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
                self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.target_value - self.current_output), axis=1),
                                           name="loss") + self.reg_loss

                tf.summary.scalar("loss", self.loss)
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss)
                self.initializer = tf.global_variables_initializer()
                self.model_saver = tf.train.Saver()
                # self.merged_summary = tf.summary.merge_all()
            self.graph.finalize()

        # Visualizing graph.
        self.summary_writer = tf.summary.FileWriter(logdir=self.log_dir + "train", graph=self.graph)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.session = tf.Session(graph=self.graph, config=config)
        self.session.run(self.initializer)

        if self.parameter.get("train_mode") != 1:
            self.restore_model(self.parameter.get("saved_model"))


class DQN3(object):
    """
    Written by Boalin without tensorflow. Two layers.
    """
    def __init__(self, input_size, hidden_size, output_size,parameter):
        self.parameter = parameter
        self.checkpoint_path = parameter.get("checkpoint_path")
        self.model = {}
        # input-hidden
        self.model['Wxh'] = self.__initWeight(input_size, hidden_size)
        self.model['bxh'] = np.zeros((1, hidden_size))

        # hidden-output
        self.model['Wd'] = self.__initWeight(hidden_size, output_size) * 0.1
        self.model['bd'] = np.zeros((1, output_size))

        self.update = ['Wxh', 'bxh', 'Wd', 'bd']
        self.regularize = ['Wxh', 'Wd']
        self.step_cache = {}
        self.clone_dqn = copy.deepcopy(self)

    def getStruct(self):
        return {'model': self.model, 'update': self.update, 'regularize': self.regularize}

    """Activation Function: Sigmoid, or tanh, or ReLu"""

    def fwdPass(self, Xs, params, **kwargs):
        """

        :param Xs: one sample with the shape of (1, input_size)
        :param params:
        :param kwargs:
        :return:
        """
        predict_mode = kwargs.get('predict_mode', False)
        active_func = params.get('activation_func', 'relu')

        # input layer to hidden layer
        Wxh = self.model['Wxh']
        bxh = self.model['bxh']
        Xsh = Xs.dot(Wxh) + bxh

        hidden_size = self.model['Wd'].shape[0]  # size of hidden layer
        H = np.zeros((1, hidden_size))  # hidden layer representation

        if active_func.lower() == 'sigmoid':
            H = 1 / (1 + np.exp(-Xsh))
        elif active_func.lower() == 'tanh':
            H = np.tanh(Xsh)
        elif active_func.lower() == 'relu':  # ReLU
            H = np.maximum(Xsh, 0)
        else:  # no activation function
            H = Xsh

        # decoder at the end; hidden layer to output layer
        Wd = self.model['Wd']
        bd = self.model['bd']
        Y = H.dot(Wd) + bd

        # cache the values in forward pass, we expect to do a backward pass
        cache = {}
        if not predict_mode:
            cache['Wxh'] = Wxh
            cache['Wd'] = Wd
            cache['Xs'] = Xs
            cache['Xsh'] = Xsh
            cache['H'] = H

            cache['bxh'] = bxh
            cache['bd'] = bd
            cache['activation_func'] = active_func

            cache['Y'] = Y

        return Y, cache

    def bwdPass(self, dY, cache):
        Wd = cache['Wd']
        H = cache['H']
        Xs = cache['Xs']
        Xsh = cache['Xsh']
        Wxh = cache['Wxh']

        active_func = cache['activation_func']
        n, d = H.shape

        dH = dY.dot(Wd.transpose())
        # backprop the decoder
        dWd = H.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims=True)

        dXsh = np.zeros(Xsh.shape)
        dXs = np.zeros(Xs.shape)

        if active_func == 'sigmoid':
            dH = (H - H ** 2) * dH
        elif active_func == 'tanh':
            dH = (1 - H ** 2) * dH
        elif active_func == 'relu':
            dH = (H > 0) * dH  # backprop ReLU
        else:
            dH = dH

        # backprop to the input-hidden connection
        dWxh = Xs.transpose().dot(dH)
        dbxh = np.sum(dH, axis=0, keepdims=True)

        # backprop to the input
        dXsh = dH
        dXs = dXsh.dot(Wxh.transpose())

        return {'Wd': dWd, 'bd': dbd, 'Wxh': dWxh, 'bxh': dbxh}

    """batch Forward & Backward Pass"""

    def batchForward(self, batch, params, predict_mode=False):
        caches = []
        Ys = []
        for i, x in enumerate(batch):
            Xs = np.array([x['cur_states']], dtype=float)

            Y, out_cache = self.fwdPass(Xs, params, predict_mode=predict_mode)
            caches.append(out_cache)
            Ys.append(Y)

        # back up information for efficient backprop
        cache = {}
        if not predict_mode:
            cache['caches'] = caches

        return Ys, cache

    def batchDoubleForward(self, batch, params, predict_mode=False):
        caches = []
        Ys = []
        tYs = []

        for i, x in enumerate(batch):
            Xs = x[0]

            Y, out_cache = self.fwdPass(Xs, params, predict_mode=predict_mode)
            caches.append(out_cache)
            Ys.append(Y)

            tXs = x[3]
            tY, t_cache = self.clone_dqn.fwdPass(tXs, params, predict_mode=False)

            tYs.append(tY)

        # back up information for efficient backprop
        cache = {}
        if not predict_mode:
            cache['caches'] = caches

        return Ys, cache, tYs

    def batchBackward(self, dY, cache):
        caches = cache['caches']

        grads = {}
        for i in range(len(caches)):
            single_cache = caches[i]
            local_grads = self.bwdPass(dY[i], single_cache)
            self.__mergeDicts(grads, local_grads)  # add up the gradients wrt model parameters

        return grads

    """ cost function, returns cost and gradients for model """

    def costFunc(self, batch, params):
        regc = params.get('reg_cost', 1e-3)
        gamma = params.get('gamma', 0.9)

        # batch forward
        Ys, caches, tYs = self.batchDoubleForward(batch, params, predict_mode=False)

        loss_cost = 0.0
        dYs = []
        for i, x in enumerate(batch):
            Y = Ys[i]
            nY = tYs[i]

            action = np.array(x[1], dtype=int)
            reward = np.array(x[2], dtype=float)  # 这一动作获得reward

            n_action = np.nanargmax(nY[0])  # 预测的Q中最大的一个action
            max_next_y = nY[0][n_action]  # 对应的最大的Q_value

            eposide_terminate = x[4]  # 这一动作之后游戏是否结束

            target_y = reward
            if eposide_terminate != True: target_y += gamma * max_next_y

            pred_y = Y[0][action]

            nY = np.zeros(nY.shape)
            nY[0][action] = target_y
            Y = np.zeros(Y.shape)
            Y[0][action] = pred_y

            # Cost Function
            loss_cost += (target_y - pred_y) ** 2

            dY = -(nY - Y)
            # dY = np.minimum(dY, 1)
            # dY = np.maximum(dY, -1)
            dYs.append(dY)

        # backprop the RNN
        grads = self.batchBackward(dYs, caches)

        # add L2 regularization cost and gradients
        reg_cost = 0.0
        if regc > 0:
            for p in self.regularize:
                mat = self.model[p]
                reg_cost += 0.5 * regc * np.sum(mat * mat)
                grads[p] += regc * mat

        # normalize the cost and gradient by the batch size
        batch_size = len(batch)
        reg_cost /= batch_size
        loss_cost /= batch_size
        for k in grads: grads[k] /= batch_size

        out = {}
        out['cost'] = {'reg_cost': reg_cost, 'loss_cost': loss_cost, 'total_cost': loss_cost + reg_cost}
        out['grads'] = grads
        return out

    """ A single batch """

    def singleBatch(self, batch, params):
        # state, agent_action, reward, next_state, episode_over
        learning_rate = params.get('learning_rate', 0.001)
        decay_rate = params.get('decay_rate', 0.999)
        momentum = params.get('momentum', 0.1)
        grad_clip = params.get('grad_clip', -1e-3)
        smooth_eps = params.get('smooth_eps', 1e-8)
        sdg_type = params.get('sdgtype', 'rmsprop')
        activation_func = params.get('activation_func', 'relu')
        temp_batch = []
        for sample in batch:
            state = np.array([sample[0]])
            action = sample[1]
            reward = sample[2]
            next_state = np.array([sample[3]])
            episode_over = sample[4]
            temp_batch.append((state, action, reward, next_state, episode_over))
        batch = copy.deepcopy(temp_batch)


        for u in self.update:
            if not u in self.step_cache:
                self.step_cache[u] = np.zeros(self.model[u].shape)

        cg = self.costFunc(batch, params)

        # cost = cg['cost']
        cost = cg["cost"]["total_cost"]
        grads = cg['grads']

        # clip gradients if needed
        if activation_func.lower() == 'relu':
            if grad_clip > 0:
                for p in self.update:
                    if p in grads:
                        grads[p] = np.minimum(grads[p], grad_clip)
                        grads[p] = np.maximum(grads[p], -grad_clip)

        # perform parameter update
        for p in self.update:
            if p in grads:
                if sdg_type == 'vanilla':
                    if momentum > 0:
                        dx = momentum * self.step_cache[p] - learning_rate * grads[p]
                    else:
                        dx = -learning_rate * grads[p]
                    self.step_cache[p] = dx
                elif sdg_type == 'rmsprop':
                    self.step_cache[p] = self.step_cache[p] * decay_rate + (1.0 - decay_rate) * grads[p] ** 2
                    dx = -(learning_rate * grads[p]) / np.sqrt(self.step_cache[p] + smooth_eps)
                elif sdg_type == 'adgrad':
                    self.step_cache[p] += grads[p] ** 2
                    dx = -(learning_rate * grads[p]) / np.sqrt(self.step_cache[p] + smooth_eps)

                self.model[p] += dx

        out = {}
        out['loss'] = cost
        return out

    """ prediction """

    def predict(self, Xs, **kwargs):
        Xs = Xs[0]
        Ys, caches = self.fwdPass(Xs, self.parameter, predict_model=True)
        pred_action = np.argmax(Ys)

        return Ys, pred_action

    def update_target_network(self):
        self.clone_dqn = copy.deepcopy(self)

    def save_model(self, model_performance,episodes_index, checkpoint_path = None):
        if checkpoint_path == None: checkpoint_path = self.checkpoint_path
        agent_id = self.parameter.get("agent_id")
        dqn_id = self.parameter.get("dqn_id")
        disease_number = self.parameter.get("disease_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_wrong_disease = model_performance["average_wrong_disease"]
        model_file_name = "model_d" + str(disease_number) + "_agent" + str(agent_id) + "_dqn" + str(dqn_id) + "_s" + str(success_rate) + "_r" + str(average_reward) + "_t" + \
                          str(average_turn) + "_wd" + str(average_wrong_disease) + "_e" + str(episodes_index) + ".p"
        pickle.dump(file=open(checkpoint_path + model_file_name, "wb"), obj=self.model)

    def __initWeight(self,n, d):
        scale_factor = math.sqrt(float(6) / (n + d))
        # scale_factor = 0.1
        return (np.random.rand(n, d) * 2 - 1) * scale_factor

    """ for all k in d0, d0 += d1 . d's are dictionaries of key -> numpy array """

    def __mergeDicts(self, d0, d1):
        for k in d1:
            if k in d0:
                d0[k] += d1[k]
            else:
                d0[k] = d1[k]