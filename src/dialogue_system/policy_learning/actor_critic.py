# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class ActorCritic(object):
    def __init__(self, input_size, hidden_size, output_size, parameter):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parameter = parameter
        self.gamma = parameter.get("gamma")

        with tf.device("/device:GPU:0"):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.actor = Actor(input_size=input_size,hidden_size=hidden_size,output_size=output_size,parameter=parameter)
                self.critic = Critic(input_size=input_size,hidden_size=hidden_size,output_size=output_size,parameter=parameter)
                self.initializer = tf.global_variables_initializer()
            self.graph.finalize()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.session = tf.Session(graph=self.graph, config=config)
        self.session.run(self.initializer)

    def train(self, trajectories):
        b_states, b_td_errors, b_taken_actions, b_critic_labels = [],[],[],[]
        for trajectory in trajectories:
            states, td_errors, taken_actions, critic_labels = self.__process_trajectory__(trajectory)
            b_states.extend(states)
            b_td_errors.extend(td_errors)
            b_taken_actions.extend(taken_actions)
            b_critic_labels.extend(critic_labels)
        # normalize rewards; don't divide by 0
        b_td_errors = (b_td_errors - np.mean(b_td_errors)) / (np.std(b_td_errors) + 1e-10 )
        self.critic.train(sess=self.session,Xs=b_states,Ys=b_critic_labels)
        self.actor.train(sess=self.session,inputs=b_states,actions=b_taken_actions,td_errors=b_td_errors)

    def __process_trajectory__(self, trajectory):
        # state, agent_action, reward, next_state, episode_over
        states = []
        next_states = []
        for i,x in enumerate(trajectory):
            state_pre = x[0]
            next_state_pre = x[3]
            states.append(state_pre)
            next_states.append(next_state_pre)
        next_state_values = self.critic.current_predict(sess=self.session, Xs=next_states)
        # next_state_values = self.critic.target_predict(sess=self.session, Xs=next_states)

        critic_labels = []
        for i, x in enumerate(trajectory):
            reward = x[2] # int
            episode_over = x[4] # bool
            action = np.zeros(self.output_size)
            action[x[1]] = 1.0

            next_state_value = next_state_values[i]
            label = reward
            if not episode_over:
                label = float(self.gamma * next_state_value + reward)
            critic_labels.append([label])

        state_values = self.critic.current_predict(sess=self.session,Xs=states)
        taken_actions = []
        td_errors = []
        for i, x in enumerate(trajectory):
            reward = x[2] # int
            episode_over = x[4] # bool
            action = np.zeros(self.output_size)
            action[x[1]] = 1.0

            state_value = state_values[i]
            next_state_value = next_state_values[i]
            label = reward
            if not episode_over:
                label = float(self.gamma * next_state_value + reward)
            td_errors.append(label - state_value)
            taken_actions.append(action)
        return states, td_errors, taken_actions, critic_labels

    def actor_predict(self, Xs):
        return self.actor.current_predict(sess=self.session,Xs=Xs)

    def update_target_network(self):
        self.actor.update_target_network(sess=self.session)
        self.critic.update_target_network(sess=self.session)

    def take_action(self,state):
        action_prob = self.actor.current_predict(sess=self.session, Xs=[state])
        action_index = np.random.choice(np.arange(self.output_size), p=action_prob[0])
        return action_index


class Actor(object):
    def __init__(self,input_size, hidden_size, output_size, parameter):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parameter = parameter
        self.learning_rate = parameter.get("actor_learning_rate")
        self.variable = {}

        with tf.variable_scope(name_or_scope="actor"):
            self.input = tf.placeholder(dtype=tf.float64,shape=(None, input_size), name="input")
            self.take_actions = tf.placeholder(dtype=tf.float64,shape=(None, output_size), name="taken_actions")
            self.td_error = tf.placeholder(dtype=tf.float64, shape=(None, 1), name="td_error")

            self.current_output = self.__build_network("current_network")
            self.current_action_prob = tf.nn.softmax(self.current_output)
            self.log_current_action_prob = tf.multiply(tf.log(self.current_action_prob), self.take_actions,name="log_current_action_prob")
            self.target_output = self.__build_network("target_network")
            self.target_action_prob = tf.nn.softmax(self.target_output)

            self.update_assign = self.__update_target_network_operations()

            # Loss
            # self.loss = tf.multiply(self.log_current_action_prob, self.td_error, name="loss")
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(self.log_current_action_prob, self.td_error),axis=0), name="loss")
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def __build_network(self, network_name):
        self.variable[network_name] = {}
        with tf.variable_scope(name_or_scope=network_name):
            # Two layers.
            self.variable[network_name]["W1"] = tf.get_variable(name="W1", shape=(self.input_size, self.hidden_size), dtype=tf.float64)
            self.variable[network_name]["b1"] = tf.get_variable(name="b1", shape=(self.hidden_size),dtype=tf.float64)
            h1 = tf.nn.tanh(tf.matmul(self.input, self.variable[network_name]["W1"]) + self.variable[network_name]["b1"])
            self.variable[network_name]["W2"] = tf.get_variable(name="W2", shape=(self.hidden_size, 1), dtype=tf.float64)
            self.variable[network_name]["b2"] = tf.get_variable(name="b2", shape=(self.output_size),dtype=tf.float64)
            # value = tf.nn.relu(tf.matmul(h1, self.variable[network_name]["W2"]) + self.variable[network_name]["b2"])
            value = tf.matmul(h1, self.variable[network_name]["W2"]) + self.variable[network_name]["b2"]

            # One layer.
            # self.variable[network_name]["W1"] = tf.get_variable(name="W1", shape=(self.input_size, self.output_size), dtype=tf.float64)
            # self.variable[network_name]["b1"] = tf.get_variable(name="b1", shape=(self.output_size),dtype=tf.float64)
            # value = tf.matmul(self.input, self.variable[network_name]["W1"]) + self.variable[network_name]["b1"]
        return value

    def __update_target_network_operations(self):
        update_assign = {}
        for key in self.variable["current_network"].keys():
            update_assign[key] = tf.assign(self.variable["target_network"][key],value=self.variable["current_network"][key].value(),name=key+"-"+key)
        return update_assign

    def train(self, sess, inputs, actions, td_errors):
        feed_dict = {self.input:inputs, self.td_error:td_errors,self.take_actions:actions}
        sess.run(fetches=self.optimizer, feed_dict=feed_dict)

    def target_predict(self,sess, Xs):
        return sess.run(fetches=self.target_action_prob, feed_dict={self.input:Xs})

    def current_predict(self,sess, Xs):
        return sess.run(fetches=self.current_action_prob, feed_dict={self.input:Xs})

    def update_target_network(self, sess):
        sess.run(fetches=list(self.update_assign.values()))


class Critic(object):
    def __init__(self,input_size, hidden_size, output_size, parameter):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parameter = parameter
        self.learning_rate = parameter.get("critic_learning_rate")
        self.variable = {}

        with tf.variable_scope(name_or_scope="critic"):
            self.input = tf.placeholder(dtype=tf.float64, shape=(None, input_size),name="input")
            self.target_value = tf.placeholder(dtype=tf.float64, shape=(None,1),name="target_value")
            self.target_output = self.__build_network(network_name="target_network")
            self.current_output = self.__build_network(network_name="current_network")
            self.update_assign = self.__update_target_network_operations()

            # Loss
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.target_value - self.current_output),axis=1),name="loss")
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def __build_network(self, network_name):
        self.variable[network_name] = {}
        with tf.variable_scope(name_or_scope=network_name):
            # Two layers.
            self.variable[network_name]["W1"] = tf.get_variable(name="W1", shape=(self.input_size, self.hidden_size), dtype=tf.float64)
            self.variable[network_name]["b1"] = tf.get_variable(name="b1", shape=(self.hidden_size),dtype=tf.float64)
            h1 = tf.nn.tanh(tf.matmul(self.input, self.variable[network_name]["W1"]) + self.variable[network_name]["b1"])
            self.variable[network_name]["W2"] = tf.get_variable(name="W2", shape=(self.hidden_size, 1), dtype=tf.float64)
            self.variable[network_name]["b2"] = tf.get_variable(name="b2", shape=(1),dtype=tf.float64)
            value = tf.nn.tanh(tf.matmul(h1, self.variable[network_name]["W2"]) + self.variable[network_name]["b2"])

            # One layer.
            # self.variable[network_name]["W1"] = tf.get_variable(name="W1", shape=(self.input_size, 1), dtype=tf.float64)
            # self.variable[network_name]["b1"] = tf.get_variable(name="b1", shape=(1),dtype=tf.float64)
            # value = tf.nn.tanh(tf.matmul(self.input, self.variable[network_name]["W1"]) + self.variable[network_name]["b1"])
            # value = tf.matmul(self.input, self.variable[network_name]["W1"]) + self.variable[network_name]["b1"]
        return value

    def __update_target_network_operations(self):
        update_assign = {}
        for key in self.variable["current_network"].keys():
            update_assign[key] = tf.assign(self.variable["target_network"][key],value=self.variable["current_network"][key].value(),name=key+"-"+key)
        return update_assign

    def train(self, sess, Xs, Ys):
        feed_dict = {self.input:Xs, self.target_value:Ys}
        sess.run(fetches=self.optimizer, feed_dict=feed_dict)

    def target_predict(self,sess, Xs):
        return sess.run(fetches=self.target_output, feed_dict={self.input:Xs})

    def current_predict(self,sess, Xs):
        return sess.run(fetches=self.current_output, feed_dict={self.input:Xs})

    def update_target_network(self, sess):
        sess.run(fetches=list(self.update_assign.values()))