# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf


class ActorCritic(object):
    def __init__(self,input_size, hidden_size,output_size, parameter):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parameter = parameter
        self.actor_learning_rate = parameter.get("actor_learning_rate")
        self.critic_learning_rate = parameter.get("critic_learning_rate")
        self.discount_factor = parameter.get("gamma")

        with tf.device("/device:GPU:0"):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.input = tf.placeholder(dtype=tf.float64, shape=(None, self.input_size), name="input")
                self.action = tf.placeholder(dtype=tf.float64, shape=(None, self.output_size), name="action")
                self.td_error = tf.placeholder(dtype=tf.float64, shape=(None, 1), name="td_error")
                self.target_value = tf.placeholder(dtype=tf.float64, shape=(None,1), name="next_value")

                with tf.variable_scope(name_or_scope="actor"):
                    self.actor_output = self.__actor_network()
                    self.action_prob = tf.nn.softmax(self.actor_output)
                    self.actor_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.log(self.action_prob), self.action,) * self.td_error), name="actor_loss")
                    self.actor_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.actor_learning_rate).minimize(-self.actor_loss)

                with tf.variable_scope(name_or_scope="critic"):
                    self.critic_output = self.__critic_network()
                    self.critic_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.target_value - self.critic_output)),name="critic_loss")
                    self.critic_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.critic_learning_rate).minimize(self.critic_loss)
                self.initializer = tf.global_variables_initializer()
            self.graph.finalize()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.session = tf.Session(graph=self.graph, config=config)
        self.session.run(self.initializer)

    def __actor_network(self):
        W1 = tf.get_variable(name="W1", shape=(self.input_size, self.hidden_size), dtype=tf.float64)
        b1 = tf.get_variable(name="b1", shape=(self.hidden_size),dtype=tf.float64)
        h1 = tf.nn.tanh(tf.matmul(self.input, W1) + b1)
        W2 = tf.get_variable(name="W2", shape=(self.hidden_size, self.output_size), dtype=tf.float64)
        b2 = tf.get_variable(name="b2", shape=(self.output_size),dtype=tf.float64)
        p = tf.matmul(h1, W2) + b2
        return p

    def __critic_network(self):
        W1 = tf.get_variable(name="W1", shape=(self.input_size, self.hidden_size), dtype=tf.float64)
        b1 = tf.get_variable(name="b1", shape=(self.hidden_size),dtype=tf.float64)
        h1 = tf.nn.tanh(tf.matmul(self.input, W1) + b1)
        W2 = tf.get_variable(name="W2", shape=(self.hidden_size, 1), dtype=tf.float64)
        b2 = tf.get_variable(name="b2", shape=(1),dtype=tf.float64)
        value = tf.matmul(h1, W2) + b2
        return value

    def actor_predict(self, Xs):
        """
        Output the probability distribution over actions for different states in Xs.
        :param Xs: A list of States.
        :return: A list of probability distribution over actions, whose length is 'len(Xs)'.
        """
        feed_dict = {self.input:Xs}
        Ys = self.session.run(fetches=self.action_prob, feed_dict=feed_dict)
        return Ys

    def critic_predict(self, Xs):
        """
        Output the value of different states in Xs.
        :param Xs: A list of state_rep.
        :return: a list of float as state_value with the length of 'len(Xs)'
        """
        feed_dict = {self.input:Xs}
        Ys = self.session.run(fetches=self.critic_output, feed_dict=feed_dict)
        return Ys

    def train(self,trajectory):
        # state, agent_action, reward, next_state, episode_over
        temp_discount = 1.0
        for turn in trajectory:
            state = turn[0]
            action = turn[1]
            reward = turn[2]
            next_state = turn[3]
            episode_over = turn[4]
            # Value of state
            if episode_over == True:
                target_value = reward
            else:
                target_value = reward + self.discount_factor * self.critic_predict(Xs=[next_state])[0][0]

            td_error = self.__update_critic(target_value=target_value,state=state)
            self.__update_actor(td_error=td_error,state=state, action=action,discount=temp_discount)
            temp_discount = temp_discount * self.discount_factor

    def __update_critic(self, state, target_value):
        feed_dict = {self.input:[state], self.target_value:[[target_value]]}
        self.session.run(fetches=self.critic_optimizer, feed_dict=feed_dict)
        td_error = target_value - self.session.run(fetches=self.critic_output, feed_dict={self.input:[state]})[0][0]
        return td_error

    def __update_actor(self, td_error, state,action,discount):
        taken_action = np.zeros(self.output_size)
        taken_action[action] = 1.0
        feed_dict = {self.input:[state], self.td_error:[[discount*td_error]], self.action:[taken_action]}
        self.session.run(fetches=self.actor_optimizer, feed_dict=feed_dict)