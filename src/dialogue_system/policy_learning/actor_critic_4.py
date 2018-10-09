# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class ActorCritic(object):
    def __init__(self, input_size, hidden_size, output_size, parameter):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parameter = parameter
        self.log_dir = parameter.get("log_dir")
        self.gamma = parameter.get("gamma")

        with tf.device("/device:GPU:0"):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.actor = Actor(input_size=input_size,hidden_size=hidden_size,output_size=output_size,parameter=parameter)
                self.critic = Critic(input_size=input_size,hidden_size=hidden_size,output_size=1,parameter=parameter)
                self.initializer = tf.global_variables_initializer()
            self.graph.finalize()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.session = tf.Session(graph=self.graph, config=config)
        self.session.run(self.initializer)

        # Visualizing graph.
        # self.summary_writer = tf.summary.FileWriter(logdir=self.log_dir + "train", graph=self.graph)

    def train(self, trajectories):
        b_obs, b_acts, b_advantages, b_advantages_actor, b_target_values = [],[],[],[],[]

        for trajectory in trajectories:
            observations, advantages, advantages_actor, actions, target_values = self.__process_trajectory__(trajectory)
            b_obs.extend(observations)
            b_acts.extend(actions)
            b_advantages.extend(advantages)
            b_target_values.extend(target_values)
            b_advantages_actor.extend(advantages_actor)

        # Normalizing advantages.
        b_advantages = (b_advantages - np.mean(b_advantages)) / (1e-10 + np.std(b_advantages))
        b_advantages = np.reshape(b_advantages,newshape=-1)
        self.critic.train(sess=self.session,observations=b_obs,target_values=b_target_values)

        b_advantages_actor = (b_advantages_actor - np.mean(b_advantages_actor)) / (1e-10 + np.std(b_advantages_actor))
        b_advantages_actor = np.reshape(b_advantages_actor,newshape=-1)
        self.actor.train(sess=self.session,observations=b_obs,
                         actions=b_acts,advantages=b_advantages)



    def train_actor(self, observations, actions, advantages):
        self.actor.train(sess=self.session,observations=observations,
                         actions=actions,advantages=advantages)

    def __process_trajectory__(self, trajectory):
        # state, agent_action, reward, next_state, episode_over

        obs, acts, rews, next_obs, dones = [], [], [], [], []
        for turn in trajectory:
            obs.append(turn[0])
            acts.append(turn[1])
            rews.append(turn[2])
            next_obs.append(turn[3])
            dones.append(turn[4])
        next_values = self.critic.predict(self.session, observations=next_obs)
        values = self.critic.predict(self.session, observations=obs)
        advantages, advantages_actor, target_values = [], [], []
        gamma = self.gamma
        for index in range(0, len(rews),1):
            if dones[index]:
                advantages.append(rews[index] - values[index])
                advantages_actor.append(gamma*(rews[index] - values[index]))
                target_values.append(rews[index])
            else:
                advantages.append(rews[index] + self.gamma * next_values[index] - values[index])
                advantages_actor.append(gamma*(rews[index] + self.gamma * next_values[index] - values[index]))
                target_values.append(rews[index] + self.gamma * next_values[index])
            gamma *= self.gamma
        return list(obs), list(advantages), list(advantages_actor), list(acts), list(target_values)

    def take_action(self, state):
        action = self.actor.take_action(sess=self.session, observation=state)
        return action


class Actor(object):
    def __init__(self, input_size, hidden_size, output_size, parameter):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parameter = parameter
        self.gamma = parameter.get("gamma")
        self.learning_rate = parameter.get("actor_learning_rate")
        self.__build_model__()

    def __build_model__(self):
        with tf.variable_scope("actor"):
            self.keep_prob = tf.placeholder(tf.float64, shape=None)
            self.observations = tf.placeholder(dtype=tf.float64, shape=[None, self.input_size],name="observations")
            self.actions = tf.placeholder(dtype=tf.int32,shape=[None],name="actions")
            self.advantages = tf.placeholder(dtype=tf.float64, shape=[None], name="advantages")
            h1 = tf.layers.dense(inputs=self.observations, units=self.hidden_size, activation=tf.nn.relu,
                             kernel_initializer=tf.random_normal_initializer,name="first_layer")
            h1_dropout = tf.nn.dropout(h1, self.keep_prob)
            self.logits = tf.layers.dense(inputs=h1, units=self.output_size,activation=None,
                                     kernel_initializer=tf.random_normal_initializer,name="second_layer")
            self.sample_action = tf.multinomial(self.logits,1, name="sample_action")
            self.log_prob = tf.nn.log_softmax(self.logits + 1e-10)
            self.log_prob_of_old_actions = tf.gather_nd(params=self.log_prob,
                                                indices=tf.stack([tf.range(tf.shape(self.actions)[0]), self.actions], axis=1))

            self.loss = - tf.reduce_mean(tf.multiply(self.advantages, self.log_prob_of_old_actions))
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, sess, observations, actions, advantages):
        assert len(observations) == len(actions), ("The number of Obs. does not equal to the number of Act.")
        assert len(actions) == len(advantages), ("The number of Act. does not equal to the number of Adv.")

        feed_dict = {self.observations:observations, self.advantages:advantages, self.actions:actions, self.keep_prob:self.parameter["keep_prob"]}
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def take_action(self, sess, observation):
        action = sess.run(self.sample_action,feed_dict = {self.observations:[observation], self.keep_prob:1.0})[0][0]
        return action


class Critic(object):
    def __init__(self, input_size, hidden_size, output_size, parameter):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parameter = parameter
        self.gamma = parameter.get("gamma")
        self.learning_rate = parameter.get("critic_learning_rate")
        self.__build_model__()

    def __build_model__(self):
        with tf.variable_scope("critic"):
            self.keep_prob = tf.placeholder(tf.float64)
            self.observations = tf.placeholder(dtype=tf.float64, shape=[None, self.input_size],name="observations")
            self.target_value = tf.placeholder(dtype=tf.float64,shape=[None],name="actions")
            h1 = tf.layers.dense(inputs=self.observations, units=self.hidden_size, activation=tf.nn.relu,
                             kernel_initializer=tf.random_normal_initializer,name="first_layer")
            h1_dropout = tf.nn.dropout(h1, self.keep_prob)

            h2 = tf.layers.dense(inputs=h1_dropout, units=self.hidden_size, activation=tf.nn.relu,
                             kernel_initializer=tf.random_normal_initializer,name="second_layer")
            h2_dropout = tf.nn.dropout(h2, self.keep_prob)
            self.value = tf.layers.dense(inputs=h2_dropout, units=self.output_size,activation=None,
                                     kernel_initializer=tf.random_normal_initializer,name="output_layer")

            self.loss = tf.reduce_mean(tf.square(self.value - self.target_value),name="loss") / 2

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, sess, observations, target_values):
        assert len(observations) == len(target_values), "The number of Obs. dose not equal to the number of targets."

        feed_dict = {self.observations:observations, self.target_value:target_values, self.keep_prob:self.parameter["keep_prob"]}
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict = feed_dict)
        return loss

    def predict(self, sess, observations):
        return sess.run(self.value, feed_dict={self.observations:observations, self.keep_prob:1.0})