# -*- coding: utf-8 -*-
"""
使用主诉里面获得症状进行分类，把疾病判断看成一个分类任务；
"""

import numpy as np
import tensorflow as tf
import sys, os
sys.path.append(os.getcwd().replace("src/classifier/symptom_as_feature",""))


class SymptomClassifier(object):
    def __init__(self, goal_set,symptom_set, disease_symptom, hidden_size,parameter):
        self.goal_set = goal_set
        self.hidden_size = hidden_size
        self.checkpoint_path = parameter.get("checkpoint_path")
        self.log_dir = parameter.get("log_dir")
        self.parameter = parameter
        self._disease_index(disease_symptom=disease_symptom)
        self._symptom_index(symptom_set=symptom_set)
        self._prepare_data_set()
        self._build_model()

    def _build_model(self):
        with tf.device("/device:GPU:0"):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.input = tf.placeholder(dtype=tf.float64, shape=(None, len(self.symptom_to_index.keys())), name="input")
                self.label = tf.placeholder(dtype=tf.float64, shape=(None, len(self.disease_to_index.keys())), name="target_value")
                # input layer.
                with tf.name_scope(name="input_hidden"):
                    self.input_hidden_weights = tf.get_variable(name="weights", shape=(len(self.symptom_to_index.keys()), self.hidden_size),dtype=tf.float64)
                    self.input_hidden_bias = tf.get_variable(name="bias", shape=(self.hidden_size), dtype=tf.float64)
                    self.input_hidden_output = tf.nn.relu(tf.add(tf.matmul(self.input, self.input_hidden_weights), self.input_hidden_bias),name="input_hidden_output")
                with tf.name_scope(name="hidden_output"):
                    self.hidden_output_weights = tf.get_variable(name="output_weights", shape=(self.hidden_size, len(self.disease_to_index.keys())),dtype=tf.float64)
                    self.hidden_output_bias = tf.get_variable(name="output_bias", shape=(len(self.disease_to_index.keys())), dtype=tf.float64)
                    self.hidden_output_softmax_output = tf.nn.softmax(logits=tf.nn.relu(tf.add(tf.matmul(self.input_hidden_output, self.hidden_output_weights), self.hidden_output_bias)),name="output_layer_output")
                    self.hidden_output_logit_output = tf.add(tf.matmul(self.input_hidden_output, self.hidden_output_weights), self.hidden_output_bias,name="output_layer_output")

                # Optimization.
                # self.loss = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(self.hidden_output_softmax_output), reduction_indices=[1]),name="loss")
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=self.hidden_output_logit_output,name="loss"))
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.parameter.get("learning_rate")).minimize(loss=self.loss)
                self.initializer = tf.global_variables_initializer()
                self.model_saver = tf.train.Saver()
            self.graph.finalize()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.session = tf.Session(graph=self.graph,config=config)
        self.session.run(self.initializer)

    def train(self):
        for index in range(0, len(self.data_set["train"]), 1):
            batch = self.data_set["train"][index]
            feed_dict = {self.input:batch["x"], self.label:batch["y"]}
            loss = self.session.run(self.loss, feed_dict=feed_dict)
            print(loss)
            print("%d, loss: %f"% (index,loss))
            self.session.run(self.optimizer,feed_dict=feed_dict)

    def predict(self, batch):
        feed_dict = {self.input: batch["x"]}
        Ys = self.session.run(self.hidden_output_softmax_output, feed_dict=feed_dict)
        max_index = np.argmax(Ys, axis=1)
        return max_index

    def evaluate(self):
        predict_lable = self.predict(batch=self.data_set["test"])
        label = np.argmax(self.data_set["test"]["y"],axis=1)
        result = np.equal(label, predict_lable)
        count = 0
        for right in result:
            if right: count += 1
        print("accuracy:", float(count)/len(result))

    def _symptom_index(self, symptom_set):
        index = 0
        symptom_to_index = {}
        index_to_symptom = {}
        if "disease" in symptom_set.keys():
            symptom_set.pop("disease")
        for key, value in symptom_set.items():
            symptom_to_index[key] = index
            index_to_symptom[index] = key
            index += 1
        self.symptom_to_index = symptom_to_index
        self.index_to_symptom = index_to_symptom

    def _disease_index(self, disease_symptom):
        index = 0
        disease_to_index = {}
        index_to_disease = {}
        for key in disease_symptom.keys():
            disease_to_index[key] = index
            index_to_disease[index] = key
            index += 1
        self.disease_to_index = disease_to_index
        self.index_to_disease = index_to_disease

    def _prepare_data_set(self):
        batch_size = self.parameter.get("batch_size")
        explicit_only = self.parameter.get("explicit_only")
        data_set = {
            "train":[],
            "test":{"x":[],"y":[]}
        }
        batch = {
            "x":[],
            "y":[]
        }
        for goal in self.goal_set["train"]:
            disease_rep = np.zeros(len(self.disease_to_index.keys()))
            disease_rep[self.disease_to_index[goal["disease_tag"]]] = 1
            symptom_rep = np.zeros(len(self.symptom_to_index.keys()))
            for symptom in goal["goal"]["explicit_inform_slots"].keys():
                symptom_rep[self.symptom_to_index[symptom]] = 1
            if explicit_only == 0:
                for symptom in goal["goal"]["implicit_inform_slots"].keys():
                    symptom_rep[self.symptom_to_index[symptom]] = 1
            if len(batch["x"]) == batch_size:
                data_set["train"].append(batch)
                batch["x"] = []
                batch["y"] = []
            else:
                batch["x"].append(symptom_rep)
                batch["y"].append(disease_rep)

        for goal in self.goal_set["test"]:
            disease_rep = np.zeros(len(self.disease_to_index.keys()))
            disease_rep[self.disease_to_index[goal["disease_tag"]]] = 1
            symptom_rep = np.zeros(len(self.symptom_to_index.keys()))
            for symptom in goal["goal"]["explicit_inform_slots"].keys():
                symptom_rep[self.symptom_to_index[symptom]] = 1
            if explicit_only == 0:
                for symptom in goal["goal"]["implicit_inform_slots"].keys():
                    symptom_rep[self.symptom_to_index[symptom]] = 1

            data_set["test"]["x"].append(symptom_rep)
            data_set["test"]["y"].append(disease_rep)
        self.data_set = data_set