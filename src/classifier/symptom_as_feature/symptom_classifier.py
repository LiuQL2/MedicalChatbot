# -*- coding: utf-8 -*-
"""
使用主诉里面获得症状进行分类，把疾病判断看成一个分类任务；
"""

import copy
import json
import random
import numpy as np
import tensorflow as tf
import sys, os
from sklearn import svm
sys.path.append(os.getcwd().replace("src/classifier/symptom_as_feature",""))


class SymptomClassifier(object):
    def __init__(self, goal_set,symptom_set, disease_symptom, hidden_size,parameter, k_fold):
        self.k_fold = k_fold
        self.goal_set = goal_set
        self.hidden_size = hidden_size
        self.checkpoint_path = parameter.get("checkpoint_path")
        self.log_dir = parameter.get("log_dir")
        self.parameter = parameter
        self._disease_index(disease_symptom=disease_symptom)
        self._symptom_index(symptom_set=symptom_set)
        self._prepare_data_set(k_fold=k_fold)
        print(self.disease_to_index)

    def _symptom_index(self, symptom_set):
        """
        Mapping symptom to index and index to symptom.
        :param symptom_set:
        :return:
        """
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
        """
        Mapping disease to index and index to disease.
        :param disease_symptom:
        :return:
        """
        index = 0
        self.disease_sample_count = {}
        disease_to_index = {}
        index_to_disease = {}
        for key in disease_symptom.keys():
            disease_to_index[key] = index
            index_to_disease[index] = key
            self.disease_sample_count[key] = 0
            index += 1
        self.disease_to_index = disease_to_index
        self.index_to_disease = index_to_disease

    def _prepare_data_set(self, k_fold):
        """
        Preparing the dataset for training and evaluating.
        :return:
        """
        batch_size = self.parameter.get("batch_size")
        explicit_number = self.parameter.get('explicit_number')
        implicit_number = self.parameter.get('implicit_number')
        data_set = {}
        all_sample = self.goal_set["train"] + self.goal_set["test"] + self.goal_set["validate"]
        random.shuffle(all_sample)
        fold_size = int(len(all_sample) / k_fold)

        fold_list = [all_sample[i:i+fold_size] for i in range(0,len(all_sample),fold_size)]

        for k in range(0, k_fold, 1):
            data_set[k] = {
                "x_ex":[],
                "x_im":[],
                "x_ex_im":[],
                "y":[]
            }
            fold = fold_list[k]
            for goal in fold:
                disease_rep = np.zeros(len(self.disease_to_index.keys()))
                disease_rep[self.disease_to_index[goal["disease_tag"]]] = 1
                symptom_rep_ex = np.zeros(len(self.symptom_to_index.keys()))
                symptom_rep_im = np.zeros(len(self.symptom_to_index.keys()))
                symptom_rep_ex_im = np.zeros(len(self.symptom_to_index.keys()))
                # explicit
                for symptom, value in goal["goal"]["explicit_inform_slots"].items():
                    if value == True:
                        symptom_rep_ex[self.symptom_to_index[symptom]] = 1
                        symptom_rep_ex_im[self.symptom_to_index[symptom]] = 1
                    else:
                        symptom_rep_ex[self.symptom_to_index[symptom]] = -1
                        symptom_rep_im[self.symptom_to_index[symptom]] = -1
                        symptom_rep_ex_im[self.symptom_to_index[symptom]] = -1

                # implicit
                for symptom, value in goal["goal"]["implicit_inform_slots"].items():
                    if value == True:
                        symptom_rep_ex_im[self.symptom_to_index[symptom]] = 1
                    else:
                        symptom_rep_ex_im[self.symptom_to_index[symptom]] = -1
                # print(data_set)
                append_or_not = False
                if len(goal["goal"]["explicit_inform_slots"].keys()) >= explicit_number and \
                        len(goal["goal"]["implicit_inform_slots"].keys()) >= implicit_number:
                    append_or_not = True
                if append_or_not:
                    self.disease_sample_count[goal["disease_tag"]] += 1
                    data_set[k]["x_ex"].append(symptom_rep_ex)
                    data_set[k]["x_im"].append(symptom_rep_im)
                    data_set[k]["x_ex_im"].append(symptom_rep_ex_im)
                    data_set[k]["y"].append(disease_rep)

        self.data_set = data_set

    def train_sklearn_svm(self):
        accuracy = 0.0
        for key in self.data_set.keys():
            print("fold index:", key)
            train_set = copy.deepcopy(self.data_set)
            test_set = train_set.pop(key)
            accuracy += self._train_and_evaluate_svm_one_fold_(train_set, test_set)

        print("accuracy on "+ str(len(self.data_set.keys())) + " cross-fold:", accuracy / len(self.data_set.keys()))


    def _train_and_evaluate_svm_one_fold_(self, train_set, test_set):
        """

        :param train_set: dict, {"fold_index":{"x":[],"x_ex":[]]}
        :param test_set: a list of batches.
        :return:
        """
        train_feature = self.parameter.get("train_feature")
        test_feature = self.parameter.get("test_feature")
        clf = svm.SVC(decision_function_shape="ovo")
        Xs = []
        Ys = []
        for fold in train_set.values():
            Ys = Ys + list(np.argmax(fold['y'], axis=1))
            if train_feature == "ex":
                Xs = Xs + fold["x_ex"]
            elif train_feature == "im":
                Xs = Xs + fold["x_im"]
            elif train_feature == "ex&im":
                Xs = Xs + fold["x_ex_im"]
        clf.fit(X=Xs, y=Ys)

        # Test
        Ys = list(np.argmax(test_set['y'],axis=1))
        if test_feature == "ex":
            Xs = test_set["x_ex"]
        elif test_set == "im":
            Xs = test_set["x_im"]
        elif test_feature == "ex&im":
            Xs = test_set["x_ex_im"]
        predict_ys = clf.predict(Xs)
        disease_accuracy = self._accuracy_for_each_disease(labels=Ys,predicted_ys=predict_ys)
        total_accuracy = disease_accuracy["total_accuracy"]
        print(disease_accuracy)
        return total_accuracy


    def _accuracy_for_each_disease(self, labels, predicted_ys):
        disease_accuracy = {}
        count = 0.0
        for disease in self.disease_to_index.keys():
            disease_accuracy[disease] = {}
            disease_accuracy[disease]["success_count"] = 0.0
            disease_accuracy[disease]["total"] = 0.0
        for sample_index in range(0, len(labels), 1):
            disease_accuracy[self.index_to_disease[labels[sample_index]]]["total"] += 1
            if labels[sample_index] == predicted_ys[sample_index]:
                count += 1
                disease_accuracy[self.index_to_disease[labels[sample_index]]]["success_count"] += 1

            # if self.index_to_disease[labels[sample_index]] in ["小儿支气管肺炎", "急性支气管炎"]:
            #     print("true:", self.index_to_disease[labels[sample_index]], "predict:", self.index_to_disease[predicted_ys[sample_index]])

        for disease in self.disease_to_index.keys():
            disease_accuracy[disease]["accuracy"] = disease_accuracy[disease]["success_count"] / disease_accuracy[disease]["total"]
        disease_accuracy["total_accuracy"] = count / len(labels)
        return disease_accuracy