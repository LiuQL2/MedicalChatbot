# -*- coding: utf-8 -*-


import copy
import json
import random
import pickle
import copy
import numpy as np
import tensorflow as tf
import sys, os
from sklearn import svm
sys.path.append(os.getcwd().replace("src/classifier/symptom_as_feature",""))


class Finder(object):
    def __init__(self, goal_set,symptom_set, disease_symptom, k_fold):
        self.k_fold = k_fold
        self.goal_set = goal_set
        self.wrong_samples = {}
        self._disease_index(disease_symptom=disease_symptom)
        self._symptom_index(symptom_set=symptom_set)
        self.goal_by_disease_set = self.__goal_by_disease__()

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
        disease_to_index = {}
        index_to_disease = {}
        for key in disease_symptom.keys():
            disease_to_index[key] = index
            index_to_disease[index] = key
            index += 1
        self.disease_to_index = disease_to_index
        self.index_to_disease = index_to_disease

    def find(self,save_path):
        self._prepare_data_set(goal_by_disease=copy.deepcopy(self.goal_by_disease_set))
        disease_accuracy = self.train_sklearn_svm()
        # print(disease_accuracy)

        save_set = True
        gap = float("%.4f"% (disease_accuracy["total_accuracy"]["ex&im"] - disease_accuracy["total_accuracy"]["ex"]))
        disease_accuracy.pop("total_accuracy")
        for disease, accuracy in disease_accuracy.items():
            if accuracy["ex&im"] <= accuracy["ex"]:
                save_set = False
                break

        if save_set:
            file_name = save_path + "goal_set_" + str(gap) + ".p"
            print("saving...",file_name)
            # self.dump_goal_set(dump_file_name=file_name)
        return gap,save_set

    def __goal_by_disease__(self):
        goal_by_disease = {}
        for key, goal_list in self.goal_set.items():
            for goal in goal_list:
                goal_by_disease.setdefault(goal["disease_tag"], list())
                append_or_not = self.__keep_sample_or_not__(goal)
                if append_or_not:
                    goal_by_disease[goal["disease_tag"]].append(goal)
        for key in goal_by_disease.keys():
            pass
            # print(key, len(goal_by_disease[key]))
        return goal_by_disease

    def _prepare_data_set(self, goal_by_disease):
        """
        Preparing the dataset for training and evaluating.
        :return:
        """
        disease_sample_count = {}
        sample_by_disease = {}
        data_set = {}

        all_sample = []
        for disease,goal_list in goal_by_disease.items():
            random.shuffle(goal_list)
            if disease == "小儿消化不良":
                all_sample = all_sample + list(random.sample(goal_list, 200))
            else:
                all_sample = all_sample + list(random.sample(goal_list, 300))

        random.shuffle(all_sample)
        fold_size = int(len(all_sample) / self.k_fold)

        fold_list = [all_sample[i:i+fold_size] for i in range(0,len(all_sample),fold_size)]

        for k in range(0, self.k_fold, 1):
            data_set[k] = {
                "x_ex":[],
                "x_im":[],
                "x_ex_im":[],
                "y":[],
                "consult_id":[]
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
                        symptom_rep_ex_im[self.symptom_to_index[symptom]] = -1

                # implicit
                for symptom, value in goal["goal"]["implicit_inform_slots"].items():
                    if value == True:
                        symptom_rep_im[self.symptom_to_index[symptom]] = 1
                        symptom_rep_ex_im[self.symptom_to_index[symptom]] = 1
                    else:
                        symptom_rep_ex_im[self.symptom_to_index[symptom]] = -1
                        symptom_rep_im[self.symptom_to_index[symptom]] = -1
                # print(data_set)
                append_or_not = self.__keep_sample_or_not__(goal)
                if append_or_not:

                    sample_by_disease.setdefault(goal["disease_tag"], dict())
                    sample_by_disease[goal["disease_tag"]][goal["consult_id"]] = goal

                    disease_sample_count.setdefault(goal["disease_tag"],0)
                    disease_sample_count[goal["disease_tag"]] += 1

                    data_set[k]["x_ex"].append(symptom_rep_ex)
                    data_set[k]["x_im"].append(symptom_rep_im)
                    data_set[k]["x_ex_im"].append(symptom_rep_ex_im)
                    data_set[k]["y"].append(disease_rep)
                    data_set[k]["consult_id"].append(goal["consult_id"])

        self.data_set = data_set
        self.sample_by_disease = sample_by_disease
        self.disease_sample_count = disease_sample_count

    def train_sklearn_svm(self):
        disease_accuracy = {}
        disease_accuracy["total_accuracy"] = {}
        disease_accuracy["total_accuracy"]["ex&im"] = 0.0
        disease_accuracy["total_accuracy"]["ex"] = 0.0

        for key in self.sample_by_disease.keys():
            disease_accuracy[key] = {}
            disease_accuracy[key]["ex&im"] = 0.0
            disease_accuracy[key]["ex"] = 0.0

        for key in self.data_set.keys():
            train_set = copy.deepcopy(self.data_set)
            test_set = train_set.pop(key)
            temp_accuracy_ex_im,temp_accuracy_ex = self._train_and_evaluate_svm_one_fold_(train_set, test_set)
            for key in temp_accuracy_ex_im.keys():
                disease_accuracy[key]["ex&im"] += temp_accuracy_ex_im[key]["accuracy"]
                disease_accuracy[key]["ex"] += temp_accuracy_ex[key]["accuracy"]

        for key,value in disease_accuracy.items():
            disease_accuracy[key]["ex&im"] = float("%.4f" % (value["ex&im"] / len(self.data_set.keys())))
            disease_accuracy[key]["ex"] = float("%.4f" % (value["ex"] / len(self.data_set.keys())))

        print(disease_accuracy)
        return disease_accuracy

    def _train_and_evaluate_svm_one_fold_(self, train_set, test_set):
        """

        :param train_set: dict, {"fold_index":{"x":[],"x_ex":[]]}
        :param test_set: a list of batches.
        :return:
        """
        clf_ex = svm.SVC(decision_function_shape="ovo")
        clf_ex_im = svm.SVC(decision_function_shape="ovo")
        Xs_ex = []
        Xs_ex_im = []
        Ys = []
        for fold in train_set.values():
            Ys = Ys + list(np.argmax(fold['y'], axis=1))
            Xs_ex = Xs_ex + fold["x_ex"]
            Xs_ex_im = Xs_ex_im + fold["x_ex_im"]
        clf_ex.fit(X=Xs_ex, y=Ys)
        clf_ex_im.fit(X=Xs_ex_im, y=Ys)

        # Test
        IDs = test_set["consult_id"]
        Ys = list(np.argmax(test_set['y'],axis=1))
        Xs_ex = test_set["x_ex"]
        Xs_ex_im = test_set["x_ex_im"]
        predict_ys_ex = clf_ex.predict(Xs_ex)
        predict_ys_ex_im = clf_ex_im.predict(Xs_ex_im)
        disease_accuracy_ex = self._accuracy_for_each_disease(labels=Ys,predicted_ys=predict_ys_ex, IDs=IDs)
        disease_accuracy_ex_im = self._accuracy_for_each_disease(labels=Ys,predicted_ys=predict_ys_ex_im, IDs=IDs)
        return disease_accuracy_ex_im, disease_accuracy_ex

    def _accuracy_for_each_disease(self, labels, predicted_ys,IDs):
        disease_accuracy = {}
        disease_accuracy["total_accuracy"]={}
        disease_accuracy["total_accuracy"]["accuracy"] = 0.0
        count = 0.0

        for disease in self.disease_sample_count.keys():
            disease_accuracy[disease] = {}
            disease_accuracy[disease]["success_count"] = 0.0
            disease_accuracy[disease]["total"] = 0.0
            disease_accuracy["total_accuracy"]["accuracy"] = 0.0
        for sample_index in range(0, len(labels), 1):
            disease_accuracy[self.index_to_disease[labels[sample_index]]]["total"] += 1
            if labels[sample_index] == predicted_ys[sample_index]:
                count += 1
                disease_accuracy[self.index_to_disease[labels[sample_index]]]["success_count"] += 1

            if labels[sample_index] != predicted_ys[sample_index] and self.index_to_disease[labels[sample_index]] in ["上呼吸道感染"]:
                self.wrong_samples.setdefault(IDs[sample_index],list())
                self.wrong_samples[IDs[sample_index]].append(self.index_to_disease[predicted_ys[sample_index]])
        for disease in self.disease_sample_count.keys():
            disease_accuracy[disease]["accuracy"] = disease_accuracy[disease]["success_count"] / disease_accuracy[disease]["total"]
        disease_accuracy["total_accuracy"]["accuracy"] = count / len(labels)
        return disease_accuracy


    def dump_goal_set(self, dump_file_name, train=0.8, test=0.2, validate=0.0):
        assert (train*100+test*100+validate*100==100), "train + test + validate not equals to 1.0."
        data_set = {
            "train":[],
            "test":[],
            "validate":[]
        }
        all_sample_list = []
        for disease in self.sample_by_disease.keys():
            for id, goal in self.sample_by_disease[disease].items():
                    all_sample_list.append(goal)

        random.shuffle(all_sample_list)

        data_set["train"] = list(all_sample_list[0:int(len(all_sample_list) * train)])
        data_set["test"] = list(all_sample_list[int(len(all_sample_list) * train):int(len(all_sample_list) * (train+test))])
        data_set["validate"] = list(all_sample_list[int(len(all_sample_list) * (train+test)):len(all_sample_list)])


        print("total",len(data_set["test"]) + len(data_set["train"]))
        print("train",len(data_set["train"]))
        print("test",len(data_set["test"]))

        for sample_train in data_set["train"]:
            for sample_test in data_set["test"]:
                if sample_test["consult_id"] == sample_train["consult_id"]:
                    print(sample_test)
                    print(sample_train)

        pickle.dump(file=open(dump_file_name,"wb"), obj=data_set)


    def __keep_sample_or_not__(self, goal):
        disease_tag = goal["disease_tag"]

        # The number of explicit and implicit symptoms.
        if disease_tag in ["小儿腹泻"]:
            keep_or_not = False
            if len(goal["goal"]["explicit_inform_slots"].keys()) >= 0 and \
                    len(goal["goal"]["implicit_inform_slots"].keys()) >= 1:
                keep_or_not = True
        elif disease_tag in ["小儿消化不良"]:
            keep_or_not = False
            if len(goal["goal"]["explicit_inform_slots"].keys()) >= 0 and \
                    len(goal["goal"]["implicit_inform_slots"].keys()) >= 1:
                keep_or_not = True
        elif disease_tag in ["小儿支气管炎"]:
            keep_or_not = False
            if len(goal["goal"]["explicit_inform_slots"].keys()) >= 1 and \
                    len(goal["goal"]["implicit_inform_slots"].keys()) >= 1:
                keep_or_not = True
        # for 上呼吸道感染
        elif disease_tag == "上呼吸道感染":
            keep_or_not = False
            if len(goal["goal"]["explicit_inform_slots"].keys()) >= 0 and \
                    len(goal["goal"]["implicit_inform_slots"].keys()) >= 2:
                keep_or_not = True
                for symptom in goal["goal"]["implicit_inform_slots"].keys():
                    # if symptom in ["腹泻","黄绿稀溏","稀便","脱水","喘鸣","呼吸不畅"]:
                    # if symptom in ["腹泻","黄绿稀溏","稀便", "涨肚","咳嗽"]:
                    if symptom in ["腹泻", "涨肚", "咳嗽", "肺纹理增粗"]:
                    # if symptom in ["腹泻","咳嗽", "肺纹理增粗","涨肚"]:
                        keep_or_not = False
                        break
        return True


if __name__ == "__main__":
    # goal_set,symptom_set, disease_symptom
    # goal_set_file = './../dialogue_system/data/4_diseases/both/goal_set.p'
    goal_set_file = './../dialogue_system/data/dataset/label/goal_set.p'
    slot_set_file = './../dialogue_system/data/dataset/label/slot_set.p'
    disease_symptom_file = './../dialogue_system/data/dataset/label/disease_symptom.p'
    save_path = "./../dialogue_system/data/found_dataset/"
    goal_set = pickle.load(open(goal_set_file,"rb"))
    slot_set = pickle.load(open(slot_set_file,"rb"))
    disease_symptom_set = pickle.load(open(disease_symptom_file,"rb"))
    # print(goal_set)
    finder = Finder(goal_set=goal_set,symptom_set=slot_set,disease_symptom=disease_symptom_set,k_fold=5)

    gap = 0.0
    save_set = False
    save_count = 0
    index = 0
    while save_count <= 500:
        index += 1
        gap, save_set = finder.find(save_path=save_path)
        print("finding...", index, "gap:",gap)
        if save_set:
            save_count += 1
