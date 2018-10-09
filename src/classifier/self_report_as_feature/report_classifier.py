# -*- coding:utf-8 -*-
"""
使用主诉内容进行分类，word embedding of n-grams.
"""
import copy
import tensorflow as tf
import time
from sklearn import svm
import os, sys
import jieba
import csv
import numpy as np
import random
sys.path.append(os.getcwd().replace("src/classifier/self_report_as_feature",""))


class ReportClassifier(object):
    def __init__(self, stop_words, data_file):
        self.corpus = Corpus(stop_words=stop_words)
        self.corpus.load_data(data_file=data_file)
        self.clf = svm.SVC(decision_function_shape='ovo')
        self.__build_tf_model()

    def train_sklearn_svm(self):
        print("fitting svm model...")
        self.clf.fit(X = self.corpus.data_set["train"]["x"],y=self.corpus.data_set["train"]["y"])

    def evaluate_sklearn_svm(self):
        predict = self.clf.predict(X=self.corpus.data_set["test"]["x"])
        count = 0
        for index in range(0,len(predict),1):
            if predict[index] == self.corpus.data_set["test"]["y"][index]:
                count += 1
        print("accuracy of sklearn svm:", float(count)/len(predict))

    def train_tf(self):
        data = self.corpus.data_set
        train_input_fn = self.__get_input_fn(data["train"], batch_size=64)
        self.estimator.fit(input_fn=train_input_fn, steps=2000)

    def evaluate_tf(self):
        data = self.corpus.data_set
        eval_input_fn = self.__get_input_fn(data["test"], batch_size=5000)
        eval_metrics = self.estimator.evaluate(input_fn=eval_input_fn, steps=1)
        print("result of tf:",eval_metrics)

    def __build_tf_model(self):
        self.feature = tf.contrib.layers.real_valued_column('word_index', dimension=self.corpus.vocabulary_size)
        self.optimizer = tf.train.FtrlOptimizer(learning_rate=50.0, l2_regularization_strength=0.001)
        self.kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(input_dim=self.corpus.vocabulary_size, output_dim=1000, stddev=5.0, name='rffm')
        kernel_mappers = {self.feature: [self.kernel_mapper]}
        self.estimator = tf.contrib.kernel_methods.KernelLinearClassifier(
            n_classes=len(self.corpus.disease_to_index), optimizer=self.optimizer, kernel_mappers=kernel_mappers)

    def __get_input_fn(self,dataset_split, batch_size, capacity=15000, min_after_dequeue=1000):
        def _input_fn():
            xs = np.array(dataset_split["x"]).astype(np.float32)
            ys = np.array(dataset_split["y"])
            report_batch, labels_batch = tf.train.shuffle_batch(
                # tensors=[dataset_split.images, dataset_split.labels.astype(np.int32)],
                tensors=[xs, ys.astype(np.int32)],
                batch_size=batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                enqueue_many=True,
                num_threads=4)
            features_map = {'word_index': report_batch}
            return features_map, labels_batch
        return _input_fn


class Corpus(object):
    def __init__(self, stop_words):
        self.max_document_length = 0
        self.data_set = {"train":{"x":[],"y":[],"word_index":[]}, "test":{"x":[],"y":[],"word_index":[]}}
        self.stop_words = self._load_stop_words(stop_words_file=stop_words)

    def load_data(self,data_file, train=0.8, test=0.2):
        """
        加载数据并分词
        :param data_file: 包含主诉内容的文件
        :return:
        """
        assert (train*100+test*100==100), "train + test + validate not equals to 1.0."
        # Mapping disease to index and index to disease.
        print("disease to index...")
        data_reader = csv.reader(open(data_file, "r",encoding="utf-8"))
        self.disease_to_index = {}
        self.index_to_disease = {}
        index = 0
        for line in data_reader:
            disease = line[5]
            if disease not in self.disease_to_index.keys() and disease != "小儿发热":
                self.disease_to_index[disease] = index
                self.index_to_disease[index] = disease
                index += 1

        # Word segmentation.
        print("word segmentation...")
        data_set = []
        data_reader = csv.reader(open(data_file, "r",encoding="utf-8"))
        all_word_list = []
        for line in data_reader:
            if line[5] == "小儿发热":
                continue
            disease_index = self.disease_to_index[line[5]]
            # seg_list = jieba.cut(line[6], cut_all=False)
            seg_list = jieba.cut_for_search(line[6])

            word_list = []
            for word in seg_list:
                if word not in self.stop_words:
                    word_list.append(word)
                    all_word_list.append(word)
            if len(word_list) > self.max_document_length:
                self.max_document_length = len(word_list)
            data_set.append({"disease":disease_index,"text":" ".join(word_list)})
            # print(" ".join(word_list))
        # word to index.
        print("word to index...")
        self.vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length=self.max_document_length,min_frequency=3)
        self.vocab_processor.fit(all_word_list)
        self.vocabulary_size = len(self.vocab_processor.vocabulary_)

        print("preparing dataset...")
        for data in data_set:
            data["text"] = next(self.vocab_processor.transform([data["text"]])).tolist()
            text_rep = np.zeros(self.vocabulary_size)
            for index in data["text"]:
                if index >= 1:
                    text_rep[index-1] += 1.0

            random_float = random.random()
            if random_float <= train:
                self.data_set["train"]["x"].append(copy.deepcopy(text_rep))
                self.data_set["train"]["word_index"].append(copy.deepcopy(data["text"]))
                self.data_set["train"]["y"].append(copy.deepcopy(data["disease"]))
            else:
                self.data_set["test"]["x"].append(copy.deepcopy(text_rep))
                self.data_set["test"]["word_index"].append(copy.deepcopy(data["text"]))
                self.data_set["test"]["y"].append(copy.deepcopy(data["disease"]))

    def _load_stop_words(self,stop_words_file):
        """
        Load stop words.
        :param stop_words_file: the path of file that contains stop words, on word for each line.
        :return: dictionary of stop words, key: word, value: word.
        """
        stop_words = [line.strip() for line in open(stop_words_file, encoding="utf-8").readlines()]
        temp_dict = {}
        for word in stop_words:
            temp_dict[word] = word
        return temp_dict


if __name__ == "__main__":
    data_file = "./../../../resources/top_self_report_extracted_symptom.csv"
    stop_words = "./data/stopwords.txt"
    classifier = ReportClassifier(stop_words=stop_words,data_file=data_file)
