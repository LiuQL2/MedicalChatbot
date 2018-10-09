# -*- coding:utf-8 -*-

import time
import argparse
import pickle
import sys, os
sys.path.append(os.getcwd().replace("src/classifier/run",""))

from src.classifier.symptom_as_feature.symptom_classifier import SymptomClassifier
from src.classifier.self_report_as_feature.report_classifier import ReportClassifier


parser = argparse.ArgumentParser()

parser.add_argument("--goal_set", dest="goal_set", type=str, default="./../../dialogue_system/data/dataset/label/goal_set.p", help='path and filename of user goal')
parser.add_argument("--slot_set", dest="slot_set", type=str, default='./../../dialogue_system/data/dataset/label/slot_set.p', help='path and filename of the slots set')
parser.add_argument("--disease_symptom", dest="disease_symptom", type=str, default="./../../dialogue_system/data/dataset/label/disease_symptom.p", help="path and filename of the disease_symptom file")


parser.add_argument("--explicit_number", dest="explicit_number", type=int, default=0, help="the number of explicit symptoms of used sample")
parser.add_argument("--implicit_number", dest="implicit_number", type=int, default=0, help="the number of implicit symptoms of used sample")


parser.add_argument("--batch_size", dest="batch_size",type=int, default=32, help="the batch size for training.")
parser.add_argument("--hidden_size", dest="hidden_size",type=int, default=40, help="the hidden size of classifier.")
parser.add_argument("--train_feature", dest="train_feature", type=str, default="ex&im", help="only use explicit symptom for classification? ex:yes, ex&im:no")
parser.add_argument("--test_feature", dest="test_feature", type=str, default="ex&im", help="only use explicit symptom for testing? ex:yes, ex&im:no")
parser.add_argument("--checkpoint_path",dest="checkpoint_path", type=str, default="./../model/checkpoint/", help="the folder where models save to, ending with /.")
parser.add_argument("--saved_model", dest="saved_model", type=str, default="./../model/dqn/checkpoint_d4_agt1_dqn1/model_d4_agent1_dqn1_s0.602_r17.036_t4.326_wd0.0_e214.ckpt")
parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.2,help="the learning rate when training the model.")

args = parser.parse_args()
parameter = vars(args)

def run():
    slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
    goal_set = pickle.load(file=open(parameter["goal_set"], "rb"))
    disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))
    hidden_size = parameter.get("hidden_size")

    print("##"*30+"\nSymptom as features\n"+"##"*30)
    classifier = SymptomClassifier(goal_set=goal_set,symptom_set=slot_set,disease_symptom=disease_symptom,hidden_size=hidden_size,parameter=parameter,k_fold=5)
    classifier.train_sklearn_svm()
    print(classifier.disease_sample_count)
    # classifier.sample_to_file("./../data/goal_set.json")
    # classifier.dump_goal_set("/Volumes/LIUQL/dataset/goal_set_6.p")


    # print("##"*30+"\nSelf-report as features\n"+"##"*30)
    # data_file = "./../../../resources/top_self_report_extracted_symptom.csv"
    # stop_words = "./../data/stopwords.txt"
    # report_classifier = ReportClassifier(stop_words=stop_words,data_file=data_file)
    # report_classifier.train_tf()
    # report_classifier.evaluate_tf()
    # report_classifier.train_sklearn_svm()
    # report_classifier.evaluate_sklearn_svm()


if __name__ == "__main__":
    run()