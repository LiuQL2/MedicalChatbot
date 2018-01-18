# -*- coding:utf-8 -*-

import time
import argparse
import pickle
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/run",""))

from src.classifier.symptom_as_feature.symptom_as_feature import SymptomClassifier


parser = argparse.ArgumentParser()
parser.add_argument("--slot_set", dest="slot_set", type=str, default='./../../dialogue_system/data/slot_set.p', help='path and filename of the slots set')
parser.add_argument("--goal_set", dest="goal_set", type=str, default='./../../dialogue_system/data/goal_set.p', help='path and filename of user goal')
parser.add_argument("--batch_size", dest="batch_size",type=int, default=32, help="the batch size for training.")
parser.add_argument("--hidden_size", dest="hidden_size",type=int, default=40, help="the hidden size of classifier.")
parser.add_argument("--disease_symptom", dest="disease_symptom", type=str, default="./../../dialogue_system/data/disease_symptom.p", help="path and filename of the disease_symptom file")
parser.add_argument("--explicit_only", dest="explicit_only", type=int, default=1, help="only use explicit symptom for classification? 1:yes, 0:no")
parser.add_argument("--checkpoint_path",dest="checkpoint_path", type=str, default="./../model/checkpoint/", help="the folder where models save to, ending with /.")
parser.add_argument("--saved_model", dest="saved_model", type=str, default="./../model/checkpoint/model_s0.89_r735.0_t7.08_wd1.55_e20.ckpt")
parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.1,help="the learning rate when training the model.")

args = parser.parse_args()
parameter = vars(args)

def run():
    slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
    goal_set = pickle.load(file=open(parameter["goal_set"], "rb"))
    disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))
    hidden_size = parameter.get("hidden_size")
    print(slot_set)
    print(disease_symptom)

    classfier = SymptomClassifier(goal_set=goal_set,symptom_set=slot_set,disease_symptom=disease_symptom,hidden_size=hidden_size,parameter=parameter)
    classfier.train()
    classfier.evaluate()


if __name__ == "__main__":
    run()