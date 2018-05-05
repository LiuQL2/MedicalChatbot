# -*- coding: utf-8 -*-

import json
import copy
import csv


class Frequency(object):
    """
    统计每个症状出现的次数，每种疾病下每个症状出现的次数。
    """
    def __init__(self):
        pass

    def load(self, goal_file, symptom_frequency_file, disease_symptom_frequency_file):
        data_file = open(file=goal_file, mode='r')
        symptom_frequency_dict = {}
        disease_symptom_frequency_dict = {}
        for line in data_file:
            line = json.loads(line)
            disease_symptom_frequency_dict.setdefault(line['disease_tag'], dict())
            for symptom, value in line['goal']['implicit_inform_slots'].items():
                symptom_frequency_dict.setdefault(symptom, 0)
                symptom_frequency_dict[symptom] += 1
                disease_symptom_frequency_dict[line['disease_tag']].setdefault(symptom, 0)
                disease_symptom_frequency_dict[line['disease_tag']][symptom] += 1

            for symptom, value in line['goal']['explicit_inform_slots'].items():
                symptom_frequency_dict.setdefault(symptom, 0)
                symptom_frequency_dict[symptom] += 1
                disease_symptom_frequency_dict[line['disease_tag']].setdefault(symptom, 0)
                disease_symptom_frequency_dict[line['disease_tag']][symptom] += 1
        symptom_file = open(file=symptom_frequency_file, mode='w', encoding='utf8')
        disease_file = open(file=disease_symptom_frequency_file, mode='w',encoding='utf8')
        symptom_writer = csv.writer(symptom_file)
        disease_writer = csv.writer(disease_file)

        for symptom, count in symptom_frequency_dict.items():
            symptom_writer.writerow([symptom, count])
        for disease, symptom_count in disease_symptom_frequency_dict.items():
            for symptom, count in symptom_count.items():
                disease_writer.writerow([disease, symptom, count])
        symptom_file.close()
        disease_file.close()
        data_file.close()


class Normalize(object):
    """
    人工对标注得到的症状进行了部分归一。
    """
    def __init__(self, normalize_file):
        self.spoken_normal = {}
        data_file = open(normalize_file, mode='r', encoding='utf8')
        reader = csv.reader(data_file)
        for line in reader:
            line = line[0].split('\t')
            self.spoken_normal[line[0]] = line[1]
        data_file.close()

    def load(self, goal_file):
        data_file = open(goal_file, 'r', encoding='utf8')
        new_file = open(goal_file.split('.json')[0] + '_normal.json', 'w', encoding='utf8')
        for line in data_file:
            line = json.loads(line)
            temp_line = copy.deepcopy(line)
            for symptom, value in temp_line['goal']['implicit_inform_slots'].items():
                if symptom in self.spoken_normal.keys():
                    line['goal']['implicit_inform_slots'][self.spoken_normal[symptom]] = value
                    line['goal']['implicit_inform_slots'].pop(symptom)
            for symptom, value in temp_line['goal']['explicit_inform_slots'].items():
                if symptom in self.spoken_normal.keys():
                    line['goal']['explicit_inform_slots'][self.spoken_normal[symptom]] = value
                    line['goal']['explicit_inform_slots'].pop(symptom)
            new_file.write(json.dumps(line) + '\n')
        data_file.close()
        new_file.close()


class FilterFrequency(object):
    """
    根据频率过滤症状。
    """
    def __init__(self, symptom_frequency_file, threshold = 1):
        self.threshold = threshold
        data_file = open(symptom_frequency_file, 'r', encoding='utf-8')
        reader = csv.reader(data_file)
        self.symptom_frequency = {}
        for line in reader:
            self.symptom_frequency[line[0]] = int(line[1])
        data_file.close()

    def load(self, goal_file):
        data_file = open(goal_file, 'r', encoding='utf8')
        new_file = open(goal_file.split('.json')[0] + '_filter.json', 'w', encoding='utf8')
        for line in data_file:
            line = json.loads(line)
            temp_line = copy.deepcopy(line)
            for symptom, value in temp_line['goal']['implicit_inform_slots'].items():
                if self.symptom_frequency[symptom] <= self.threshold:
                    line['goal']['implicit_inform_slots'].pop(symptom)
            for symptom, value in temp_line['goal']['explicit_inform_slots'].items():
                if self.symptom_frequency[symptom] <= self.threshold:
                    line['goal']['explicit_inform_slots'].pop(symptom)
            new_file.write(json.dumps(line) + '\n')
        data_file.close()
        new_file.close()


class FirstRun(object):
    """
    去除疾病名称、症状名词里面的空格。
    """
    def read(self, goal_file):
        data_file = open(file=goal_file, mode="r")
        new_file = open(file=goal_file.split('.json')[0] + '_2.json', mode='w')
        for line in data_file:
            line = json.loads(line)
            line['disease_tag'] = line['disease_tag'].replace(' ', '')
            temp_line = copy.deepcopy(line)
            for symptom, value in temp_line['goal']['implicit_inform_slots'].items():
                line['goal']['implicit_inform_slots'].pop(symptom)
                line['goal']['implicit_inform_slots'][symptom.replace(' ', '')] = value

            for symptom, value in temp_line['goal']['explicit_inform_slots'].items():
                line['goal']['explicit_inform_slots'].pop(symptom)
                line['goal']['explicit_inform_slots'][symptom.replace(' ', '')] = value
            new_file.write(json.dumps(line) + '\n')
        data_file.close()
        new_file.close()


if __name__ == '__main__':
    goal_file = './../../resources/label/goal2_normal_filter.json'
    # goal_file = './../../resources/label/goal2_normal.json'
    # goal_file = './../../resources/label/goal2.json'
    symptom_file = './../../resources/label/symptom_frequency.csv'
    disease_file = './../../resources/label/disease_frequency.csv'

    # first = FirstRun()
    # first.read(goal_file)

    #
    # normal_file = './../../resources/label/症状归一手动.csv'
    # normal = Normalize(normal_file)
    # normal.load(goal_file)

    frequency = Frequency()
    frequency.load(goal_file,symptom_file, disease_file)


    # filter = FilterFrequency(symptom_file,threshold=9)
    # filter.load(goal_file)
