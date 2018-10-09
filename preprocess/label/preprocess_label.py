# -*- coding:utf-8 -*-
"""
将txt的action和json的action、症状都转化为只包含action, symptom的list，并持久化到文件保存，
后期使用中直接调用持久化的文件即可。这里每一个symptom都作为一个slot进行处理。
"""
import pickle
import json
import random
import copy


class ActionDumper(object):
    """
    处理action文件，保存成list并进行持久化处理。
    """
    def __init__(self, action_set_file):
        self.file_name = action_set_file

    def dump(self, dump_file_name):
        data_file = open(self.file_name, "r")
        action_set = []
        for line in data_file:
            action_set.append(line.replace("\n",""))
        data_file.close()
        action_set_dict = {}
        for index in range(0, len(action_set), 1):
            action_set_dict[action_set[index]] = index
        pickle.dump(file=open(dump_file_name,"wb"), obj=action_set_dict)


class SlotDumper(object):
    """
    处理disease_symptom文件，将里面的每一个symptom作为一个slot处理，进行持久化。
    """
    def __init__(self, slots_file, hand_crafted_symptom=True):
        self.file_name = slots_file
        self.hand_crafted_symptom = hand_crafted_symptom

    def dump(self, slot_dump_file_name, disease_dump_file_name):
        self._load_slot()
        self.slot_set.add("disease")
        # self.slot_set.add("taskcomplete")

        slot_set = list(self.slot_set)
        slot_set_dict = {}
        for index in range(0, len(slot_set), 1):
            slot_set_dict[slot_set[index]] = index
        pickle.dump(file=open(slot_dump_file_name,"wb"), obj=slot_set_dict)
        pickle.dump(file=open(disease_dump_file_name, "wb"), obj=self.disease_symptom)

    def _load_slot(self):
        self.slot_set = set()
        self.disease_symptom = {}
        data_file = open(file=self.file_name, mode="r",encoding="utf-8")
        if self.hand_crafted_symptom == True:
            index = 0
            for line in data_file:
                line = json.loads(line)
                self.disease_symptom[line["name"]] = {}
                self.disease_symptom[line["name"]]["index"] = index
                self.disease_symptom[line["name"]]["symptom"] = list(line["symptom"].keys())
                for key in line["symptom"].keys():
                    self.slot_set.add(key)
                index += 1
        else:
            index = 0
            for line in data_file:
                line = json.loads(line)
                self.disease_symptom[line["name"]] = {}
                self.disease_symptom[line["name"]]["index"] = index
                self.disease_symptom[line["name"]]["symptom"] = line["symptom"]
                for symptom in line["symptom"]:
                    self.slot_set.add(symptom)
                index += 1

        data_file.close()


class GoalDumper(object):
    def __init__(self, goal_file):
        self.file_name = goal_file
        self.slot_set = set()
        self.disease_symptom = {}

    def dump(self, dump_file_name, train=0.8, test=0.2, validate=0.0):
        assert (train*100+test*100+validate*100==100), "train + test + validate not equals to 1.0."
        self.goal_set = []
        data_file = open(file=self.file_name, mode="r")
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

            self.goal_set.append(line)
        data_file.close()
        goal_number = len(self.goal_set)
        data_set = {
            "train":[],
            "test":[],
            "validate":[]
        }

        for goal in self.goal_set:
            random_float = random.random()
            if random_float <= train:
                data_set["train"].append(goal)
            elif train < random_float and random_float <= train+test:
                data_set["test"].append(goal)
            else:
                data_set["validate"].append(goal)

            for slot, value in goal["goal"]["explicit_inform_slots"].items():
                if value == False: print(goal)
                break
            for slot, value in goal["goal"]["implicit_inform_slots"].items():
                if value == False: print(goal)
                break

            # for slot.
            for symptom in goal["goal"]["explicit_inform_slots"].keys(): self.slot_set.add(symptom)
            for symptom in goal["goal"]["implicit_inform_slots"].keys(): self.slot_set.add(symptom)

            # for disease_symptom
            key_num = len(self.disease_symptom.keys())
            disease = goal['disease_tag']
            self.disease_symptom.setdefault(disease,{'index':key_num,'symptom':dict()})
            for symptom in goal["goal"]["explicit_inform_slots"].keys():
                self.disease_symptom[disease]['symptom'].setdefault(symptom, 0)
                self.disease_symptom[disease]['symptom'][symptom] += 1
            for symptom in goal["goal"]["implicit_inform_slots"].keys():
                self.disease_symptom[disease]['symptom'].setdefault(symptom, 0)
                self.disease_symptom[disease]['symptom'][symptom] += 1

        pickle.dump(file=open(dump_file_name,"wb"), obj=data_set)

    def dump_slot(self,slot_file):
        slot_set_dict = {}
        slot_set = list(self.slot_set)
        for index in range(0, len(slot_set), 1):
            slot_set_dict[slot_set[index]] = index
        slot_set_dict['disease'] = index + 1
        print(slot_set_dict)
        pickle.dump(file=open(slot_file,"wb"),obj=slot_set_dict)

    def dump_disease_symptom(self, disease_symptom_file):
        print(self.disease_symptom)
        pickle.dump(file=open(disease_symptom_file,'wb'), obj=self.disease_symptom)





if __name__ == "__main__":
    # Action
    # action_file = "./../../../resources/action_set.txt"
    # action_dump_file = "./../data/action_set.p"
    #
    # action_dumper = ActionDumper(action_set_file=action_file)
    # action_dumper.dump(dump_file_name=action_dump_file)

    # Slots.
    # slots_file = "./../../../resources/top_disease_symptom_aligned.json"
    # slots_dump_file = "./../data/slot_set.p"
    # disease_dump_file = "./../data/disease_symptom.p"
    # slots_dumper = SlotDumper(slots_file=slots_file)
    # slots_dumper.dump(slot_dump_file_name=slots_dump_file,disease_dump_file_name=disease_dump_file)

    # Goal
    goal_file = "./../../resources/label/goal2_normal_filter.json"
    goal_dump_file = "./../../resources/label/goal_set.p"
    slots_dump_file = "./../../resources/label/slot_set.p"
    disease_dump_file = "./../../resources/label/disease_symptom.p"
    goal_dumper = GoalDumper(goal_file=goal_file)
    goal_dumper.dump(dump_file_name=goal_dump_file)
    goal_dumper.dump_slot(slots_dump_file)
    goal_dumper.dump_disease_symptom(disease_dump_file)