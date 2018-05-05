# -*- coding: utf-8 -*-

import pickle


class GoalReader(object):
    def __init__(self):
        pass

    def load(self, goal_file):
        self.goal_file = goal_file
        goal_set = pickle.load(open(goal_file, 'rb'))
        self.goal_set = goal_set['train'] + goal_set['test'] + goal_set['validate']

    def dump(self, slot_file, disease_symptom_file):
        self.slot_set = {}
        self.disease_symptom = {}
        for goal in self.goal_set:
            disease = goal['disease_tag']
            self.disease_symptom.setdefault(disease, {'index':len(self.disease_symptom), 'symptom':dict()})
            slot_set = goal['goal']['explicit_inform_slots']
            slot_set.update(goal['goal']['implicit_inform_slots'])
            for slot, value in slot_set.items():
                self.slot_set.setdefault(slot, len(self.slot_set))
                self.disease_symptom[disease]['symptom'].setdefault(slot,0)
                self.disease_symptom[disease]['symptom'][slot] += 1
        self.slot_set['disease'] = len(self.slot_set)
        # for key in self.disease_symptom.keys():
        #     self.disease_symptom[key]['symptom'] = list(self.disease_symptom[key]['symptom'])

        pickle.dump(obj=self.slot_set, file=open(slot_file, 'wb'))
        pickle.dump(obj=self.disease_symptom, file=open(disease_symptom_file, 'wb'))
        print(len(self.slot_set), self.slot_set)
        for key, value in self.disease_symptom.items():
            print(key, len(value['symptom']))
        print(len(self.disease_symptom), self.disease_symptom)


if __name__ == '__main__':
    path = './../../resources/label/used/'
    reader = GoalReader()
    reader.load(path + 'goal_set.p')
    reader.dump(path + 'slot_set.p', path + 'disease_symptom.p')