
import pickle

class Goal2Slot(object):
    def __init__(self):
        pass

    def load_goal(self,goal_file):
        slot_set = set()
        goal_set = pickle.load(open(goal_file,"rb"))
        for key in goal_set.keys():
            for goal in goal_set[key]:
                for symptom in goal["goal"]["explicit_inform_slots"].keys():
                    slot_set.add(symptom)
                for symptom in goal["goal"]["implicit_inform_slots"].keys():
                    slot_set.add(symptom)
        self.slot_set = list(slot_set)
        print(len(self.slot_set))


if __name__ == "__main__":
    goal_file = "./../data/dataset/1200/1/goal_set.p"
    goal2slot = Goal2Slot()
    goal2slot.load_goal(goal_file)
