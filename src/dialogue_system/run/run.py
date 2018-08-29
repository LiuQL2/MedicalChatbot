# -*- coding:utf-8 -*-

import time
import argparse
import pickle
import sys, os
import json
sys.path.append(os.getcwd().replace("src/dialogue_system/run",""))

from src.dialogue_system.dialogue_manager import DialogueManager
from src.dialogue_system.agent import AgentRandom
from src.dialogue_system.agent import AgentDQN
from src.dialogue_system.agent import AgentRule
from src.dialogue_system.agent import AgentActorCritic
from src.dialogue_system.user_simulator import UserRule as User
from src.dialogue_system import dialogue_configuration

from src.dialogue_system.run import RunningSteward

disease_number = 4

parser = argparse.ArgumentParser()
parser.add_argument("--disease_number", dest="disease_number", type=int,default=disease_number,help="the number of disease.")
parser.add_argument("--device_for_tf", dest="device_for_tf", type=str, default="/device:GPU:3", help="the device for tensorflow running on.")

# TODO: simulation configuration
parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default=1500, help="the number of simulate epoch.")
parser.add_argument("--epoch_size", dest="epoch_size", type=int, default=50, help="the size of each simulate epoch.")
parser.add_argument("--evaluate_epoch_number", dest="evaluate_epoch_number", type=int, default=2000, help="the size of each simulate epoch when evaluation.")
parser.add_argument("--experience_replay_pool_size", dest="experience_replay_pool_size", type=int, default=20000, help="the size of experience replay.")
parser.add_argument("--hidden_size_dqn", dest="hidden_size_dqn", type=int, default=300, help="the hidden_size of DQN.")
parser.add_argument("--warm_start", dest="warm_start",type=int, default=1, help="use rule policy to fill the experience replay buffer at the beginning, 1:True; 0:False")
parser.add_argument("--warm_start_epoch_number", dest="warm_start_epoch_number", type=int, default=20, help="the number of epoch of warm starting.")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=30, help="the batch size when training.")
parser.add_argument("--log_dir", dest="log_dir", type=str, default="./../../../log/", help="directory where event file of training will be written, ending with /")
parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.1, help="the greedy of DQN")
parser.add_argument("--gamma", dest="gamma", type=float, default=1.0, help="The discount factor of immediate reward.")
parser.add_argument("--train_mode", dest="train_mode", type=int, default=1, help="training mode? True:1 or False:0")

# TODO: Save model, performance and dialogue content ? And what is the path if yes?
parser.add_argument("--save_performance",dest="save_performance", type=int, default=0, help="save the performance? 1:Yes, 0:No")
parser.add_argument("--performance_save_path",dest="performance_save_path", type=str, default="./../model/dqn/learning_rate04/", help="the folder where learning rate save to, ending with /.")
parser.add_argument("--save_model", dest="save_model", type=int, default=1,help="save model? 1:Yes,0:No")
parser.add_argument("--checkpoint_path",dest="checkpoint_path", type=str, default="./../model/dqn/checkpoint/", help="the folder where models save to, ending with /.")
parser.add_argument("--saved_model", dest="saved_model", type=str, default="./../model/dqn/checkpoint/checkpoint_d4_agt1_dqn1/model_d4_agent1_dqn1_s0.619_r18.221_t4.266_wd0.0_e432.ckpt")
parser.add_argument("--dialogue_file", dest="dialogue_file", type=str, default="./../data/dialogue_output/dialogue_file.txt", help="the file that used to save dialogue content.")
parser.add_argument("--save_dialogue", dest="save_dialogue", type=int, default=0, help="save the dialogue? 1:Yes, 0:No")


parser.add_argument("--run_id", dest='run_id', type=int, default=1, help='the id of this running.')

# TODO: user configuration.
parser.add_argument("--allow_wrong_disease", dest="allow_wrong_disease", type=int, default=0, help="Allow the agent to inform wrong disease? 1:Yes, 0:No")

# TODO: Learning rate for actor-critic and dqn.
parser.add_argument("--dqn_learning_rate", dest="dqn_learning_rate", type=float, default=0.001, help="the learning rate of dqn.")
parser.add_argument("--actor_learning_rate", dest="actor_learning_rate", type=float, default=0.001, help="the learning rate of actor")
parser.add_argument("--critic_learning_rate", dest="critic_learning_rate", type=float, default=0.001, help="the learning rate of critic")
parser.add_argument("--trajectory_pool_size", dest="trajectory_pool_size", type=int, default=48, help="the size of trajectory pool")

# TODO: the number condition of explicit symptoms and implicit symptoms in each user goal.
parser.add_argument("--explicit_number", dest="explicit_number", type=int, default=0, help="the number of explicit symptoms of used sample")
# parser.add_argument("--implicit_number", dest="implicit_number", type=int, default=1, help="the number of implicit symptoms of used sample")
parser.add_argument("--implicit_number", dest="implicit_number", type=int, default=0, help="the number of implicit symptoms of used sample")

# TODO: agent to use.
parser.add_argument("--agent_id", dest="agent_id", type=int, default=1, help="the agent to be used:{0:AgentRule, 1:AgentDQN, 2:AgentActorCritic, 3:AgentRandom}")
parser.add_argument("--dqn_id", dest="dqn_id", type=int, default=1, help="the dqn to be used in agent:{0:initial dqn of qianlong, 1:dqn with one layer of qianlong, 2:dqn with two layers of qianlong, 3:dqn of Baolin.}")

# TODO: goal set, slot set, action set.
# for 7 diseases.
if disease_number == 7:
    parser.add_argument("--action_set", dest="action_set", type=str, default='./../data/7_diseases/action_set.p',help='path and filename of the action set')
    parser.add_argument("--slot_set", dest="slot_set", type=str, default='./../data/7_diseases/slot_set.p',help='path and filename of the slots set')
    parser.add_argument("--goal_set", dest="goal_set", type=str, default='./../data/7_diseases/goal_set.p',help='path and filename of user goal')
    parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,default="./../data/7_diseases/disease_symptom.p",help="path and filename of the disease_symptom file")
    parser.add_argument("--max_turn", dest="max_turn", type=int, default=26, help="the max turn in one episode.")
    parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=325, help="the input_size of DQN.")
elif disease_number == 10:
    # for 10 diseases.
    parser.add_argument("--action_set", dest="action_set", type=str, default='./../data/10_diseases/action_set.p',help='path and filename of the action set')
    parser.add_argument("--slot_set", dest="slot_set", type=str, default='./../data/10_diseases/slot_set.p',help='path and filename of the slots set')
    parser.add_argument("--goal_set", dest="goal_set", type=str, default='./../data/10_diseases/goal_set.p',help='path and filename of user goal')
    parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,default="./../data/10_diseases/disease_symptom.p",help="path and filename of the disease_symptom file")
    parser.add_argument("--max_turn", dest="max_turn", type=int, default=42, help="the max turn in one episode.")
    parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=494, help="the input_size of DQN.")
elif disease_number == 8:
    # for 10 diseases.
    parser.add_argument("--action_set", dest="action_set", type=str, default='./../data/8_diseases/action_set.p',help='path and filename of the action set')
    parser.add_argument("--slot_set", dest="slot_set", type=str, default='./../data/8_diseases/slot_set.p',help='path and filename of the slots set')
    parser.add_argument("--goal_set", dest="goal_set", type=str, default='./../data/8_diseases/goal_set.p',help='path and filename of user goal')
    parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,default="./../data/8_diseases/disease_symptom.p",help="path and filename of the disease_symptom file")
    parser.add_argument("--max_turn", dest="max_turn", type=int, default=22, help="the max turn in one episode.")
    parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=480, help="the input_size of DQN.")
elif disease_number == 4:
    # for 4 diseases.
    # max_turn = 22
    max_turn = 22
    parser.add_argument("--action_set", dest="action_set", type=str, default='./../data/dataset/label/action_set.p',help='path and filename of the action set')
    parser.add_argument("--slot_set", dest="slot_set", type=str, default='./../data/dataset/label/slot_set.p',help='path and filename of the slots set')
    parser.add_argument("--goal_set", dest="goal_set", type=str, default='./../data/dataset/label/goal_set.p',help='path and filename of user goal')
    parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,default="./../data/dataset/label/disease_symptom.p",help="path and filename of the disease_symptom file")
    parser.add_argument("--max_turn", dest="max_turn", type=int, default=max_turn, help="the max turn in one episode.")
    # parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=max_turn+137, help="the input_size of DQN.")
    parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=max_turn + 357, help="the input_size of DQN.")
    parser.add_argument("--reward_for_not_come_yet", dest="reward_for_not_come_yet", type=float,default=-1)
    parser.add_argument("--reward_for_success", dest="reward_for_success", type=float,default=2*max_turn)
    parser.add_argument("--reward_for_fail", dest="reward_for_fail", type=float,default=-1.0*max_turn)
    parser.add_argument("--reward_for_inform_right_symptom", dest="reward_for_inform_right_symptom", type=float,default=-1)
    parser.add_argument("--minus_left_slots", dest="minus_left_slots", type=int, default=0,help="Reward for success minus left slots? 1:Yes, 0:No")


args = parser.parse_args()
parameter = vars(args)

agent_id = parameter.get("agent_id")
dqn_id = parameter.get("dqn_id")
disease_number = parameter.get("disease_number")
max_turn = parameter.get("max_turn")

if agent_id == 1:
    checkpoint_path = "./../model/dqn/checkpoint04/checkpoint_d" + str(disease_number) + "_agt" + str(agent_id) + "_dqn" + str(dqn_id) + "_T" + str(max_turn) +  "/"
else:
    checkpoint_path = "./../model/dqn/checkpoint04/checkpoint_d" + str(disease_number) + "_agt" + str(agent_id) + "_T" + str(max_turn) +  "/"
print(json.dumps(parameter, indent=2))
time.sleep(1)


def run():
    slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
    action_set = pickle.load(file=open(parameter["action_set"], "rb"))
    disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))
    steward = RunningSteward(parameter=parameter,checkpoint_path=checkpoint_path)

    warm_start = parameter.get("warm_start")
    warm_start_epoch_number = parameter.get("warm_start_epoch_number")
    train_mode = parameter.get("train_mode")
    agent_id = parameter.get("agent_id")
    simulate_epoch_number = parameter.get("simulate_epoch_number")

    # Warm start.
    if warm_start == 1 and train_mode == 1:
        print("warm starting...")
        agent = AgentRule(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
        steward.warm_start(agent=agent,epoch_number=warm_start_epoch_number)
    # exit()
    if agent_id == 1:
        agent = AgentDQN(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
    elif agent_id == 2:
        agent = AgentActorCritic(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
    elif agent_id == 3:
        agent = AgentRandom(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
    elif agent_id == 0:
        agent = AgentRule(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)

    steward.simulate(agent=agent,epoch_number=simulate_epoch_number, train_mode=train_mode)


if __name__ == "__main__":
    run()