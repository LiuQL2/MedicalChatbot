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


def run(parameter, main_checkpoint_path):
    agent_id = parameter.get("agent_id")
    dqn_id = parameter.get("dqn_id")
    disease_number = parameter.get("disease_number")

    lr = parameter.get("dqn_learning_rate")
    reward_for_success = parameter.get("reward_for_success")
    reward_for_fail = parameter.get("reward_for_fail")
    reward_for_not_come_yet = parameter.get("reward_for_not_come_yet")
    reward_for_inform_right_symptom = parameter.get("reward_for_inform_right_symptom")

    max_turn = parameter.get("max_turn")
    minus_left_slots = parameter.get("minus_left_slots")
    gamma = parameter.get("gamma")
    epsilon = parameter.get("epsilon")
    run_id = parameter.get('run_id')


    if agent_id == 1:
        checkpoint_path = main_checkpoint_path + "checkpoint_d" + str(disease_number) + "_agt" + str(agent_id) + \
                          "_dqn" + str(dqn_id) + "_T" + str(max_turn) + "_lr" + str(lr) + "_RFS" + str(reward_for_success) +\
                          "_RFF" + str(reward_for_fail) + "_RFNCY" + str(reward_for_not_come_yet) + "_RFIRS" + str(reward_for_inform_right_symptom) +\
                          "_mls" + str(minus_left_slots) + "_gamma" + str(gamma) + "_epsilon" + str(epsilon) + "_RID" + str(run_id) + "/"
    else:
        checkpoint_path = main_checkpoint_path + "checkpoint_d" + str(disease_number) + "_agt" + str(agent_id) + \
                          "_T" + str(max_turn) + "_lr" + str(lr) + "_RFS" + str(reward_for_success) + \
                          "_RFF" + str(reward_for_fail) + "_RFNCY" + str(reward_for_not_come_yet) + "_RFIRS" + str(reward_for_inform_right_symptom) + \
                          "_mls" + str(minus_left_slots) + "_gamma" + str(gamma) + "_epsilon" + str(epsilon) + "_RID" + str(run_id) + "/"

    print(json.dumps(parameter, indent=2))
    time.sleep(8)

    slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
    action_set = pickle.load(file=open(parameter["action_set"], "rb"))
    disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))
    steward = RunningSteward(parameter=parameter, checkpoint_path=checkpoint_path)

    warm_start = parameter.get("warm_start")
    warm_start_epoch_number = parameter.get("warm_start_epoch_number")
    train_mode = parameter.get("train_mode")
    agent_id = parameter.get("agent_id")
    simulate_epoch_number = parameter.get("simulate_epoch_number")

    # Warm start.
    if warm_start == 1 and train_mode == 1:
        print("warm starting...")
        agent = AgentRule(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom,
                          parameter=parameter)
        steward.warm_start(agent=agent, epoch_number=warm_start_epoch_number)
    # exit()
    if agent_id == 1:
        agent = AgentDQN(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom, parameter=parameter)
    elif agent_id == 2:
        agent = AgentActorCritic(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom,
                                 parameter=parameter)
    elif agent_id == 3:
        agent = AgentRandom(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom,
                            parameter=parameter)
    elif agent_id == 0:
        agent = AgentRule(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom,
                          parameter=parameter)

    steward.simulate(agent=agent, epoch_number=simulate_epoch_number, train_mode=train_mode)


learning_rate_list = [0.001, 0.0001]
max_turn_list = [22,44]
data_set_list = ["model_tag"]
minus_left_slots_list = [0,-1]
reward_for_inform_right_symptom_list = [-1,-0.5,0]

for data_set in data_set_list:
    data_set_path = "./../data/dataset/" + data_set + "/"
    performance_path = "./../data/dataset/" + data_set + "/result/learning_rate/"
    main_checkpoint_path = "./../data/dataset/" + data_set + "/result/checkpoint/"
    log_dir = "./../data/dataset/" + data_set + "/result/log/"
    for max_turn in max_turn_list:
        for minus_left_slots in minus_left_slots_list:
            for lr in learning_rate_list:
                for reward_for_inform_right_symptom in reward_for_inform_right_symptom_list:
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--disease_number", dest="disease_number", type=int, default=5,
                                        help="the number of disease.")
                    parser.add_argument("--device_for_tf", dest="device_for_tf", type=str, default="/device:GPU:2",
                                        help="the device for tensorflow running on.")

                    # TODO: simulation configuration
                    parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default=1500,
                                        help="the number of simulate epoch.")
                    parser.add_argument("--epoch_size", dest="epoch_size", type=int, default=100,
                                        help="the size of each simulate epoch.")
                    parser.add_argument("--evaluate_epoch_number", dest="evaluate_epoch_number", type=int, default=1000,
                                        help="the size of each simulate epoch when evaluation.")
                    parser.add_argument("--experience_replay_pool_size", dest="experience_replay_pool_size", type=int,
                                        default=10000, help="the size of experience replay.")
                    parser.add_argument("--hidden_size_dqn", dest="hidden_size_dqn", type=int, default=500,
                                        help="the hidden_size of DQN.")
                    parser.add_argument("--warm_start", dest="warm_start", type=int, default=1,
                                        help="use rule policy to fill the experience replay buffer at the beginning, 1:True; 0:False")
                    parser.add_argument("--warm_start_epoch_number", dest="warm_start_epoch_number", type=int,
                                        default=30,
                                        help="the number of epoch of warm starting.")
                    parser.add_argument("--batch_size", dest="batch_size", type=int, default=30,
                                        help="the batch size when training.")
                    parser.add_argument("--log_dir", dest="log_dir", type=str, default=log_dir,
                                        help="directory where event file of training will be written, ending with /")
                    parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.1, help="the greedy of DQN")
                    parser.add_argument("--gamma", dest="gamma", type=float, default=1.0,
                                        help="The discount factor of immediate reward.")
                    parser.add_argument("--train_mode", dest="train_mode", type=int, default=1,
                                        help="training mode? True:1 or False:0")

                    # TODO: Save model, performance and dialogue content ? And what is the path if yes?
                    parser.add_argument("--save_performance", dest="save_performance", type=int, default=1,
                                        help="save the performance? 1:Yes, 0:No")
                    parser.add_argument("--performance_save_path", dest="performance_save_path", type=str,
                                        default=performance_path,
                                        help="the folder where learning rate save to, ending with /.")
                    parser.add_argument("--save_model", dest="save_model", type=int, default=1,
                                        help="save model? 1:Yes,0:No")
                    parser.add_argument("--checkpoint_path", dest="checkpoint_path", type=str,
                                        default=main_checkpoint_path,
                                        help="the folder where models save to, ending with /.")
                    parser.add_argument("--saved_model", dest="saved_model", type=str,
                                        default="./../model/dqn/checkpoint/checkpoint_d4_agt1_dqn1/model_d4_agent1_dqn1_s0.619_r18.221_t4.266_wd0.0_e432.ckpt")
                    parser.add_argument("--dialogue_file", dest="dialogue_file", type=str,
                                        default="./../data/dialogue_output/dialogue_file.txt",
                                        help="the file that used to save dialogue content.")
                    parser.add_argument("--save_dialogue", dest="save_dialogue", type=int, default=0,
                                        help="save the dialogue? 1:Yes, 0:No")

                    # TODO: user configuration.
                    parser.add_argument("--allow_wrong_disease", dest="allow_wrong_disease", type=int, default=0,
                                        help="Allow the agent to inform wrong disease? 1:Yes, 0:No")

                    # TODO: Learning rate for actor-critic and dqn.
                    parser.add_argument("--dqn_learning_rate", dest="dqn_learning_rate", type=float, default=lr,
                                        help="the learning rate of dqn.")
                    parser.add_argument("--actor_learning_rate", dest="actor_learning_rate", type=float, default=0.001,
                                        help="the learning rate of actor")
                    parser.add_argument("--critic_learning_rate", dest="critic_learning_rate", type=float,
                                        default=0.001,
                                        help="the learning rate of critic")
                    parser.add_argument("--trajectory_pool_size", dest="trajectory_pool_size", type=int, default=100,
                                        help="the size of trajectory pool")

                    # TODO: the number condition of explicit symptoms and implicit symptoms in each user goal.
                    parser.add_argument("--explicit_number", dest="explicit_number", type=int, default=0,
                                        help="the number of explicit symptoms of used sample")
                    parser.add_argument("--implicit_number", dest="implicit_number", type=int, default=0,
                                        help="the number of implicit symptoms of used sample")

                    # TODO: agent to use.
                    parser.add_argument("--agent_id", dest="agent_id", type=int, default=1,
                                        help="the agent to be used:{0:AgentRule, 1:AgentDQN, 2:AgentActorCritic, 3:AgentRandom}")
                    parser.add_argument("--dqn_id", dest="dqn_id", type=int, default=1,
                                        help="the dqn to be used in agent:{0:initial dqn of qianlong, 1:dqn with one layer of qianlong, 2:dqn with two layers of qianlong, 3:dqn of Baolin.}")

                    # TODO: 4 diseases. goal set, slot set, action set
                    parser.add_argument("--action_set", dest="action_set", type=str,
                                        default=data_set_path + 'action_set.p',
                                        help='path and filename of the action set')
                    parser.add_argument("--slot_set", dest="slot_set", type=str,
                                        default=data_set_path + 'slot_set.p',
                                        help='path and filename of the slots set')
                    parser.add_argument("--goal_set", dest="goal_set", type=str,
                                        default=data_set_path + 'goal_set.p',
                                        help='path and filename of user goal')
                    parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,
                                        default=data_set_path + "disease_symptom.p",
                                        help="path and filename of the disease_symptom file")
                    parser.add_argument("--max_turn", dest="max_turn", type=int, default=max_turn,
                                        help="the max turn in one episode.")
                    # parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=max_turn + 137,
                    #                     help="the input_size of DQN.")
                    parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=max_turn + 382,
                                        help="the input_size of DQN.")

                    # TODO: reward for different dialogue status.
                    parser.add_argument("--reward_for_not_come_yet", dest="reward_for_not_come_yet", type=float,
                                        default=-1)
                    parser.add_argument("--reward_for_success", dest="reward_for_success", type=float,
                                        default=2 * max_turn)
                    parser.add_argument("--reward_for_fail", dest="reward_for_fail", type=float, default=-1*max_turn)
                    parser.add_argument("--reward_for_inform_right_symptom", dest="reward_for_inform_right_symptom",
                                        type=float, default=reward_for_inform_right_symptom)
                    parser.add_argument("--minus_left_slots", dest="minus_left_slots", type=int,
                                        default=minus_left_slots,
                                        help="Reward for success minus left slots? 1:Yes, 0:No")
                    parser.add_argument("--run_id", dest='run_id',type=int, default=10,help='the id of this running.')

                    args = parser.parse_args()
                    parameter = vars(args)

                    run(parameter=parameter, main_checkpoint_path=main_checkpoint_path)