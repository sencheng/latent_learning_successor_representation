# basic imports
import numpy as np
import pickle
import os
import pyqtgraph as qg
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import tensorflow as tf
# CoBel-RL framework
from application.agents.tolman_dsr import SimpleDSR
from application.rl_monitoring.rl_performance_monitors import RLPerformanceMonitorBaseline
from application.interface.discrete import InterfaceDiscrete
from cobel.misc.gridworld_tools import make_gridworld

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'!
visual_output = True

def trialEndCallback(trial,rlAgent,logs):
    if visual_output:
        # update the visual elements if required
        rlAgent.performanceMonitor.update(trial,logs)

def set_inv_tran(invalid_transitions):
    new_inv_tran = []
    for inv_tran in invalid_transitions:
        invert = (inv_tran[1], inv_tran[0])
        new_inv_tran.append(inv_tran)
        new_inv_tran.append(invert)
    return new_inv_tran

def single_run(maze='tolman', policy='softmax', beta=10, latent=False, design='targeted', act_doors=False):
    '''
    maze:   tolman
    policy: softmax or greedy
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Demo: Dyna-Q Agent')
    #################################################################
    
    # maze info  
    reachable_states = [184, 170, 156, 155, 154, 157, 158, 159, 173, 187, 145, 131, 117, 118, 119, 116, 115, 114, 128, 142, 100,  86,  72,  71,
                        70,  73,  74,  75,  89, 103,  61,  47,  33,  32,  31,  34,  35,  36,  22,   8,  50,  64,  78,  77,  76,  79,  80,  81,
                        67,  53,  95, 109, 123, 124, 125, 122, 121, 120, 106,  92, 134, 148, 162, 161, 160, 163, 164, 165, 151, 137, 179, 193]        
    num_states = len(reachable_states)
    goal_state = reachable_states[-1]
    terminal = reachable_states[-1]
    invalid_states = list(set(np.arange(14*14)).difference(set(reachable_states)))

    invalid_transition_one_way = [(184, 183), (184, 185), (170, 169), (170, 171), (156, 142), (155, 141), (155, 169), (154, 140), (154, 168), (157, 143), (157, 171), (158, 144), (158, 172),
                    (159, 160), (173, 172), (173, 174), (187, 186), (187, 188), (145, 144), (145, 146), (131, 130), (131, 132), (117, 103), (118, 104), (118, 132), (119, 105), (119, 133), (119, 120),
                    (116, 102), (116, 130), (115, 101), (115, 129), (114, 113), (128, 127), (128, 129), (142, 141), (142, 143), (142, 156), (100, 99), (100, 101), (86, 85), (86, 87), (72, 58), (71, 57), (71, 85), (70, 56), (70, 84),
                    (73, 59), (73, 87), (74, 60), (74, 88), (75, 76), (89, 88), (89, 90), (103, 102), (103, 104), (103, 117), (61, 60), (61, 62), (47, 46), (47, 48), (33, 19), (32, 46), (32, 18), (31, 45), (31, 17), (31, 30),
                    (34, 20), (34, 48), (35, 49), (35, 21), (36, 37), (22, 21), (22, 23), (8, 7), (8, 9), (50, 49), (50, 51), (64, 63), (64, 65), (78, 92), (77, 63), (77, 91), (76, 62), (76, 90), (76, 75), (79, 65), (79, 93), 
                    (80, 66), (80, 94), (81, 82), (67, 66), (67, 68), (53, 52), (53, 54), (53, 39), (95, 94), (95, 96), (109, 108), (109, 110), (123, 137), (124, 110), (124, 138), (125, 111), (125, 139), (122, 108), (122, 136), 
                    (121, 107), (121, 135), (120, 119), (106, 105), (106, 107), (92, 91), (92, 93), (92, 78), (134, 133), (134, 135), (148, 147), (148, 149), (162, 176), (161, 147), (161, 175), (160, 146), (160, 174), (160, 159), 
                    (163, 149), (163, 177), (164, 150), (164, 178), (165, 166), (151, 150), (151, 152), (137, 136), (137, 138), (137, 123), (179, 178), (179, 180), (193, 192), (193, 194)  ]
    
    invalid_transitions = set_inv_tran(invalid_transition_one_way)

    if act_doors:
        doors = [(156, 170 ), (159, 158), (117, 131), (114, 115), (72, 86), (75, 74), (33, 47), ( 36, 35), ( 78, 64), ( 81, 80), ( 123, 109), ( 120, 121), ( 162, 148), ( 165, 164)]
        invalid_transitions += doors

    # define initial reward environment
    step_r = 0
    env_rewards = [[i, step_r] for i in range(14*14)]
  
    # initialize world
    world = make_gridworld(14, 14, terminals=[terminal], rewards=np.array(env_rewards), goals=[goal_state], invalid_states=invalid_states,
                           invalid_transitions=invalid_transitions)

    if design == 'continuous':
        world['starting_states'] = np.array([184, 119, 103, 76, 92, 193])
    else:
        world['starting_states'] = np.array([184])

    ######################################################################
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = world
    # make observations, we use one-hot encodings
    observations = np.zeros((14*14, num_states))
    observations[reachable_states] = np.eye(num_states)

    modules['rl_interface'] = InterfaceDiscrete(modules, world['sas'], observations, world['rewards'],
                                                world['terminals'], world['starting_states'], world['coordinates'], world['goals'],
                                                visual_output, main_window)

    # set the experimental parameters for latent learning
    modules['rl_interface'].total_episode = 100
    modules['rl_interface'].rewarding_episode = 50
    modules['rl_interface'].latent_learning = latent
    maxSteps = 1500

    # initialize RL agent
    rlAgent = SimpleDSR(interfaceOAI=modules['rl_interface'], epsilon=0.3, beta=beta, learningRate=0.9,
                      gamma=0.95, trialEndFcn=trialEndCallback, observations=observations)
    rlAgent.mask_actions = True
    rlAgent.use_follow_up_state = True
    rlAgent.not_learning_doors = True
    rlAgent.policy = policy
    rlAgent.design = design

    if latent == False:
        rlAgent.trialNumber=modules['rl_interface'].rewarding_episode
    else:
        rlAgent.trialNumber=modules['rl_interface'].total_episode

    if design == 'targeted':
        modules['rl_interface'].if_terminal = True
    elif design == 'continuous':
        modules['rl_interface'].if_terminal = False
    
    # set false if doors are inactivated and you want to enable doors
    modules['rl_interface'].not_learning_doors = True

    # initialize performance Monitor
    perfMon = RLPerformanceMonitorBaseline(rlAgent, rlAgent.trialNumber + 1, main_window, visual_output, [maxSteps*step_r, 8])
    rlAgent.performanceMonitor = perfMon
    modules['rl_interface'].rlAgent = rlAgent

    # define episodes where the agent is evaluated on each state to get the DRs
    if latent:
        recording_episodes = np.arange(0, 100+10, 10)
    else:
        recording_episodes = np.arange(0, 50+10, 10)

    # placeholders for the DRs at each state
    SRs = {}
    Q_values = {}
    reward_f = {}
    states_trans_probs = {}
    for e in recording_episodes:
        SRs[e] = np.zeros((num_states, 4, num_states))
        Q_values[e] = np.zeros((num_states, 4))
        reward_f[e] = np.zeros(num_states)

    # before training, record SR, Q-values and reward function; state_idx is the index of the available state in the entire maze
    for i, state_idx in enumerate(reachable_states):
        SRs[0][i] = rlAgent.retrieve_SR(state_idx)
        Q_values[0][i] = rlAgent.retrieve_Q(observations[state_idx])
    reward_f[0] = rlAgent.retrieve_reward_function(reachable_states )

    # let the agent learn, with extremely large number of allowed maximum steps
    acc_episode = 0
    for m in range(1, len(recording_episodes)):
        if acc_episode >= modules['rl_interface'].rewarding_episode:
            modules['rl_interface'].starting_states = np.array([184])
        rlAgent.train(recording_episodes[m] - recording_episodes[m - 1], maxSteps, replayBatchSize=128)
        acc_episode += recording_episodes[m] - recording_episodes[m - 1]
        # before next training stage, record SR
        for i, state_idx in enumerate(reachable_states):
            SRs[recording_episodes[m]][i] = rlAgent.retrieve_SR(state_idx)
            Q_values[recording_episodes[m]][i] = rlAgent.retrieve_Q(observations[state_idx])
        reward_f[recording_episodes[m]] = rlAgent.retrieve_reward_function(reachable_states)
    
    # Copy the state transition probabilities from the DSR agent
    states_trans_probs = rlAgent.transition_probs_episodes.copy()
    
    trajectories = modules['rl_interface'].trajectories.copy()

    K.clear_session()

    # and also stop visualization
    if visual_output:
        main_window.close()

    return SRs, Q_values, reward_f, trajectories, states_trans_probs

if __name__ == '__main__':
    env = 'tolman'
    design = 'targeted' # targeted, continuous
    # defines latent or direct learning
    latent = True
    act_doors = False # if doors are inactivated and you want to enable doors
    policy = 'greedy'
    beta = 1 # for softmax policy
    epochs = 10
    complement = 'init'

    for epoch in range(epochs):
        
        data_folder = os.path.dirname(os.path.abspath(__file__)) + '/../data/%s-%s-%s-%s/' % (env, policy, design, complement)
        os.makedirs(data_folder, exist_ok=True)

        data_path = data_folder + 'Trainingdata_%s.pickle' % (epoch + 1)
        if not os.path.exists(data_path):

            SRs, Qvalues, reward_f, escape_latency, states_trans_probs = single_run(policy=policy, latent=latent, beta=beta, design=design, act_doors=act_doors)
            data = {'SRs': SRs, 'Qvalues': Qvalues,  'Reward': reward_f, 'Escape': escape_latency, 'SAProbs': states_trans_probs}
        
            with open(data_path, 'wb') as handle:
                pickle.dump(data, handle)




