# basic imports
import numpy as np
import pickle
import os
import pyqtgraph as qg
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
# import tensorflow as tf
# CoBel-RL framework
from application.agents.gridworld_dsr import SimpleDSR
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

def single_run(maze='gridworld', policy='softmax', design='targeted', beta=10, latent=False):
    '''
    policy:  softmax or greedy
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
    num_states = 10*10
    
    # define initial reward environment
    step_r = 0
    env_rewards = [[i, step_r] for i in range(num_states)]
    # initialize world
    world = make_gridworld(10, 10, terminals=[99], rewards=np.array(env_rewards), goals=[99])

    if design == 'mistargeted':
        world = make_gridworld(10, 10, terminals=[9], rewards=np.array(env_rewards), goals=[9])
    else:
        world = make_gridworld(10, 10, terminals=[99], rewards=np.array(env_rewards), goals=[99])

    if design == 'continuous': 
        world['starting_states'] = np.array([0, 9, 50, 59, 90, 99])
    else:
        world['starting_states'] = np.array([0])

    ######################################################################
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = world
    # make observations, we use one-hot encodings
    observations = np.eye(num_states)
    modules['rl_interface'] = InterfaceDiscrete(modules, world['sas'], observations, world['rewards'],
                                                world['terminals'], world['starting_states'], world['coordinates'], world['goals'],
                                                visual_output, main_window)

    # set the experimental parameters for latent learning
    modules['rl_interface'].total_episode = 100
    modules['rl_interface'].rewarding_episode = 50
    modules['rl_interface'].latent_learning = latent
    maxSteps = 300

    # initialize RL agent
    rlAgent = SimpleDSR(interfaceOAI=modules['rl_interface'], epsilon=0.3, beta=beta, learningRate=0.9,
                      gamma=0.95, trialEndFcn=trialEndCallback, observations=observations)
    rlAgent.mask_actions = True
    rlAgent.use_follow_up_state = True
    rlAgent.policy = policy
    rlAgent.design = design

    if latent == False:
        rlAgent.trialNumber=modules['rl_interface'].rewarding_episode
    else:
        rlAgent.trialNumber=modules['rl_interface'].total_episode

    if design == 'targeted' or design == 'mistargeted':
        modules['rl_interface'].if_terminal = True
    elif design == 'continuous':
        modules['rl_interface'].if_terminal = False

    # initialize performance Monitor
    perfMon = RLPerformanceMonitorBaseline(rlAgent, rlAgent.trialNumber + 1, main_window, visual_output, [maxSteps*step_r, 8])
    rlAgent.performanceMonitor = perfMon
    modules['rl_interface'].rlAgent = rlAgent

    # define episodes where the agent is evaluated on each state to get the DRs
    if latent:
        recording_episodes = np.arange(0, 100+10, 10)
    else:
        recording_episodes = np.arange(0, 50+10, 10)

    SRs = {}
    Q_values = {}
    reward_f = {}
    states_trans_probs = {}
    for e in recording_episodes:
        SRs[e] = np.zeros((num_states, 4, num_states))
        Q_values[e] = np.zeros((num_states, 4))
        reward_f[e] = np.zeros(num_states)
        states_trans_probs[e] = np.zeros((2,num_states))

    # before training, record SR, Q-values and reward function; state_idx is the index of the available state in the entire maze
    for i, state_idx in enumerate(range(num_states)):
        SRs[0][i] = rlAgent.retrieve_SR(state_idx)
        Q_values[0][i] = rlAgent.retrieve_Q(observations[state_idx])
    reward_f[0] = rlAgent.retrieve_reward_function(range(num_states))

    # let the agent learn, with extremely large number of allowed maximum steps
    acc_episode = 0
    for m in range(1, len(recording_episodes)):
        if acc_episode >= modules['rl_interface'].rewarding_episode:
            modules['rl_interface'].starting_states = np.array([0])
        rlAgent.train(recording_episodes[m] - recording_episodes[m - 1], maxSteps, replayBatchSize=32)
        acc_episode += recording_episodes[m] - recording_episodes[m - 1]
        # before next training stage, record SR
        for i, state_idx in enumerate(range(num_states)):
            SRs[recording_episodes[m]][i] = rlAgent.retrieve_SR(state_idx)
            Q_values[recording_episodes[m]][i] = rlAgent.retrieve_Q(observations[state_idx])
        reward_f[recording_episodes[m]] = rlAgent.retrieve_reward_function(range(num_states))

    trajectories = modules['rl_interface'].trajectories.copy()

    K.clear_session()

    # and also stop visualization
    if visual_output:
        main_window.close()

    return SRs, Q_values, reward_f, trajectories, states_trans_probs

if __name__ == '__main__':
    env = 'gridworld'
    latent = True # defines latent or direct learning
    design = 'targeted' # targeted, mistargeted, continuous
    policy = 'greedy'
    beta = 1 # for softmax policy
    epochs = 10
    complement = 'init'

    for epoch in range(epochs):
        
        data_folder = os.path.dirname(os.path.abspath(__file__)) + '/../data/%s-%s-%s-%s/' % (env, policy, design, complement)
        os.makedirs(data_folder, exist_ok=True)

        data_path = data_folder + 'Trainingdata_%s.pickle' % (epoch + 1)
        if not os.path.exists(data_path):

            SRs, Qvalues, reward_f, escape_latency, states_trans_probs = single_run(policy=policy, latent=latent, beta=beta, design=design)
            data = {'SRs': SRs, 'Qvalues': Qvalues,  'Reward': reward_f, 'Escape': escape_latency, 'SAProbs': states_trans_probs}
        
            with open(data_path, 'wb') as handle:
                pickle.dump(data, handle)


