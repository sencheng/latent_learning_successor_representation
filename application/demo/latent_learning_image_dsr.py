# basic imports
import os
import sys
import numpy as np
import pickle
import pyqtgraph as qg
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import smart_resize
from PIL import Image
# framework imports
from application.frontends.frontends_blender import FrontendBlenderInterface
from application.spatial_representations.topology_graphs.manual_topology_graph_no_rotation import ManualTopologyGraphNoRotation
from application.agents.image_dsr import DynaDSR
from cobel.observations.image_observations import ImageObservationBaseline
from cobel.interfaces.baseline import InterfaceBaseline
from cobel.networks.network_tensorflow import SequentialKerasNetwork
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor, EscapeLatencyMonitor, RepresentationMonitor

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True


def build_model(shape: tuple, output_units: int) -> Sequential:
    '''
    This function builds a simple network model. 
    
    Parameters
    ----------
    input_shape :                       The network model's input shape.
    output_units :                      The network model's number of output units.
    
    Returns
    ----------
    model :                             The built network model.
    '''
    model = Sequential()
    model.add(Dense(units=64, input_shape=(shape,), activation='tanh'))
    model.add(Dense(units=64, activation='tanh'))
    model.add(Dense(units=output_units, activation='linear', name='output')) # sucessor feature
    model.compile(optimizer='adam', loss='mse')
    
    return model

def build_model_reward(shape: tuple, output_units: int) -> Sequential:
    '''
    This function builds a simple network model for the reward. 
    
    Parameters
    ----------
    input_shape :                       The network model's input shape.
    output_units :                      The network model's number of output units.
    
    Returns
    ----------
    model :                             The built network model.
    '''
    # build reward model which needs to be a linear model in theory
    model = Sequential()
    # linear reward model
    model.add(Dense(units=output_units, input_shape=(shape,), activation='linear', use_bias=False))    
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model

def on_trial_begin_mistargeted_callback(values):
    '''
    this is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.
    
    Parameters
    ----------
    values :                            A dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.
    
    Returns
    ----------
    reward :                            The reward that will be provided.
    end_trial :                         Flag indicating whether the trial ended.
    '''
    # the started 
    modules = values['rl_parent'].interface_OAI.modules
    
    # setting the starting states during pre-exposure 
    mistargeted_goal_node = [9]
    
    # setting the starting states during learning 
    learning_goal_node = mistargeted_goal_node[0:1]

    modules['spatial_representation'].nodes[9].goal_node = False
    modules['spatial_representation'].update()
    
    if values['rl_parent'].current_trial < 50:
        for index in mistargeted_goal_node:
            modules['spatial_representation'].nodes[index].goal_node = True
            modules['spatial_representation'].update()
        for index in learning_goal_node:
            modules['spatial_representation'].nodes[index].goal_node = False

    else:    
        for index in mistargeted_goal_node:
            modules['spatial_representation'].nodes[index].goal_node = True
        for index in learning_goal_node:
            modules['spatial_representation'].nodes[index].goal_node = False

def reward_targeted_callback(values):
    '''
    This is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.
    
    Parameters
    ----------
    values :                            A dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.
    
    Returns
    ----------
    reward :                            The reward that will be provided.
    end_trial :                         Flag indicating whether the trial ended.
    '''
    # the standard reward for each step taken is negative, making the agent seek short routes   
    reward = 0.0
    end_trial = False
    
    if values['current_node'].goal_node and values['rl_agent'].current_trial >= 50:
        reward = 5.0
        end_trial = True
        
    elif values['current_node'].goal_node:
        end_trial = True
    
    return reward, end_trial

def reward_continuous_callback(values):
    '''
    This is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.
    
    Parameters
    ----------
    values :                            A dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.
    
    Returns
    ----------
    reward :                            The reward that will be provided.
    end_trial :                         Flag indicating whether the trial ended.
    '''
    # the standard reward for each step taken is negative, making the agent seek short routes   
    reward = 0.0
    end_trial = False
    
    if values['current_node'].goal_node and values['rl_agent'].current_trial >= 50:
        reward = 5.0
        end_trial = True
    
    return reward, end_trial

def reward_direct_callback(values):
    '''
    This is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.
    
    Parameters
    ----------
    values :                            A dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.
    
    Returns
    ----------
    reward :                            The reward that will be provided.
    end_trial :                         Flag indicating whether the trial ended.
    '''
    # the standard reward for each step taken is negative, making the agent seek short routes   
    reward = 0.0
    end_trial = False
    
    if values['current_node'].goal_node:
        reward = 5.0
        end_trial = True
    
    return reward, end_trial

# defining a dictionary to map learning paradigms to reward callbacks
reward_callbacks = {
    'targeted': reward_targeted_callback,
    'mistargeted': reward_targeted_callback,  # Assuming mistargeted uses the same callback
    'continuous': reward_continuous_callback,
    'direct': reward_direct_callback
}

def extract_features(observations: np.ndarray, min_range: int, max_range: int) -> np.ndarray:
    """
    Extract features from observations using ResNet50.

    Parameters
    ----------
    observations : np.ndarray
        A numpy array of shape (N, 64, 256, 3), where N is the number of observations.

    Returns
    ----------
    features : np.ndarray
        A numpy array of extracted features for each observation.
    """
    # Initialize ResNet50 model (without the top classification layer)
    resnet_model = VGG16(weights='imagenet', include_top=False)
    # Resize observations to (224, 224, 3) and preprocess them
    features_observations = []
    print(f'\n\nExtracting features from observations using VGG16...\n\n')
    for i in range(min_range, max_range):
        img = Image.fromarray(observations[i], 'RGB')
        resized_img = smart_resize(img, (32, 128))
        preprocessed_img = K.expand_dims(resized_img, axis=0)  # Add batch dimension
        preprocessed_img = preprocess_input(preprocessed_img) 
        # Extract features using ResNet50
        features = resnet_model.predict(preprocessed_img, steps=1)
        # Flatten the features
        flatten_features = features.flatten()
        normalized_features = flatten_features / np.linalg.norm(flatten_features)
        features_observations.append(normalized_features)
    
    features_observations = np.array(features_observations)
    return features_observations

def learning_paradigm(type_of_learning: str = 'latent'):
    '''
    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    type_of_learning :                  The learning paradigm that will be used. \'latent\' by default.
    
    Returns
    ----------
    latency_trace :                     The escape latency trace.
    activtiy_trace_left :               The activity trace for the left action (action == 0).
    activtiy_trace_up :                 The activity trace for the up action (action == 1).
    activtiy_trace_right :              The activity trace for the right action (action == 2).
    activtiy_trace_down :               The activity trace for the down action (action == 3).
    ''' 
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Demo: DSR')
    
    # determine demo scene path
    demo_scene = 'application/environments/gridworld.blend' # 'tolman_maze' or 'gridworld'
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendBlenderInterface(demo_scene)
    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, visual_output)
    goal_nodes = 29 if env == 'tolman' else 99
    modules['spatial_representation'] = ManualTopologyGraphNoRotation(modules, {'start_nodes': [0], 'goal_nodes': [goal_nodes], 'clique_size': 4})
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)
    modules['spatial_representation'].set_state_action_state_transition() # matheus -> setting the state-action-state transitions
    modules['rl_interface'] = InterfaceBaseline(modules, visual_output)  
    
    # setting the reward callback based on the type of learning
    if type_of_learning in reward_callbacks:
        modules['rl_interface'].reward_callback = reward_callbacks[type_of_learning]
    else:
        raise ValueError(f"Unknown type_of_learning: {type_of_learning}")
    
    #number of trials and max steps for trial
    number_of_trials = 100 if type_of_learning != 'direct' else 50 
    max_steps = 500

    # set observations from spatial_representation
    observations = np.array(modules['spatial_representation'].state_space)
    
    # Usign vgg16 to extract features from the observations. The pretrained network is belived to facilitate the learning process.
    trainable_flatten_observation = extract_features(np.array(observations), 0, observations.shape[0])
    trainable_flatten_observation = np.array(trainable_flatten_observation)
   
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [0, 10])
    escape_latency_monitor = EscapeLatencyMonitor(number_of_trials, max_steps, main_window, visual_output)
    
    custom_callbacks = {'on_trial_end': [reward_monitor.update, escape_latency_monitor.update]} #  + [representation_monitors[action].update for action in range(4)]}

    # build models
    model_SR = SequentialKerasNetwork(build_model(trainable_flatten_observation.shape[1], trainable_flatten_observation.shape[1])) 
    model_reward = SequentialKerasNetwork(build_model_reward(trainable_flatten_observation.shape[1], 1)) # matheus -> linear output
    
    # initialize RL agent
    rl_agent = DynaDSR(interface_OAI=modules['rl_interface'], epsilon=0.3, beta=1, gamma=0.95,
                       model_SR=model_SR, observations=trainable_flatten_observation, model_reward=model_reward, custom_callbacks=custom_callbacks)

    rl_agent.mask_actions = True
    rl_agent.use_follow_up_state = True
    rl_agent.design = type_of_learning
    rl_agent.ignore_terminality = False
    if env == 'tolman':
        rl_agent.compute_transition_probs = True

    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent

    # define episodes where the agent is evaluated on each state to get the DRs
    if type_of_learning == 'direct':
        recording_episodes = np.arange(0, 50+25, 25)
        
    else:
        recording_episodes = np.arange(0, 100+25, 25)

    SRs = {}
    Q_values = {}
    states_trans_probs = {}
    for e in recording_episodes:
        SRs[e] = np.zeros((trainable_flatten_observation.shape[1], 4, trainable_flatten_observation.shape[1]))
        Q_values[e] = np.zeros((trainable_flatten_observation.shape[1], 4))
        
    # before training, record SR, Q-values and reward function; state_idx is the index of the available state in the entire maze
    for i, state_idx in enumerate(range(trainable_flatten_observation.shape[0])):
        SRs[0][i] = rl_agent.retrieve_SR(state_idx)
        Q_values[0][i] = rl_agent.retrieve_Q(state_idx)
    
    # let the agent learn, with extremely large number of allowed maximum steps
    for m in range(1, len(recording_episodes)):
        
        # let the agent learn, with extremely large number of allowed maximum steps
        if (type_of_learning == 'continuous'):
            if recording_episodes[m] <=50:
                if env == 'tolman':
                    rl_agent.interface_OAI.modules['spatial_representation'].set_starting_nodes([0, 6, 12, 18, 24, 29])
                if env == 'gridworld':
                    rl_agent.interface_OAI.modules['spatial_representation'].set_starting_nodes([0, 9, 50, 59, 90, 99])
            else:
                rl_agent.interface_OAI.modules['spatial_representation'].set_starting_nodes([0])
        
        # mistargeted learning
        if (type_of_learning == 'mistargeted' and env == 'gridworld'):
            if recording_episodes[m] <=50:
                rl_agent.interface_OAI.modules['spatial_representation'].set_goal_nodes([9])
            else:
                rl_agent.interface_OAI.modules['spatial_representation'].set_goal_nodes([99])
                
        rl_agent.train(recording_episodes[m] - recording_episodes[m - 1], max_steps, replay_batch_size=32)
        # before next training stage, record SR
        for i, state_idx in enumerate(range(trainable_flatten_observation.shape[0])):
            SRs[recording_episodes[m]][i] = rl_agent.retrieve_SR(state_idx)
            Q_values[recording_episodes[m]][i] = rl_agent.retrieve_Q(state_idx)
            
    states_trans_probs = rl_agent.transition_probs_episodes.copy()
    trajectories = modules['rl_interface'].trajectories.copy()

    # clear Keras session (for performance)
    K.clear_session()
    
    # stop simulation
    modules['world'].stop_blender()

    # and also stop visualization
    if visual_output:
        main_window.close()

    # clear Keras session (for performance)
    K.clear_session()

    if env == 'tolman':
        return trainable_flatten_observation, SRs, Q_values, trajectories, states_trans_probs    
    else:
        return trainable_flatten_observation, SRs, Q_values, trajectories

# env = 'gridworld' # 'gridworld' or 'tolman'
env = 'gridworld' # 'gridworld' or 'tolman'

if __name__ == '__main__':

    # params
    number_of_runs = 10
    paradigms = ['mistargeted', 'targeted']
    # ensure that the directory for storing the results exists
    path = env + '-' +'image-encoded-vgg16/'
    for paradigmam in paradigms:
        os.makedirs(path + paradigmam, exist_ok=True)
    
    # run simulations
    for run in range(number_of_runs):
        for paradigm in paradigms:
            print(f'Running simulation {run + 1} of paradigm {paradigm}')
            data_path = os.path.join(path, paradigm, f'run_{run + 1}.pickle')
            
            if env == 'tolman':
                Observations, SRs, Qvalues, Escape, Probs = learning_paradigm(paradigm)
                data = {'Observations': Observations, 'SRs': SRs, 'Qvalues': Qvalues, 'Escape': Escape, 'Probs': Probs}
            else:
                Observations, SRs, Qvalues, Escape = learning_paradigm(paradigm)
                data = {'Observations': Observations, 'SRs': SRs, 'Qvalues': Qvalues, 'Escape': Escape}
            
            with open(data_path, 'wb') as handle:
                pickle.dump(data, handle)
        

