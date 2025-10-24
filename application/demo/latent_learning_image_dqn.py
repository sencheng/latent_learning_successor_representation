# basic imports
import os
import numpy as np
import pyqtgraph as qg
import pickle
# tensorflow
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from PIL import Image
# framework imports
from application.frontends.frontends_blender import FrontendBlenderInterface
from application.spatial_representations.topology_graphs.manual_topology_graph_no_rotation import ManualTopologyGraphNoRotation
# from cobel.agents.keras_rl.dqn import DQNAgentBaseline
from cobel.agents.dyna_dqn import DynaDQN
from cobel.observations.image_observations import ImageObservationBaseline
from cobel.interfaces.baseline import InterfaceBaseline
from cobel.networks.network_tensorflow import SequentialKerasNetwork
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor, EscapeLatencyMonitor

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True


def build_model(shape: tuple, output_units: int):

    model = Sequential()
    model.add(Dense(units=64, input_shape=(shape,), activation='tanh'))
    model.add(Dense(units=64, activation='tanh'))
    model.add(Dense(units=output_units, activation='linear', name='output')) # sucessor feature
    model.compile(optimizer='adam', loss='mse')

    print(model.summary())
    
    return model

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
    
    if values['current_node'].goal_node and values['rl_agent'].current_trial > 50:
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
    
    if values['current_node'].goal_node and values['rl_agent'].current_trial > 50:
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

def reward_callback(values):
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
    
    if values['current_node'].goal_node and values['rlAgent'].current_trial > 50:
        reward = 10.0
        end_trial = True
    elif values['current_node'].goal_node:
        end_trial = True
    
    return reward, end_trial

# defining a dictionary to map learning paradigms to reward callbacks
reward_callbacks = {
    'targeted': reward_targeted_callback,
    'mistargeted': reward_targeted_callback,  # Assuming mistargeted uses the same callback
    'continuous': reward_continuous_callback,
    'direct': reward_direct_callback
}

def extract_features(observations: np.ndarray) -> np.ndarray:
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
    for obs in observations:
        img = Image.fromarray(obs, 'RGB')
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
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Demo: DQN')
    
    # determine demo scene path
    demo_scene = 'application/environments/gridworld.blend' # 'tolman_maze' or 'gridworld'
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendBlenderInterface(demo_scene)
    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, visual_output)
    modules['spatial_representation'] = ManualTopologyGraphNoRotation(modules, {'start_nodes': [0], 'goal_nodes': [99], 'clique_size': 4})
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)
    modules['spatial_representation'].set_state_action_state_transition() # matheus -> setting the state-action-state transitions
    modules['rl_interface'] = InterfaceBaseline(modules, visual_output, reward_callback=reward_callback)
    
    
    # setting the reward callback based on the type of learning
    if type_of_learning in reward_callbacks:
        modules['rl_interface'].reward_callback = reward_callbacks[type_of_learning]
    else:
        raise ValueError(f"Unknown type_of_learning: {type_of_learning}")
    
    #number of trials and max steps for trial
    number_of_trials = 100 if type_of_learning != 'direct' else 50 
    max_steps = 300

    # set observations from spatial_representation
    observations = np.array(modules['spatial_representation'].state_space)
    flatten_observation = extract_features(np.array(observations))
   
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, [0, 10])
    escape_latency_monitor = EscapeLatencyMonitor(number_of_trials, max_steps, main_window, visual_output)
    main_window.setGeometry(50, 50, 1600, 450)

    # define callbacks
    custom_callbacks = {'on_trial_end': [reward_monitor.update, escape_latency_monitor.update]}
    model = SequentialKerasNetwork(build_model(flatten_observation.shape[1], 4)) 

    # initialize RL agent
    rl_agent = DynaDQN(interface_OAI=modules['rl_interface'], epsilon=0.3, beta=5, gamma=0.95,
                       model=model, observations=flatten_observation, custom_callbacks=custom_callbacks)
    
    rl_agent.mask_actions = True
    rl_agent.design = type_of_learning

    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent
    
    # define episodes where the agent is evaluated on each state to get the DRs
    if type_of_learning == 'direct':
        recording_episodes = np.arange(0, 50+25, 25)
        
    else:
        recording_episodes = np.arange(0, 100+25, 25)

  
    Q_values = {}

    for e in recording_episodes:
        Q_values[e] = np.zeros((flatten_observation.shape[1], 4))
        
    # before training, record SR, Q-values and reward function; state_idx is the index of the available state in the entire maze
    for i, state_idx in enumerate(range(flatten_observation.shape[0])):
        Q_values[0][i] = rl_agent.retrieve_Q(state_idx)

    # let the agent learn, with extremely large number of allowed maximum steps
    for m in range(1, len(recording_episodes)):
        
        # let the agent learn, with extremely large number of allowed maximum steps
        if (type_of_learning == 'continuous'):
            if (type_of_learning == 'continuous' and recording_episodes[m] <=50):
                if env == 'tolman':
                    rl_agent.interface_OAI.modules['spatial_representation'].set_starting_nodes([0, 6, 12, 18, 24, 29])
                if env == 'gridworld':
                    rl_agent.interface_OAI.modules['spatial_representation'].set_starting_nodes([0, 9, 50, 59, 90, 99])
            else:
                rl_agent.interface_OAI.modules['spatial_representation'].set_starting_nodes([0])

        if (type_of_learning == 'mistargeted' and env == 'gridworld'):
            if recording_episodes[m] <=50:
                rl_agent.interface_OAI.modules['spatial_representation'].set_goal_nodes([9])
            else:
                rl_agent.interface_OAI.modules['spatial_representation'].set_goal_nodes([99])
        
        rl_agent.train(recording_episodes[m] - recording_episodes[m - 1], max_steps, replay_batch_size=32)
        # before next training stage, record SR
        for i, state_idx in enumerate(range(flatten_observation.shape[0])):
            Q_values[recording_episodes[m]][i] = rl_agent.retrieve_Q(state_idx)

    
    trajectories = modules['rl_interface'].trajectories.copy()

    
    # clear keras session (for performance)
    K.clear_session()
    
    # stop simulation
    modules['world'].stop_blender()
    
    # and also stop visualization
    if visual_output:
        main_window.close()
        
    # clear Keras session (for performance)
    K.clear_session()

    return Q_values, trajectories

env = 'gridworld' # 'tolman' or 'gridworld'

if __name__ == '__main__':
    # params
    number_of_runs = 30
    paradigms = ['targeted'] # 'direct' (learning), 'targeted' (latent learning), 'mistargeted' (for gridworld), 'continuous'
    # ensure that the directory for storing the results exists
    path = env + '-' +'DQN-image-encoded-vgg16/'
    for paradigmam in paradigms:
        os.makedirs(path + paradigmam, exist_ok=True)
    
    # run simulations
    for run in range(1, number_of_runs):
        for paradigm in paradigms:
            print(f'Running simulation {run + 1} of paradigm {paradigm}')
            data_path = os.path.join(path, paradigm, f'run_{run + 1}.pickle')
            
            if env == 'tolman':
                Qvalues, Escape = learning_paradigm(paradigm)
                data = {'Qvalues': Qvalues, 'Escape': Escape}
            else:
                Qvalues, Escape = learning_paradigm(paradigm)
                data = {'Qvalues': Qvalues, 'Escape': Escape}
            
            with open(data_path, 'wb') as handle:
                pickle.dump(data, handle)
