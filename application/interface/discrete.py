# basic imports
import numpy as np
import gym
from gym import spaces
import pyqtgraph as pg
import PyQt5 as qt
# framework imports
from cobel.interfaces.rl_interface import AbstractInterface


class InterfaceDiscrete(AbstractInterface):

    def __init__(self, modules: dict, transitions: np.ndarray, observations: np.ndarray,
                 rewards: np.ndarray, terminals: np.ndarray, starting_states: list,
                 coordinates=None, goals=[], with_GUI=True, gui_parent=None):
        '''
        Open AI interface for use with gridworld environments.

        Parameters
        ----------
        modules :                           Contains framework modules.
        transitions :                       The state-action-state transition matrix.
        observations :                      The observations of all states.
        rewards :                           The reward function.
        terminals :                         Defines whether a state is terminal or non-terminal.
        starting_states :                   A list of starting states.
        coordinates :                       An array containing the state coordinates (Used for visualization). If undefined, states will be placed on a circle.
        goals :                             A list of goal states (Used for visualization).
        with_GUI :                          If true, observations and policy will be visualized.
        gui_parent :                        The main window used for visualization.

        Returns
        ----------
        None
        '''
        super().__init__(modules, with_GUI)
        # store environment relevant variables
        self.T = transitions  # state-action-state transiton matrix
        self.O = observations  # observations for all states
        self.R = rewards  # reward function
        self.E = terminals  # encodes whether a state is terminal or non-terminal
        self.starting_states = starting_states  # list of starting states
        self.deterministic = True  # always transition to state with highest probability
        self.coordinates = coordinates  # array containing the coordinates of each state
        self.goals = goals  # list of goal nodes
        # self.explored_actions = np.full((self.T.shape[0], self.T.shape[1]), True) # record which actions have been explored
        self.noise = 0.  # noise to be added to the observations
        self.range = None  # value range
        self.gui_parent = gui_parent
        self.world = modules['world']
        # prepare observation and action spaces
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=observations.shape[1:])
        self.action_space = gym.spaces.Discrete(transitions.shape[1])
        # initialize visualization
        self.initialize_visualization()
        # execute initial environment reset
        self.total_episode = 100
        self.rewarding_episode = 0
        self.latent_learning = False
        self.currentStep = 0
        self.currentEpisode = 0
        # record trajecotries
        self.trajectories = []
        # controls whether to actually set the terminal flag when the terminal is reached
        self.if_terminal = False
        self.not_learning_doors = True


    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        '''
        AI Gym's step function.
        Executes the agent's action and propels the simulation.

        Parameters
        ----------
        action :                            The action selected by the agent.

        Returns
        ----------
        observation :                       The observation of the new current state.
        reward :                            The reward received.
        end_trial :                         Flag indicating whether the trial ended.
        logs :                              The (empty) logs dictionary.
        '''
        # execute action, 0: left, 1: up, 2: right, 3: down

        transitionProbabilities = self.T[self.currentState][action]

        # print('transitionProbabilities')
        # print(transitionProbabilities)

        # record the already explored action
        # self.explored_actions[self.currentState][action] = True

        state = self.currentState

        self.trajectories[-1].append(self.currentState)

        if self.deterministic:
            self.currentState = np.argmax(transitionProbabilities)
        else:
            self.currentState = np.random.choice(np.arange(transitionProbabilities.shape[0]), p=transitionProbabilities)
        self.observation = np.copy(self.O[self.currentState])

        if self.noise > 0.:
            self.observation = np.random.normal(self.observation, self.noise)
            if self.range is not None:
                self.observation = np.clip(self.observation, a_min=self.range[0], a_max=self.range[1])
        
        next_state = self.currentState

    
        # determine reward and whether the episode should end
        reward = self.R[self.currentState]
        end_trial = self.E[self.currentState]
        
        if self.currentEpisode > self.rewarding_episode and not self.not_learning_doors:
            # add doors restrictions
            self.update_world()
            self.not_learning_doors = True

        if end_trial:
            if not self.latent_learning:
                reward = 5.0
            else:
                if self.currentEpisode > self.rewarding_episode:
                    reward = 5.0
                    
                # distinguish time fixed and single trial experiments
                else:
                    end_trial = self.if_terminal
        
        # update visualization
        self.update_visualization()
        self.currentStep += 1

        return self.observation, reward, end_trial, {}


    def update_world(self):
        door_transition = [(156, 3, 170 ), (159, 0, 158), (117, 3, 131), (114, 2, 115), 
                           (72, 3, 86), (75, 0, 74), (33, 3, 47), ( 36, 0, 35), ( 78, 1, 64), 
                           ( 81, 0, 80), ( 123, 1, 109), ( 120, 2, 121), ( 162, 1, 148), ( 165, 0, 164)]
        
        for (s,a,s_t) in door_transition:
            self.world['sas'][s][a][s_t] = 0 # no transition allowed
            self.world['sas'][s][a][s] = 1
        
        self.T = self.world['sas']
        
    def get_transition(self) -> np.ndarray:
        
        return self.world['sas'].copy()

    def reset(self) -> np.ndarray:
        '''
        AI Gym's reset function.
        Resets the environment and the agent's state.

        Parameters
        ----------
        None

        Returns
        ----------
        observation :                       The observation of the new current state.
        '''
        # select randomly from possible starting states
        self.currentState = self.starting_states[np.random.randint(self.starting_states.shape[0])]
        self.observation = np.copy(self.O[self.currentState])
        if self.noise > 0.:
            self.observation = np.random.normal(self.observation, self.noise)
        self.currentStep = 0
        self.currentEpisode += 1
        self.trajectories.append([])

        return self.observation

    def initialize_visualization(self):
        '''
        This function initializes the elements required for visualization.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        if self.with_GUI:
            # check if observations can be visualized
            self.observation_type = 'unknown'
            if len(self.O.shape) == 2:
                self.observation_type = 'sensor'
            elif len(self.O.shape) == 3:
                self.observation_type = 'grey'
            elif len(self.O.shape) == 4:
                self.observation_type = 'color'
            # ensure validity of state coordinates
            if self.coordinates is None:
                angles = np.arange(self.O.shape[0]) * 2 * np.pi / self.O.shape[0]
                self.coordinates = np.zeros((self.O.shape[0], 2))
                self.coordinates[:, 0] = np.sin(angles)
                self.coordinates[:, 1] = np.cos(angles)
            # determine minimum and maximum coordinates
            self.coord_min, self.coord_max = np.amin(self.coordinates, axis=0), np.amax(self.coordinates, axis=0)
            # determine connections between states
            self.connections = []
            for state in range(self.T.shape[0]):
                for action in range(self.T.shape[1]):
                    for follow_up_state in range(self.T.shape[2]):
                        if self.T[state, action, follow_up_state] > 0 and not [state,
                                                                               follow_up_state] in self.connections:
                            self.connections += [[state, follow_up_state]]
            # state information panel
            self.state_information_panel = self.gui_parent.addPlot(title='State Information')
            self.state_information_panel.hideAxis('bottom')
            self.state_information_panel.hideAxis('left')
            self.state_information_panel.setXRange(0, 1)
            self.state_information_panel.setYRange(0, 1)
            self.state_information_panel.setAspectLocked()
            self.state_info = pg.TextItem(text='-1')
            self.state_label = pg.TextItem(text='Current State:')
            self.coord_info = pg.TextItem(text='(-1, -1)')
            self.coord_label = pg.TextItem(text='Current Coordinates:')
            self.font = pg.Qt.QtGui.QFont()
            self.font.setPixelSize(20)
            self.state_info.setFont(self.font)
            self.state_label.setFont(self.font)
            self.coord_info.setFont(self.font)
            self.coord_label.setFont(self.font)
            self.state_info.setPos(0.1, 0.95)
            self.state_label.setPos(0.1, 1.)
            self.coord_info.setPos(0.1, 0.8)
            self.coord_label.setPos(0.1, 0.85)
            if self.observation_type == 'unknown':
                self.observation_info = pg.TextItem('unknown')
                self.observation_info.setFont(self.font)
                self.observation_info.setPos(0.1, 0.55)
            else:
                self.observation_info = pg.ImageItem()
                self.observation_info.setOpts(axisOrder='row-major')
            self.observation_label = pg.TextItem('Current Observation:')
            self.observation_label.setFont(self.font)
            self.observation_label.setPos(0.1, 0.6)
            self.state_information_panel.addItem(self.state_info)
            self.state_information_panel.addItem(self.state_label)
            self.state_information_panel.addItem(self.coord_info)
            self.state_information_panel.addItem(self.coord_label)
            self.state_information_panel.addItem(self.observation_info)
            self.state_information_panel.addItem(self.observation_label)
            # behavior panel
            self.behavior_panel = self.gui_parent.addPlot(title='Behavior')
            width, height = (self.coord_max[0] - self.coord_min[0]), (self.coord_max[1] - self.coord_min[1])
            self.behavior_panel.setXRange(self.coord_min[0] - width * 0.05, self.coord_max[0] + width * 0.05)
            self.behavior_panel.setYRange(self.coord_min[1] - height * 0.05, self.coord_max[1] + height * 0.05)
            self.behavior_panel.setAspectLocked()
            self.state_graph = pg.GraphItem()
            symbolBrushes = [pg.mkBrush(color=(128, 128, 128))] * self.O.shape[0]
            for goal in self.goals:
                symbolBrushes[goal] = pg.mkBrush(color=(0, 255, 0))
            self.state_graph.setData(pos=np.array(self.coordinates), adj=np.array(self.connections),
                                     symbolBrush=symbolBrushes)
            self.behavior_panel.addItem(self.state_graph)

    def update_visualization(self):
        '''
        This function updates the visualization.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        if self.with_GUI:
            # update state information panel
            self.state_info.setText(str(self.currentState))
            self.coord_info.setText(str(self.coordinates[self.currentState]))
            if self.observation_type == 'unknown':
                self.observation_info.setText('Unknown format.')
            else:
                obs_data = None
                if self.observation_type == 'sensor':
                    obs_data = np.tile(self.observation, 3).reshape((1, self.O.shape[1], 3), order='F')
                elif self.observation_type == 'grey':
                    obs_data = np.tile(self.observation.reshape(self.O.shape[1], self.O.shape[2], 1), 3)
                else:
                    obs_data = self.observation
                self.observation_info.setImage(np.flip(obs_data, axis=0))
                self.observation_info.setRect(qt.QtCore.QRectF(0.0, 0.52 - obs_data.shape[0] / obs_data.shape[1], 1.,
                                                               obs_data.shape[0] / obs_data.shape[1]))
            # update behavior panel
            symbolBrushes = [pg.mkBrush(color=(128, 128, 128))] * self.O.shape[0]
            for goal in self.goals:
                symbolBrushes[goal] = pg.mkBrush(color=(0, 255, 0))
            symbolBrushes[self.currentState] = pg.mkBrush(color=(255, 0, 0))
            self.state_graph.setData(pos=np.array(self.coordinates), adj=np.array(self.connections),
                                     symbolBrush=symbolBrushes)
            # process changes
            if hasattr(qt.QtGui, 'QApplication'):
                qt.QtGui.QApplication.instance().processEvents()
            else:
                qt.QtWidgets.QApplication.instance().processEvents()

    def get_position(self) -> np.ndarray:
        '''
        This function returns the agent's position in the environment.

        Parameters
        ----------
        None

        Returns
        ----------
        position :                          Numpy array containing the agent's position.
        '''
        position = np.array([])
        if self.coordinates is not None:
            position = np.copy(self.coordinates[self.currentState])

        return position
