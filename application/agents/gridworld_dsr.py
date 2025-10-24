# basic imports
import numpy as np
# keras imports
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense

# memory module
from application.memory_modules.dyna_memory import FreqMemory

class AbstractDynaQAgent():
    '''
    Implementation of a Dyna-Q agent.
    Q-function is represented as a static table.

    | **Args**
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learningRate:                 The learning rate with which the Q-function is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    '''

    def __init__(self, interfaceOAI, epsilon=0.3, beta=5, learningRate=0.9, gamma=0.9):
        # store the Open AI Gym interface
        self.interfaceOAI = interfaceOAI
        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.numberOfActions = self.interfaceOAI.action_space.n
        self.numberOfStates = self.interfaceOAI.world['states']
        # Q-learning parameters
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
        self.learningRate = learningRate
        self.policy = 'greedy'
        # mask invalid actions?
        self.mask_actions = False

    def replay(self, replayBatchSize=200):
        '''
        This function replays experiences to update the Q-function.

        | **Args**
        | replayBatchSize:              The number of random that will be replayed.
        '''
        # sample random batch of experiences
        replayBatch = self.M.retrieve_batch(replayBatchSize)
        # update the Q-function with each experience
        for experience in replayBatch:
            self.update_Q(experience)

    def select_action(self, state, state_index, epsilon=0.3, beta=5):
        '''
        This function selects an action according to the Q-values of the current state.

        | **Args**
        | state:                        The current state.
        | epsilon:                      The epsilon parameter used under greedy action selection.
        | beta:                         The temperature parameter used when applying the softmax function to the Q-values.
        '''
        # revert to 'greedy' in case that the method name is not valid
        if not self.policy in ['greedy', 'softmax', 'inhib']:
            self.policy = 'greedy'
        # retrieve Q-values
        qVals = self.retrieve_Q(state)
        actions = np.arange(qVals.shape[0])
        
        # actions_te_b = np.full(self.numberOfActions, False)
        # remove masked actions
        if self.mask_actions:
            self.compute_action_mask()
            # a exploration policy for these unknown places.
            qVals = qVals[self.action_mask[state_index]]
            actions = actions[self.action_mask[state_index]]
        
        # select action with highest value
        if self.policy == 'greedy':
            # act greedily and break ties
            action = np.argmax(qVals)
            # in case that Q-values are equal select a random action
            ties = np.arange(qVals.shape[0])[(qVals == qVals[action])]
            action = ties[np.random.randint(ties.shape[0])]
            if np.random.rand() < epsilon:
                action = np.random.randint(qVals.shape[0])
            return int(actions[action])
        # select action probabilistically
        elif self.policy == 'softmax':
            qVals -= np.amax(qVals)
            probs = np.exp(beta * qVals) / np.sum(np.exp(beta * qVals))
            action = np.random.choice(qVals.shape[0], p=probs)
            return int(actions[action])

    def compute_action_mask(self):
        '''
        This function computes the action mask which prevents the selection of invalid actions.
        '''
        # retrieve number of states and actions
        s, a = self.interfaceOAI.world['states'], self.numberOfActions
        # determine follow-up states
        self.action_mask = self.interfaceOAI.world['sas'].reshape((s * a, s), order='F')
        self.action_mask = np.argmax(self.action_mask, axis=1)
        # make action mask
        self.action_mask = (self.action_mask != np.tile(np.arange(s), a)).reshape((s, a), order='F')


    def compute_exploration_action_mask(self):
        '''
        This function computes the action mask for unexplored states from a particular state
        '''
        # retrieve explored actions for the current state, explored actions are true here
        possible_actions = np.copy(self.interfaceOAI.explored_actions[self.interfaceOAI.currentState])
        # now turn the unexplored actions to be true
        for i in range(len(possible_actions)): possible_actions[i] = not possible_actions[i]
        # mask with the valid/invalid action masks
        possible_actions *= self.action_mask[self.interfaceOAI.currentState]
        return possible_actions

    def train(self, numberOfTrials=100, maxNumberOfSteps=50, replayBatchSize=100, noReplay=False):
        '''
        This function is called to train the agent.

        | **Args**
        | numberOfTrials:               The number of trials the Dyna-Q agent is trained.
        | maxNumberOfSteps:             The maximum number of steps per trial.
        | replayBatchSize:              The number of random that will be replayed.
        | noReplay:                     If true, experiences are not replayed.
        '''
        raise NotImplementedError('.train() function not implemented!')

    def update_Q(self, experience):
        '''
        This function updates the Q-function with a given experience.

        | **Args**
        | experience:                   The experience with which the Q-function will be updated.
        '''
        raise NotImplementedError('.update_Q() function not implemented!')

    def retrieve_Q(self, state):
        '''
        This function retrieves Q-values for a given state.

        | **Args**
        | state:                        The state for which Q-values should be retrieved.
        '''
        raise NotImplementedError('.retrieve_Q() function not implemented!')

    def retrieve_action_inhibition(self, state):
        '''
        This function retrieves Q-values for a given state.

        | **Args**
        | state:                        The state for which Q-values should be retrieved.
        '''
        raise NotImplementedError('.retrieve_Q() function not implemented!')

    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.

        | **Args**
        | batch:                        The batch of states.
        '''
        raise NotImplementedError('.predict_on_batch() function not implemented!')


class SimpleDSR(AbstractDynaQAgent):
    '''
    Implementation of a Deep Successor Representation agent using the Dyna-Q model.
    This agent uses the Dyna-Q agent's memory module and then maps gridworld states to predefined observations.
    
    | **Args**
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learningRate:                 The learning rate with which the SR is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    | trialEndFcn:                  The callback function called at the end of each trial, defined for more flexibility in scenario control.
    | observations:                 The set of observations that will be mapped to the gridworld states.
    | model:                        The DNN model to be used by the agent.
    '''
    
    class callbacks():
        '''
        Callback class. Used for visualization and scenario control.
        
        | **Args**
        | rlParent:                     Reference to the Dyna-Q agent.
        | trialEndFcn:                  Maximum number of experiences that will be stored by the memory module.
        '''

        def __init__(self, rlParent, trialEndFcn=None):
            # store the hosting class
            self.rlParent = rlParent
            # store the trial end callback function
            self.trialEndFcn = trialEndFcn
        
        def on_episode_end(self, epoch, logs):
            '''
            The following function is called whenever an episode ends, and updates the reward accumulator,
            simultaneously updating the visualization of the reward function.
            
            | **Args**
            | rlParent:                     Reference to the Dyna-Q agent.
            | trialEndFcn:                  Maximum number of experiences that will be stored by the memory module.
            '''
            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch, self.rlParent, logs)
                
            
    def __init__(self, interfaceOAI, epsilon=0.3, beta=5, learningRate=0.9, gamma=0.9, trialEndFcn=None, observations=None, model=None):
        super().__init__(interfaceOAI, epsilon=epsilon, beta=beta, learningRate=learningRate, gamma=gamma)
        # prepare observations
        if observations is None or observations.shape[0] != self.numberOfStates:
            # one-hot encoding of states
            self.observations = np.eye(self.numberOfStates)
        else:
            self.observations = observations
        # build target and online models
        self.build_models()
        # memory module
        self.M = FreqMemory(self.numberOfStates, self.numberOfActions)
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engagedCallbacks = self.callbacks(self, trialEndFcn)
        # perform replay at the end of an episode instead of each step
        self.episodic_replay = False
        # the rate at which the target model is updated (for values < 1 the target model is blended with the online model)
        self.target_model_update = 10**-2
        # count the steps since the last update of the target model
        self.steps_since_last_update = 0
        # compute DR instead of SR
        self.deep_dr = False
        # computes SR based on the follow-up state (i.e. each action stream represents the SR of the follow-up state)
        self.use_follow_up_state = False
        # ignores the terminality of states when computing the target values
        self.ignore_terminality = True
        # the design of the pre-exposure
        self.design = 'targeted'
        # logging
        self.steps = []
        self.action_mask = True
        # record the number of trials
        self.trial_number = 0


    def build_models(self, model=None):
        '''
        This function builds the Dyna-DSR's target and online models.
        
        | **Args**
        | model:                        The DNN model to be used by the agent (not working currently). If None, a small dense DNN is created by default.
        '''
        # build target and online models for all actions
        self.models_target, self.models_online = {}, {}
        for action in range(self.numberOfActions):
            # build target model
            self.models_target[action] = Sequential()
            self.models_target[action].add(Dense(units=64, input_shape=(self.observations.shape[1],), activation='tanh'))
            self.models_target[action].add(Dense(units=64, activation='tanh'))
            self.models_target[action].add(Dense(units=self.observations.shape[1], activation='linear'))
            self.models_target[action].compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            # build online model
            self.models_online[action] = clone_model(self.models_target[action])
            self.models_online[action].compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        # build reward model which needs to be a linear model in theory
        self.model_reward = Sequential()
        # linear reward model
        self.model_reward.add(Dense(units=1, input_shape=(self.observations.shape[1],), activation='linear', use_bias=False))
        
        self.model_reward.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        print(self.models_target[0].summary())
        print(self.model_reward.summary())
        
    def train(self, numberOfTrials=100, maxNumberOfSteps=50, replayBatchSize=100, noReplay=False):
        '''
        This function is called to train the agent.
        
        | **Args**
        | numberOfTrials:               The number of trials the Dyna-Q agent is trained.
        | maxNumberOfSteps:             The maximum number of steps per trial.
        | replayBatchSize:              The number of random that will be replayed.
        | noReplay:                     If true, experiences are not replayed.
        '''
        for trial in range(numberOfTrials):
            # reset environment
            state = self.interfaceOAI.reset()
            state_idx = self.interfaceOAI.currentState
            # log cumulative reward
            logs = {'episode_reward': 0}
            self.action_inhibtion = np.zeros((self.numberOfStates, self.numberOfActions))
            for step in range(maxNumberOfSteps):
                self.steps_since_last_update += 1
                action=self.select_action(state, state_idx, self.epsilon, self.beta)
                next_state, reward, stopEpisode, _ = self.interfaceOAI.step(action)
                next_state_idx = self.interfaceOAI.currentState

                # make experience, we store the state index
                experience = {'state': state_idx, 'action': action, 'reward': reward, 'next_state': next_state_idx, 'terminal': (1 - stopEpisode)}
                self.M.store(experience)

                # to store the last experience transition to itself in memory module
                if (self.design == 'mistargeted' and next_state_idx == 9) or (self.design == 'targeted' and next_state_idx == self.numberOfStates - 1) or (self.design == 'continuous' and next_state_idx == self.numberOfStates - 1 and self.trial_number > self.interfaceOAI.rewarding_episode): 
                    experience={'state': next_state_idx, 'action': self.get_opposite(action), 'reward': reward, 'next_state': next_state_idx, 'terminal': (1 - stopEpisode)}
                    self.M.store(experience)
                    
                state = next_state
                state_idx = next_state_idx
                
                # perform experience replay
                if not noReplay and not self.episodic_replay:
                    self.replay(replayBatchSize)
                
                # update cumulative reward
                logs['episode_reward'] += reward
                # stop trial when the terminal state is reached
                if stopEpisode:
                    break

            self.steps += [step + 1]
            print('Trial: %s, episode step: %s'%(trial, step))

            if not noReplay and self.episodic_replay:
                self.replay(replayBatchSize)
            
            # callback
            self.engagedCallbacks.on_episode_end(self.trial_number, logs)
            self.trial_number += 1

    def replay(self, replayBatchSize=200):
        '''
        This function replays experiences to update the DSR and reward function.
        
        | **Args**
        | replayBatchSize:              The number of random that will be replayed.
        '''
        # sample random batch of experiences
        
        replayBatch, replayBatchSize = self.M.retrieve_batch(replayBatchSize) # FreqMemory
        states, next_states = np.zeros((replayBatchSize, self.observations.shape[1])), np.zeros((replayBatchSize, self.observations.shape[1]))
        states_idx = []
        rewards, terminals, actions = np.zeros(replayBatchSize),  np.zeros(replayBatchSize),  np.zeros(replayBatchSize)
        for e, experience in enumerate(replayBatch):
            states[e] = self.observations[experience['state']]
            states_idx.append(experience['state'])
            next_states[e] = self.observations[experience['next_state']]
            rewards[e] = experience['reward']
            actions[e] = experience['action']
            terminals[e] = experience['terminal']
        # compute the follow-up states' SR streams and values    
        future_SR, future_values = {}, {}
        for action in range(self.numberOfActions):
            future_SR[action] = self.models_target[action].predict_on_batch(next_states)
            future_values[action] = self.model_reward.predict_on_batch(future_SR[action])
        # compute targets
        inputs, targets = {}, {}
        states_idx = np.asarray(states_idx)
        for action in range(self.numberOfActions):
            # filter out experiences irrelevant for this action stream
            idx = (actions == action)
            inputs[action] = states[idx]
            if self.use_follow_up_state:
                targets[action] = next_states[idx]
            else:
                targets[action] = states[idx]
            # compute indices of relevant experiences
            idx = np.arange(len(replayBatch))[idx]
            for i, index in enumerate(idx):
                # Deep SR
                if not self.deep_dr:
                    best = np.argmax(np.array([future_values[action][index] for action in future_values])) 
                    targets[action][i] += self.gamma * future_SR[best][index]
                    
                # Deep DR
                else:
                    targets[action][i] += self.gamma * np.mean(np.array([future_SR[stream][index] for stream in future_SR]), axis=0)
                    
        # update online models
        for action in range(self.numberOfActions):
            if inputs[action].shape[0] > 0:
                self.models_online[action].train_on_batch(inputs[action], targets[action])
        # update reward model
        self.model_reward.train_on_batch(np.array(next_states), np.array(rewards))
        
        # update target models
        if self.target_model_update < 1.:
            for action in range(self.numberOfActions):
                # retrieve weights
                weights_target = self.models_target[action].get_weights()
                weights_online = self.models_online[action].get_weights()
                # blend weights
                for layer in range(len(weights_target)):
                    weights_target[layer] += self.target_model_update * (weights_online[layer] - weights_target[layer])
                self.models_target[action].set_weights(weights_target)
            # reset update timer
            self.steps_since_last_update = 0
        elif self.steps_since_last_update >= self.target_model_update:
            for action in range(self.numberOfActions):
                # copy weights from online model
                self.models_target[action].set_weights(self.model_online[action].get_weights())
            # reset update timer
            self.steps_since_last_update = 0

    def get_opposite(self, action):
        '''
        This function is used to get the opposite action in the gridworld.
        | **Args**
        | action:                       The action for which the opposite action is required.
        '''
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        return opposites.get(action) 
    
    def update_Q(self, experience):
        '''
        This function is a dummy function and does nothing (implementation required by parent class).
        
        | **Args**
        | experience:                   The experience with which the Q-function will be updated.
        '''
        pass
    
    def retrieve_Q(self, state):
        '''
        This function retrieves Q-values for a given state.
        
        | **Args**
        | state:                        The state for which Q-values should be retrieved.
        '''
        Q = []        
        for action in range(self.numberOfActions):
            SR = self.models_online[action].predict_on_batch(np.array([state]))[0]
            Q += [self.model_reward.predict_on_batch(np.array([SR]))[0]]
        Q = np.array(Q)[:, 0]

        return Q
    
    def retrieve_SR(self, state):
        '''
        This function computes (predicts) the SR of a particular state.
        
        | **Args**
        | state:                        The desired state to compute the SR.
        '''
        SR = []
        for action in range(self.numberOfActions):
            SR_action = self.models_online[action].predict_on_batch(np.array([self.observations[state]]))[0]
            SR += [SR_action]
        SR = np.array(SR)
        return SR
    
    def retrieve_reward_function(self, reacheable_states):
        '''
        This function retrieves the learned reward function from the states.
        
        | **Args**
        | reacheable_states:                        The states the agent can visit.
        '''
        mask_observation = []
        for r_states in reacheable_states:
            mask_observation.append(np.array(self.observations[r_states][:]))
        retrieved_reward = np.array(self.model_reward.predict_on_batch(np.array(mask_observation)))
        return retrieved_reward
    
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.
        
        | **Args**
        | batch:                        The batch of states.
        '''
        Q = []
        for action in range(self.numberOfActions):
            SR = self.models_online[action].predict_on_batch(self.observations[batch])
            Q += [self.model_reward.predict_on_batch(SR)[:, 0]]
        
        return np.array(Q).T

