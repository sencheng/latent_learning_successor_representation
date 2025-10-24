# basic imports
import numpy as np
from tensorflow.keras.layers import Flatten
# framework imports
from cobel.agents.dyna_q import AbstractDynaQAgent
from cobel.memory_modules.dyna_memory import FreqMemory


class DynaDQN(AbstractDynaQAgent):

    def __init__(self, interface_OAI, epsilon: float = 0.3, beta: float = 5, gamma: float = 0.95, observations=None, model=None, custom_callbacks={}):
        '''
        Implementation of a DQN agent using the Dyna-Q model.
        This agent uses the Dyna-Q agent's memory module and then maps gridworld states to predefined observations.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        epsilon :                           The epsilon value for the epsilon greedy policy.
        beta :                              The inverse temperature parameter for the softmax policy.
        gamma :                             The discount factor used for computing the target values.
        observations :                      The set of observations that will be mapped to the gridworld states.
        model :                             The network model to be used by the agent.
        custom_callbacks :                  The custom callbacks defined by the user.
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, epsilon=epsilon, beta=beta, learning_rate=0.9, gamma=gamma, custom_callbacks=custom_callbacks)
        # prepare observations
        if observations is None or observations.shape[0] != self.number_of_states:
            # one-hot encoding of states
            self.observations = np.eye(self.number_of_states)
        else:
            self.observations = observations
        # prepare target and online models
        self.prepare_models(model)
        self.current_predictions = self.model_online.predict_on_batch(self.observations)
        # memory module
        # self.M = DynaQMemory(self.number_of_states, self.number_of_actions) -> changing to freqmemory
        self.M = FreqMemory(self.number_of_states, self.number_of_actions)
        # perform replay at the end of an episode instead of each step
        self.episodic_replay = False
        # the rate at which the target model is updated (for values < 1 the target model is blended with the online model)
        self.target_model_update = 10**-2
        # count the steps since the last update of the target model
        self.steps_since_last_update = 0
        # design
        self.design = 'direct'
        
    def prepare_models(self, model=None):
        '''
        This functions prepares target and online models. 
        
        Parameters
        ----------
        model :                             The network model to be used by the agent.
        
        Returns
        ----------
        None
        '''
        # prepare target model
        self.model_target = model
        # prepare online model by cloning the target model
        self.model_online = self.model_target.clone_model()
        
    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 50, replay_batch_size: int = 100, no_replay: bool = False):
        '''
        This functions is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the Dyna-Q agent is trained.
        max_number_of_steps :               The maximum number of steps per trial.
        replay_batch_size :                 The number of random that will be replayed.
        no_replay :                         If true, experiences are not replayed.
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                self.steps_since_last_update += 1
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                # store experience
                self.M.store(experience)
                # matheus -> store the last experience to last experience transition
                if (self.design != 'continuous' and next_state == self.number_of_states - 1) or (self.design == 'continuous' and trial > self.interface_OAI.rewarding_episode and next_state == self.number_of_states - 1): 
                    experience={'state': next_state, 'action': self.get_opposite(action), 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                    self.M.store(experience)
                # update current state
                state = next_state
                # perform experience replay
                if not no_replay and not self.episodic_replay:
                    self.replay(replay_batch_size)
                # update cumulative reward
                logs['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # to save performance store state predictions after each trial only
            self.current_predictions = self.model_online.predict_on_batch(self.observations)
            # perform experience replay
            if not no_replay and self.episodic_replay:
                self.replay(replay_batch_size)
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            
    def test(self, number_of_trials: int = 100, max_number_of_steps: int = 50):
        '''
        This functions is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the Dyna-Q agent is tested.
        max_number_of_steps :               The maximum number of steps per trial.
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta, True)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # to save performance store state predictions after each trial only
            self.current_predictions = self.model_online.predict_on_batch(self.observations)
            # callback
            self.engaged_callbacks.on_trial_end(logs)
        
    def replay(self, replay_batch_size: int = 200):
        '''
        This functions replays experiences to update the Q-function.
        
        Parameters
        ----------
        replay_batch_size :                 The number of random experiences that will be replayed.
        
        Returns
        ----------
        None
        '''
        # sample random batch of experiences
        replay_batch, replay_batch_size = self.M.retrieve_batch(replay_batch_size)
        # compute update targets
        states, next_states = np.zeros((replay_batch_size)), np.zeros((replay_batch_size))
        rewards, terminals, actions = np.zeros(replay_batch_size),  np.zeros(replay_batch_size),  np.zeros(replay_batch_size)
        for e, experience in enumerate(replay_batch):
            states[e] = experience['state']
            next_states[e] = experience['next_state']
            rewards[e] = experience['reward']
            actions[e] = experience['action']
            terminals[e] = experience['terminal']
        future_values = np.amax(self.model_target.predict_on_batch(self.observations[next_states.astype(int)]), axis=1) * terminals
        targets = self.model_target.predict_on_batch(self.observations[states.astype(int)])
        for a, action in enumerate(actions):
            targets[a, int(action)] = rewards[a] + self.gamma * future_values[a]
        # update online model
        self.model_online.train_on_batch(np.array(self.observations[states.astype(int)]), np.array(targets))
        # update target model
        if self.target_model_update < 1.:
            weights_target = np.array(self.model_target.get_weights(), dtype=object)
            weights_online = np.array(self.model_online.get_weights(), dtype=object)
            weights_target += self.target_model_update * (weights_online - weights_target)
            self.model_target.set_weights(weights_target)
        elif self.steps_since_last_update >= self.target_model_update:
            self.model_target.set_weights(self.model_online.get_weights())
            self.steps_since_last_update = 0
        
    def update_Q(self, experience: dict):
        '''
        This function is a dummy function and does nothing (implementation required by parent class).
        
        Parameters
        ----------
        experience :                        The number of random experiences that will be replayed.
        
        Returns
        ----------
        None
        '''
        pass
    
    def retrieve_Q(self, state: int) -> np.ndarray:
        '''
        This function retrieves Q-values for a given state.
        
        Parameters
        ----------
        state :                             The state for which Q-values should be retrieved.
        
        Returns
        ----------
        q_values :                          The state's predicted Q-values.
        '''
        return self.model_online.predict_on_batch(np.array([self.observations[state]]))[0]
    
    def predict_on_batch(self, batch: np.ndarray) -> np.ndarray:
        '''
        This function retrieves Q-values for a batch of states.
        
        Parameters
        ----------
        batch :                             The batch of states for which Q-values should be retrieved.
        
        Returns
        ----------
        predictions :                       The batch of predicted Q-values.
        '''
        return self.current_predictions[batch]

    def get_opposite(self, action):
        '''
        This function is a dummy function and does nothing (implementation required by parent class).
        
        Parameters
        ----------
        experience :                        The number of random experiences that will be replayed.
        
        Returns
        ----------
        None
        '''
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        return opposites.get(action) 
    
    
class DynaDSR(AbstractDynaQAgent):
           
    def __init__(self, interface_OAI, epsilon: float = 0.3, beta: float = 5, gamma: float = 0.99, observations=None, model_SR=None, model_reward=None, custom_callbacks={}):
        '''
        Implementation of a Deep Successor Representation agent using the Dyna-Q model.
        This agent uses the Dyna-Q agent's memory module and then maps gridworld states to predefined observations.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        epsilon :                           The epsilon value for the epsilon greedy policy.
        beta :                              The inverse temperature parameter for the softmax policy.
        gamma :                             The discount factor used for computing the target values.
        observations :                      The set of observations that will be mapped to the gridworld states.
        model_SR :                          The network model to be used by the agent for the successor representation.
        model_reward :                      The network model to be used by the agent for the reward model.
        custom_callbacks :                  The custom callbacks defined by the user.
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, epsilon=epsilon, beta=beta, learning_rate=0.9, gamma=gamma, custom_callbacks=custom_callbacks)
        # prepare observations
        if observations is None:
            print('Using one-hot encoding of states')
            # one-hot encoding of states
            self.observations = np.eye(self.number_of_states)
        else:
            self.observations = observations
        # prepare target and online models
        self.prepare_models(model_SR)
        # reward model
        self.model_reward = model_reward
        # used for visualization (currently all-zeros array)        
        self.current_predictions = np.zeros((self.number_of_states, self.number_of_actions))
        # memory module
        # self.M = DynaQMemory(self.number_of_states, self.number_of_actions)
        self.M = FreqMemory(self.number_of_states, self.number_of_actions)
        # perform replay at the end of an episode instead of each step
        self.episodic_replay = False
        # the rate at which the target model is updated (for values < 1 the target model is blended with the online model)
        self.target_model_update = 10**-2
        # count the steps since the last update of the target model
        self.steps_since_last_update = 0   
        # compute DR instead of SR
        self.use_Deep_DR = False
        # computes SR based on the follow-up state (i.e. each action stream represents the SR of the follow-up state)
        self.use_follow_up_state = True
        # ignores the terminality of states when computing the target values
        self.ignore_terminality = True
        # design
        self.design = 'direct'
        # state-action-probability
        self.compute_transition_probs = False
        self.transition_probs = np.zeros((2, self.number_of_states))
        self.transition_probs_episodes = {}
        for i in range(100): # number of maximum trials
            self.transition_probs_episodes[i] = np.zeros((2, self.number_of_states))
        
    def prepare_models(self, model_SR):
        '''
        This function prepares the Dyna-DSR's target and online models.
        
        Parameters
        ----------
        model_SR :                          The DNN model to be used for the SR.
        
        Returns
        ----------
        None
        '''
        # prepare target and online models for all actions
        self.models_target, self.models_online = {}, {}
        for action in range(self.number_of_actions):
            # prepare target model
            self.models_target[action] = model_SR.clone_model()
            # prepare online model
            self.models_online[action] = model_SR.clone_model()
        
    def train(self, number_of_trials: int = 100, max_number_of_steps: int = 50, replay_batch_size: int = 100, no_replay: bool = False):
        '''
        This functions is called to train the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the Dyna-Q agent is trained.
        max_number_of_steps :               The maximum number of steps per trial.
        replay_batch_size :                 The number of random that will be replayed.
        no_replay :                         If true, experiences are not replayed.
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            # reset transition probabilities
            if self.compute_transition_probs:
                self.reset_transition_probs()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                self.steps_since_last_update += 1
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta)
                if self.compute_transition_probs:
                    self.update_transition_probs(state, action) # updating transition probs
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                # store experience
                self.M.store(experience)
                # store the last experience to last experience transition
                if (self.design == 'targeted' and next_state == self.number_of_states - 1) or (self.design == 'mistargeted' and (trial <= self.interface_OAI.rewarding_episode and next_state == 9) or (trial > self.interface_OAI.rewarding_episode and next_state == self.number_of_states - 1)) or (self.design == 'continuous' and trial > self.interface_OAI.rewarding_episode and next_state == self.number_of_states - 1): 
                    experience={'state': next_state, 'action': self.get_opposite(action), 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                    self.M.store(experience)
                # update current state
                state = next_state
                # perform experience replay
                if not no_replay and not self.episodic_replay:
                    self.replay(replay_batch_size)
                # update cumulative reward
                logs['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    print(f'Trial {trial} ended at step {step} with reward {reward}')
                    break
            self.current_trial += 1
            logs['steps'] = step
            # perform experience replay
            if not no_replay and self.episodic_replay:
                self.replay(replay_batch_size)
            # callback
            self.engaged_callbacks.on_trial_end(logs)
            # update transition probabilities
            self.transition_probs_episodes[trial] = self.transition_probs
            
    def test(self, number_of_trials: int = 100, max_number_of_steps: int = 50):
        '''
        This functions is called to test the agent.
        
        Parameters
        ----------
        number_of_trials :                  The number of trials the Dyna-Q agent is tested.
        max_number_of_steps :               The maximum number of steps per trial.
        
        Returns
        ----------
        None
        '''
        for trial in range(number_of_trials):
            # log cumulative reward
            logs = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial}
            # callback
            self.engaged_callbacks.on_trial_begin(logs)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                self.engaged_callbacks.on_step_begin(logs)
                self.steps_since_last_update += 1
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta, True)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                self.engaged_callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(logs)
        
    def replay(self, replay_batch_size: int = 200):
        '''
        This function replays experiences to update the DSR and reward function.
        
        Parameters
        ----------
        replay_batch_size :                 The number of random experiences that will be replayed.
        
        Returns
        ----------
        None
        '''
        # sample random batch of experiences
        replay_batch, replay_batch_size = self.M.retrieve_batch(replay_batch_size) # FreqMemory
        # recover variables from experiences)
        states, next_states = np.zeros((replay_batch_size)), np.zeros((replay_batch_size))
        # states, next_states = np.zeros((replay_batch_size, self.observations.shape)), np.zeros((replay_batch_size, self.observations.shape))
        rewards, terminals, actions = np.zeros(replay_batch_size),  np.zeros(replay_batch_size),  np.zeros(replay_batch_size)
        for e, experience in enumerate(replay_batch):
            # matheus -> state, next_state are scalars
            states[e] = experience['state']
            next_states[e] = experience['next_state']
            rewards[e] = experience['reward']
            actions[e] = experience['action']
            terminals[e] = experience['terminal']
        
        if len(states) > 0:
            observation_shape = np.prod(self.observations[states[0].astype(int)].shape).astype(int)
        # compute the follow-up states' SR streams and values    
        future_SR, future_values = {}, {}
        for action in range(self.number_of_actions):
            # for each state, predcit the future sr from self.observations[state]
            future_SR[action] = self.models_target[action].predict_on_batch(self.observations[next_states.astype(int)]) 
            future_values[action] = self.model_reward.predict_on_batch(future_SR[action])
        # compute targets
        inputs, targets = {}, {}
        for action in range(self.number_of_actions):
            # filter out experiences irrelevant for this action stream
            idx = (actions == action)
            # matheus -> the inputs are the current observation of states in the relevant indexes
            inputs[action] = self.observations[states[idx].astype(int)]
            if self.use_follow_up_state:
                # the targeteds are the next states from the current input
                targets[action] = self.observations[next_states[idx].astype(int)]
            else:
                targets[action] = self.observations[states[idx].astype(int)]
            
            # compute indices of relevant experiences
            idx = np.arange(len(replay_batch))[idx]
            for i, index in enumerate(idx):
                # Deep SR
                if not self.use_Deep_DR:
                    best = np.argmax(np.array([future_values[action][index] for action in future_values]))
                    targets[action][i] += self.gamma * future_SR[best][index] 
                # Deep DR
                else:
                    targets[action][i] += self.gamma * np.mean(np.array([future_SR[stream][index] for stream in future_SR]), axis=0)

        # update online models
        for action in range(self.number_of_actions):
            if inputs[action].shape[0] > 0:
                self.models_online[action].train_on_batch(inputs[action], targets[action])
        # update reward model
        self.model_reward.train_on_batch(self.observations[next_states.astype(int)], np.array(rewards))
        
        # update target models
        if self.target_model_update < 1.:
            for action in range(self.number_of_actions):
                weights_target = np.array(self.models_target[action].get_weights(), dtype=object)
                weights_online = np.array(self.models_online[action].get_weights(), dtype=object)
                weights_target += self.target_model_update * (weights_online - weights_target)
                self.models_target[action].set_weights(weights_target)
        elif self.steps_since_last_update >= self.target_model_update:
            for action in range(self.number_of_actions):
                # copy weights from online model
                self.models_target[action].set_weights(self.model_online[action].get_weights())
            # reset update timer
            self.steps_since_last_update = 0
    
    def get_opposite(self, action):
        '''
        This function is a dummy function and does nothing (implementation required by parent class).
        
        Parameters
        ----------
        experience :                        The number of random experiences that will be replayed.
        
        Returns
        ----------
        None
        '''
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        return opposites.get(action) 
    
    def update_Q(self, experience: dict):
        '''
        This function is a dummy function and does nothing (implementation required by parent class).
        
        Parameters
        ----------
        experience :                        The number of random experiences that will be replayed.
        
        Returns
        ----------
        None
        '''
        pass
    
    def retrieve_SR(self, state):
        '''
        This function computes (predicts) the SR of a particular state.
        
        | **Args**
        | state:                        The desired state to compute the SR.
        '''
        SR = []
        for action in range(self.number_of_actions):
            SR_action = self.models_online[action].predict_on_batch(np.array([self.observations[state]]))[0]
            SR += [SR_action]
        SR = np.array(SR)
        return SR
    
    def retrieve_SR_observation(self, observation):
        '''
        This function computes (predicts) the SR of a particular observation.
        
        | **Args**
        | observation:                        The desired observation to compute the SR.
        '''
        SR = []
        for action in range(self.number_of_actions):
            SR_action = self.models_online[action].predict_on_batch(np.array([observation]))[0]
            SR += [SR_action]
        SR = np.array(SR)
        return SR

    def retrieve_Q(self, state: int) -> np.ndarray:
        '''
        This function retrieves Q-values for a given state.
        
        Parameters
        ----------
        state :                             The state for which Q-values should be retrieved.
        
        Returns
        ----------
        q_values :                          The state's predicted Q-values.
        '''
        Q = []
        for action in range(self.number_of_actions):
            SR = self.models_online[action].predict_on_batch(np.array([self.observations[state]]))[0]
            Q += [self.model_reward.predict_on_batch(np.array([SR]))[0]]
        Q = np.array(Q)[:, 0]
        self.current_predictions[state] = Q
        
        return Q
    
    def retrieve_Q_observation(self, observation: np.ndarray) -> np.ndarray:
        '''
        This function retrieves Q-values for a given observation.
        
        Parameters
        ----------
        observation :                       The observation for which Q-values should be retrieved.
        
        Returns
        ----------
        q_values :                          The observation's predicted Q-values.
        '''
        Q = []
        for action in range(self.number_of_actions):
            SR = self.models_online[action].predict_on_batch(np.array([observation]))[0]
            Q += [self.model_reward.predict_on_batch(np.array([SR]))[0]]
        Q = np.array(Q)[:, 0]
        
        return Q
    
        
    def predict_on_batch(self, batch: np.ndarray) -> np.ndarray:
        '''
        This function retrieves Q-values for a batch of states.
        
        Parameters
        ----------
        batch :                             The batch of states for which Q-values should be retrieved.
        
        Returns
        ----------
        predictions :                       The batch of predicted Q-values.
        '''
        Q = []
        for action in range(self.number_of_actions):
            SR = self.models_online[action].predict_on_batch(batch)
            Q += [self.model_reward.predict_on_batch(SR)[:, 0]]
        
        return np.array(Q).T
    
    def update_transition_probs(self, state, action):
        '''
        This function updates the probability of the agent select the optimal action in a particular state.

        | **Args**
        | states:                        The current agent state.
        | action:                        The action selected by the agent.
        '''
        optimal_actions_tolman = [1, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 0, 0, 3, 3, 2, 2, 3, 3, 3]
        
        self.transition_probs[0][state] += 1
        if action == optimal_actions_tolman[state]:
            self.transition_probs[1][state] += 1
    
    def reset_transition_probs(self):
        '''
        This function simply resets the state transition probability matrix.

        | **Args**
        |
        '''
        self.transition_probs = np.zeros((2, self.number_of_states))
