# basic imports
import numpy as np
import math


class FreqMemory():
    '''
    Memory whose items are retreived based on their visited frequency
    Experiences are stored as a static table.

    | **Args**
    | numberOfStates:               The number of states in the env.
    | numberOfActions:              The number of the agent's actions.
    | learningRate:                 The learning rate with which reward experiences are updated.
    '''

    def __init__(self, numberOfStates, numberOfActions, decay_strength=1., learningRate=0.9):
        # initialize variables
        self.numberOfStates = numberOfStates
        self.numberOfActions = numberOfActions
        self.decay_strength = decay_strength
        self.learningRate = learningRate
        # prepare memory structures
        self.rewards = np.zeros((self.numberOfStates, self.numberOfActions))
        self.states = np.tile(np.arange(self.numberOfStates).reshape(self.numberOfStates, 1),
                              self.numberOfActions).astype(int)
        self.terminals = np.ones((self.numberOfStates, self.numberOfActions)).astype(int)
        # prepare replay-relevant structures
        self.C = np.zeros(self.numberOfStates * self.numberOfActions)  # strength
        # increase step size
        self.C_step = 1.


    def store(self, experience):
        '''
        This function stores a given experience.

        | **Args**
        | experience:                   The experience to be stored.
        '''
        state, action = experience['state'], experience['action']
        # update experience
        self.rewards[state][action] += self.learningRate * (experience['reward'] - self.rewards[state][action])
        self.states[state][action] = experience['next_state']
        self.terminals[state][action] = experience['terminal']
        # update replay-relevent structures
        self.C *= self.decay_strength
        self.C[self.numberOfStates * action + state] = self.C_step
        # just for testing
        for action in range(4):
            self.C[self.numberOfStates * action + state] = self.C_step


    def retrieve_batch(self, replayLength):
        '''
        This function replays experiences.

        | **Args**
        | replayLength:                 The number of experiences that will be replayed.
        '''
        # replay
        experiences = []
        replayLength = int(min(replayLength, np.sum(self.C)))
        # retrieve experience strengths
        R = np.copy(self.C)
        # compute activation probabilities
        probs = R / np.sum(R)
        for step in range(replayLength):
            exp = np.random.choice(np.arange(0, probs.shape[0]), p=probs)
            # determine experience tuple
            action = int(exp / self.numberOfStates)
            current_state = exp - (action * self.numberOfStates)
            next_state = self.states[current_state][action]
            # "reactivate" experience
            experience = {'state': current_state, 'action': action, 'reward': self.rewards[current_state][action],
                          'next_state': next_state, 'terminal': self.terminals[current_state][action]}
            experiences += [experience]

        return experiences, replayLength
