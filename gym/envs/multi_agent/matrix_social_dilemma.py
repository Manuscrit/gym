"""
Code modified from: https://github.com/alshedivat/lola/tree/master/lola
"""
# TODO change max_steps values

import numpy as np
import gym
from gym.spaces import Discrete, Tuple

class MatrixSocialDilemma(gym.Env):
    """
    A two-agent vectorized environment for matrix games.

    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5

    def __init__(self, payout_matrix, max_steps=50):
        """
        :arg payout_matrix: numpy 2x2 array. Along dim 0 (rows), action of
        current agent change. Along dim 1 (col), action of the
        other agent change. (0,0) = (C,C), (1,1) = (D,D)
        :arg max_steps: max steps per episode before done equal True
        """
        self.max_steps = max_steps
        self.payout_mat = payout_matrix
        self.action_space = Tuple([Discrete(self.NUM_ACTIONS),
                                   Discrete(self.NUM_ACTIONS)])
        self.observation_space = Tuple([Discrete(self.NUM_STATES),
                                        Discrete(self.NUM_STATES)])

        self.step_count = None

    def reset(self):
        self.step_count = 0
        observations = (self.NUM_STATES-1, self.NUM_STATES-1)
        return observations

    def step(self, action):
        ac0, ac1 = action
        self.step_count += 1

        rewards = (self.payout_mat[ac0][ac1], self.payout_mat[ac1][ac0])
        observations = (ac0 * 2 + ac1, ac0 * 2 + ac1)
        done = (self.step_count == self.max_steps)

        return observations, rewards, done, {}


class IteratedMatchingPennies(MatrixSocialDilemma):
    """
    A two-agent vectorized environment for the Matching Pennies game.
    """
    def __init__(self, max_steps=50):
        payout_mat = np.array([[1, -1],
                               [-1, 1]])
        super().__init__(payout_mat, max_steps)


class IteratedPrisonersDilemma(MatrixSocialDilemma):
    """
    A two-agent vectorized environment for the Prisoner's Dilemma game.
    """
    def __init__(self, max_steps=50):
        payout_mat = np.array([[-1., -3],
                               [0., -2.]])
        super().__init__(payout_mat, max_steps)
        self.NAME = "IPD"


class IteratedStagHunt(MatrixSocialDilemma):
    """
    A two-agent vectorized environment for the Stag Hunt game.
    """
    def __init__(self, max_steps=50):
        payout_mat = np.array([[3, 0],
                               [2, 1]])
        super().__init__(payout_mat, max_steps)


class IteratedChicken(MatrixSocialDilemma):
    """
    A two-agent vectorized environment for the Chicken game.
    """
    def __init__(self, max_steps=50):
        payout_mat = np.array([[0, -1],
                               [1, -10]])
        super().__init__(payout_mat, max_steps)


