"""
Code modified from: https://github.com/alshedivat/lola/tree/master/lola
"""
# TODO change max_steps values

import numpy as np
import gym
from gym.spaces import Discrete, Tuple
from loguru import logger

class MatrixSocialDilemma(gym.Env):
    """
    A two-agent vectorized environment for matrix games.

    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5
    VIEWPORT_W = 400
    VIEWPORT_H = 400

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
        self.viewer = None
        self.observations = None

    def reset(self):
        self.step_count = 0
        self.observations = (self.NUM_STATES-1, self.NUM_STATES-1)

        # self.observations = self._one_hot_np_arrays(self.observations, n_values=self.NUM_STATES)
        # self.observations = self._np_arrays(self.observations)

        return self.observations

    def step(self, action):
        ac0, ac1 = action
        self.step_count += 1

        rewards = (self.payout_mat[ac0][ac1], self.payout_mat[ac1][ac0])
        self.observations = (ac0 * 2 + ac1, ac1 * 2 + ac0)
        done = (self.step_count == self.max_steps)

        # self.observations = self._one_hot_np_arrays(self.observations, n_values=self.NUM_STATES)
        # rewards = self._np_arrays(rewards)
        # self.observations = self._np_arrays(self.observations)

        return self.observations, rewards, done, {}

    # def _np_arrays(self, tup):
    #     return tuple([ np.array(el) for el in tup])
    #
    # def _one_hot_np_arrays(self, tup, n_values):
    #     return tuple([ self._to_one_hot(np.array(el), n_values) for el in tup])
    #
    # def _to_one_hot(self, array, n_values):
    #     return np.eye(n_values)[array.astype(np.int)]

    def render(self, mode='human'):

        # Set windows
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
            self.viewer.set_bounds(0, self.VIEWPORT_W, 0, self.VIEWPORT_H)

        state_size = [[0,0],[self.VIEWPORT_W//2, 0],
                      [self.VIEWPORT_W//2, self.VIEWPORT_H//2],[0, self.VIEWPORT_H//2]]

        # From one_hot_to_state
        # logger.debug("self.observations {}".format(self.observations))
        # state = np.nonzero(np.array(self.observations)[0])[0][0]
        state = self.observations[0]

        # logger.info("state {}".format(state))

        assert state < self.NUM_STATES and state >= 0, state
        if state == 0:
            # C & C
            delta_x = 0
            delta_y = self.VIEWPORT_H//2
        elif state == 1:
            # C & D
            delta_x = self.VIEWPORT_W//2
            delta_y = self.VIEWPORT_H//2
        elif state == 2:
            # D < C
            delta_x = 0
            delta_y = 0
        elif state == 3:
            # D & D
            delta_x = self.VIEWPORT_W//2
            delta_y = 0
        elif state == 4:
            delta_x = - self.VIEWPORT_W
            delta_y = - self.VIEWPORT_H

        current_agent_pos = state_size
        current_agent_pos = [ [x + delta_x, y + delta_y] for x,y in current_agent_pos]
        self.viewer.draw_polygon(current_agent_pos, color=(0,0,0))
        # import time
        # time.sleep(0.5)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

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


