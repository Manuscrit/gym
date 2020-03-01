"""
Coin Game environment.
Code modified from: https://github.com/alshedivat/lola/tree/master/lola
"""
import gym
import numpy as np

import gym.spaces
from gym.utils import seeding


class CoinGameVec(gym.Env):
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = [
        np.array([0,  1]),
        np.array([0, -1]),
        np.array([1,  0]),
        np.array([-1, 0]),
    ]

    def __init__(self, max_steps = 50, grid_size=3):
        # TODO finish to remove batch size (not computing by batch)
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = 1
        # The 4 channels stand for 2 players and 2 coin positions
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(4, self.grid_size, self.grid_size),
            dtype='uint8'
        )
        self.action_space = gym.spaces.Tuple([
            gym.spaces.Tuple([ gym.spaces.Discrete(self.NUM_ACTIONS),
                              gym.spaces.Discrete(self.NUM_ACTIONS)])
            for _ in range(self.batch_size)
        ])


        self.step_count = None

        self.np_random = None
        self.seed()


    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self.step_count = 0

        self.red_coin = self.np_random.randint(2, size=self.batch_size)
        # Agent and coin positions
        self.red_pos = self.np_random.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.blue_pos = self.np_random.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)
        for i in range(self.batch_size):
            # Make sure coins don't overlap
            while self._same_pos(self.red_pos[i], self.blue_pos[i]):
                self.blue_pos[i] = self.np_random.randint(self.grid_size, size=2)
            self._generate_coin(i)
        return self._generate_observation()

    def _generate_coin(self, i):

        self.red_coin[i] = 1 - self.red_coin[i]
        # Make sure coin has a different position than the agents
        success = 0
        while success < 2:
            self.coin_pos[i] = self.np_random.randint(self.grid_size, size=(2))
            success  = 1 - self._same_pos(self.red_pos[i],
                                          self.coin_pos[i])
            success += 1 - self._same_pos(self.blue_pos[i],
                                          self.coin_pos[i])

    def _same_pos(self, x, y):
        return (x == y).all()

    def _generate_observation(self):
        observation = np.zeros([self.batch_size] + list(self.observation_space.shape))
        for i in range(self.batch_size):
            observation[i, 0, self.red_pos[i][0], self.red_pos[i][1]] = 1
            observation[i, 1, self.blue_pos[i][0], self.blue_pos[i][1]] = 1
            if self.red_coin[i]:
                observation[i, 2, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
            else:
                observation[i, 3, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
        observation = np.squeeze(observation, axis=0).astype(np.uint8)
        return observation

    def step(self, actions):
        for j in range(self.batch_size):
            ac0, ac1 = actions[j]
            assert ac0 in {0, 1, 2, 3} and ac1 in {0, 1, 2, 3}

            # Move players
            self.red_pos[j] = \
                (self.red_pos[j] + self.MOVES[ac0]) % self.grid_size
            self.blue_pos[j] = \
                (self.blue_pos[j] + self.MOVES[ac1]) % self.grid_size

        # Compute rewards
        reward_red, reward_blue = [], []
        for i in range(self.batch_size):
            generate = False
            if self.red_coin[i]:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red.append(1)
                    reward_blue.append(0)
                elif self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red.append(-2)
                    reward_blue.append(1)
                else:
                    reward_red.append(0)
                    reward_blue.append(0)

            else:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red.append(1)
                    reward_blue.append(-2)
                elif self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red.append(0)
                    reward_blue.append(1)
                else:
                    reward_red.append(0)
                    reward_blue.append(0)

            if generate:
                self._generate_coin(i)

        # reward = (np.array(reward_red), np.array(reward_blue))
        reward = (reward_red[0], reward_blue[0])
        self.step_count += 1
        # done = np.array([
        #     (self.step_count == self.max_steps) for _ in range(self.batch_size)
        # ])
        done = (self.step_count == self.max_steps)
        observation = self._generate_observation()

        return observation, reward, done, {}
