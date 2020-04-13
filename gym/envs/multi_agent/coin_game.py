"""
Coin Game environment.
Code modified from: https://github.com/alshedivat/lola/tree/master/lola
"""
# import queue
import copy
import time
from collections import deque

import numpy as np

import gym
import gym.spaces
from gym.utils import seeding


class CoinGame(gym.Env):
    """
    Coin Game environment.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = [
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1, 0]),
    ]
    VIEWPORT_W = 400
    VIEWPORT_H = 400

    def __init__(self, max_steps=20, grid_size=3):

        assert self.NUM_AGENTS == 2

        self.max_steps = max_steps
        self.grid_size = grid_size

        # The 4 channels stand for 2 players and 2 coin positions
        """ 
        obersation: 4 x self.grid_size x self.grid_size
        in dim 0: (current_agent_pos, other agent_pos, red_coin_pos, blue_coin_pos)
        """
        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Box(
                low=0,
                high=1,
                shape=(4, self.grid_size, self.grid_size),
                dtype='uint8'
            ), gym.spaces.Box(
                low=0,
                high=1,
                shape=(4, self.grid_size, self.grid_size),
                dtype='uint8'
            )
        ])
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(self.NUM_ACTIONS),
                                              gym.spaces.Discrete(self.NUM_ACTIONS)])

        self.step_count = None

        self.np_random = None
        # self.seed()
        self.viewer = None
        player_scale = 0.3
        self.PLAYER_SHAPE = [[[el[0] * player_scale,
                               el[1] * player_scale] for el in poly] for poly in self.PLAYER_SHAPE]
        coin_scale = 0.1
        self.COIN_SHAPE = [[[el[0] * coin_scale,
                             el[1] * coin_scale] for el in poly] for poly in self.COIN_SHAPE]
        self.cell_size = self.VIEWPORT_W // self.grid_size

        x_max = int(np.array([[el[0] for el in poly] for poly in self.PLAYER_SHAPE]).max())
        y_max = int(np.array([[el[1] for el in poly] for poly in self.PLAYER_SHAPE]).max())
        self.PLAYER_SHAPE = [[[x_max - el[0],
                               y_max - el[1]] for el in poly] for poly in self.PLAYER_SHAPE]

        self.coin_picking_speed = -1
        self.queue_size = max_steps #200
        self.n_steps_in_queue = 1e-6
        self.n_steps_in_queue_picking_own = 1e-6
        self.queue_picking_speed = deque(maxlen=self.queue_size)
        self.queue_picking_own = deque(maxlen=self.queue_size)
        self.good_actions = [deque(maxlen=self.queue_size) for i in range(self.NUM_AGENTS)]
        self.coin_picking_speed = np.nan
        self.coin_picking_own_fract = np.nan
        self.good_act_ag0 = np.nan
        self.good_act_ag1 = np.nan

        self.render_every = 100
        self.epi = 0

    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        self.np_random, seed = seeding.np_random(seed)
        # self.np_random = np.random.RandomState()
        # seed = None
        
        print("Coin Game SEED", seed)
        time.sleep(3.0)
        return [seed]

    def reset(self):
        self.epi += 1
        self.step_count = 0

        # Randomly start with a red or blue coin
        self.red_coin = self.np_random.randint(2, size=1)

        # agents positions
        self.red_pos = self.np_random.randint(self.grid_size, size=(2,))
        self.blue_pos = self.np_random.randint(self.grid_size, size=(2,))
        # Make sure agents don't overlap
        while self._same_pos(self.red_pos, self.blue_pos):
            self.blue_pos = self.np_random.randint(self.grid_size, size=2)

        # Coins
        self.coin_pos = np.zeros((2,), dtype=np.int8)
        self._generate_coin()

        self.observation = self._generate_observation()
        return self.observation

    def _generate_coin(self):

        # Switch between red and blue coins
        self.red_coin = 1 - self.red_coin

        # Make sure coin has a different position than the agents
        success = 0
        while success < 2:
            self.coin_pos = self.np_random.randint(self.grid_size, size=2)
            success = 1 - self._same_pos(self.red_pos,self.coin_pos)
            success += 1 - self._same_pos(self.blue_pos,self.coin_pos)

    def _same_pos(self, x, y):
        return (x == y).all()

    def _generate_observation(self):
        observation = np.zeros(list(self.observation_space[0].shape))
        observation[0, self.red_pos[0], self.red_pos[1]] = 1
        observation[1, self.blue_pos[0], self.blue_pos[1]] = 1
        if self.red_coin:
            observation[2, self.coin_pos[0], self.coin_pos[1]] = 1
        else:
            observation[3, self.coin_pos[0], self.coin_pos[1]] = 1
        observation = observation.astype(np.uint8)

        second_player_observation = observation[[1, 0, 3, 2], ...]

        observation = (observation, np.copy(second_player_observation))
        # observation = tuple([observation for i in range(self.NUM_AGENTS)])
        # print(observation)
        return observation

    # import line_profiler
    # @profile
    def step(self, actions):
        ac0, ac1 = actions
        ac0, ac1 = int(ac0), int(ac1)
        # print("ac0, ac1",ac0, ac1)
        assert ac0 in {0, 1, 2, 3} and ac1 in {0, 1, 2, 3}


        self.good_actions[0].append(self.is_a_good_action(self.observation[0], ac0))
        self.good_actions[1].append(self.is_a_good_action(self.observation[1], ac1))


        # Move players
        self.red_pos = \
            (self.red_pos + self.MOVES[ac0]) % self.grid_size
        self.blue_pos = \
            (self.blue_pos + self.MOVES[ac1]) % self.grid_size

        # Compute rewards
        generate = False
        reward_red = 0
        reward_blue = 0
        if self.red_coin:
            if self._same_pos(self.red_pos, self.coin_pos):
                generate = True
                reward_red += 1
            if self._same_pos(self.blue_pos, self.coin_pos):
                generate = True
                reward_red += -2
                reward_blue += 1

        else:
            if self._same_pos(self.red_pos, self.coin_pos):
                generate = True
                reward_red += 1
                reward_blue += -2
            if self._same_pos(self.blue_pos, self.coin_pos):
                generate = True
                reward_blue += 1

        if generate:
            if (self.red_coin and reward_red > 0) or (not self.red_coin and reward_blue > 0):
                # Picked own
                self.queue_picking_own.append(True)
            elif (self.red_coin and reward_red == -1) or (not self.red_coin and reward_blue == -1):
                # Picked own but opponent picked too
                self.queue_picking_own.append(True)
                self.queue_picking_own.append(False)
                if self.n_steps_in_queue_picking_own < self.queue_size:
                    self.n_steps_in_queue_picking_own += 1
            else:
                self.queue_picking_own.append(False)
            self._generate_coin()
        if self.n_steps_in_queue < self.queue_size:
            self.n_steps_in_queue += 1
        if self.n_steps_in_queue_picking_own < self.queue_size:
            self.n_steps_in_queue_picking_own += 1

        self.queue_picking_speed.append(generate)

        reward = (reward_red, reward_blue)
        self.step_count += 1
        done = (self.step_count == self.max_steps)
        self.observation = self._generate_observation()

        # Log extra info
        if done:
            self.coin_picking_speed = round(float(sum(list(self.queue_picking_speed)) /self.n_steps_in_queue), 2)
            self.coin_picking_own_fract = round(float(sum(list(self.queue_picking_own)) / self.n_steps_in_queue_picking_own), 2)
            self.good_act_ag0 = round(float(sum(list(self.good_actions[0])) /self.n_steps_in_queue), 2)
            self.good_act_ag1 = round(float(sum(list(self.good_actions[1])) /self.n_steps_in_queue), 2)

        info = {"extra_info_to_log": {
            "pck_speed": self.coin_picking_speed,
            "pck_own": self.coin_picking_own_fract,
            "good_act_ag0": self.good_act_ag0,
            "good_act_ag1": self.good_act_ag1,
        }
        }

        return self.observation, reward, done, info

    PLAYER_SHAPE = [
        [[94.67, 271.00], [102.58, 159.62], [126.67, 160.68], [140.75, 269.74], [144.92, 380.85], [190.33, 381.47],
         [190.42, 270.47], [222.50, 270.47], [224.67, 381.47], [264.96, 381.47], [269.25, 268.47], [271.83, 163.47],
         [301.00, 163.47], [308.67, 267.47], [327.17, 266.63], [320.00, 130.26], [234.83, 129.90], [247.67, 98.53],
         [243.33, 58.53], [199.67, 50.28], [197.00, 16.03], [191.17, 17.15], [186.33, 50.28], [155.67, 55.53],
         [151.33, 98.53], [160.42, 126.15], [79.50, 128.76], [68.58, 268.38]]]
    COIN_SHAPE = [
        [[69.67, 193.00], [69.58, 226.62], [86.50, 268.24], [116.33, 302.47], [159.67, 326.47], [200.00, 331.47],
         [237.33, 328.47], [266.67, 315.47], [293.33, 294.10], [312.00, 264.74], [327.67, 232.37], [333.33, 200.00],
         [331.17, 161.63], [313.00, 129.26], [293.83, 102.90], [266.67, 84.53], [236.33, 71.53], [198.00, 68.53],
         [163.67, 72.53], [133.33, 84.53], [113.42, 103.15], [94.50, 127.76], [81.58, 158.38]]]
    RED_COLOR = (255, 0, 0)
    BLUE_COLOR = (0, 0, 255)

    def draw_shape(self, shape, pos, color):
        x_delta, y_delta = pos
        shape_pos = [[[el[0] + x_delta,
                       el[1] + y_delta] for el in poly] for poly in shape]
        for polygon in shape_pos:
            self.viewer.draw_polygon(polygon, color=color)

    def render(self, mode='human'):

        # Set windows
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
            self.viewer.set_bounds(0, self.VIEWPORT_W, 0, self.VIEWPORT_H)

        if self.epi % self.render_every == 0:

            for object_idx in range(self.observation[0].shape[0]):
                if object_idx == 0:  # Red player
                    color = self.RED_COLOR
                    shape = self.PLAYER_SHAPE
                elif object_idx == 1:  # Blue player
                    color = self.BLUE_COLOR
                    shape = self.PLAYER_SHAPE
                elif object_idx == 2:  # Red coin
                    color = self.RED_COLOR
                    shape = self.COIN_SHAPE
                elif object_idx == 3:  # blue coin
                    color = self.BLUE_COLOR
                    shape = self.COIN_SHAPE

                x, y = np.nonzero(self.observation[0][object_idx])
                if len(x) > 0 and len(y) > 0:
                    scale = self.cell_size
                    x, y = int(x) * scale, int(y) * scale
                    if object_idx % 2 == 1:
                        x += (self.cell_size // 2)
                    self.draw_shape(shape=shape, pos=(x, y), color=color)

            time.sleep(0.25)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def is_a_good_action(self, state, action):
        own_ini_pos_x, own_ini_pos_y = np.nonzero(state[0, ...])

        coin = state[2, ...] + state[3, ...]
        coin_pos_x, coin_pos_y = np.nonzero(coin)

        initial_dist = self.compute_dist(coin_pos_x, coin_pos_y, own_ini_pos_x, own_ini_pos_y)

        own_new_pos_x = (own_ini_pos_x + self.MOVES[action][0]) % self.grid_size
        own_new_pos_y = (own_ini_pos_y + self.MOVES[action][1]) % self.grid_size

        new_dist = self.compute_dist(coin_pos_x, coin_pos_y, own_new_pos_x, own_new_pos_y)
        # print(action)
        # print(state)
        # print("initial_dist, new_dist", initial_dist, new_dist)
        # print("is_a_good_action", new_dist < initial_dist)
        return new_dist < initial_dist

    def compute_dist(self, x1,y1,x2,y2):
        return abs(int(x1 - x2)) % 2 + abs(int(y1 - y2)) % 2