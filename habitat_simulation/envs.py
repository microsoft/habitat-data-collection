"""Implement Habitat environments.

Returns:
    _type_: _description_
"""
from habitat import Env


class SimpleEnv(Env):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0