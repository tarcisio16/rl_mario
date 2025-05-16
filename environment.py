import numpy as np
import gym
from gym import Wrapper, ObservationWrapper
from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack

def wrap_env(env, skip=4, render_mode=None, idle_steps=1000):
    env = EnvironmentWrapper(env, skip, idle_steps)
    
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=False)

    return env

class EnvironmentWrapper(Wrapper):
    def __init__(self, env, skip, idle_steps=1000):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete), "Only discrete action spaces are supported"
        assert idle_steps > 0, "Idle steps must be greater than 0"
        self.skip = skip
        self.idle_steps = idle_steps
        self.last_position = None
        self.idle_counter = 0

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            current_position = info.get('position', 0)
            total_reward += reward
            if self.last_position is not None and current_position == self.last_position:
                self.idle_counter += 1
            else:
                self.idle_counter = 0
            if terminated or truncated:
                break
            if self.idle_counter >= self.idle_steps:
                self.idle_counter = 0
                truncated = True
                break
            self.last_position = current_position

        return obs, total_reward, terminated, truncated, info
