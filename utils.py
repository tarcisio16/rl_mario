import gym_super_mario_bros.actions
import gymnasium as gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import cv2
from gym import spaces
import numpy as np

class CustomJoypadSpace(JoypadSpace):
    def __init__(self, env, action_space):
        super().__init__(env, action_space)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 1),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = state_preprocessor(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = state_preprocessor(obs)
        return obs, reward, terminated, truncated, info

def make_env(render_mode=None, action_space=SIMPLE_MOVEMENT, version="SuperMarioBros-v0"):
    env = gym_super_mario_bros.make(
        version,
        apply_api_compatibility=True,
        render_mode=render_mode
    )
    env = CustomJoypadSpace(env, SIMPLE_MOVEMENT)
    return env

def state_preprocessor(frame):
    # state is RGB images of shape (240, 256, 3), convert to grayscale and resize to (84, 84) and normalize
    frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA) / 255.0
    # add a channel dimension
    frame = np.expand_dims(frame, axis=-1)
    return frame