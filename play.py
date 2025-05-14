import time
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
import numpy as np
from utils import make_env, state_preprocessor

# Load the trained model
model = PPO.load("ppo_mario")

# Initialize the environment
env = make_env(render_mode="human", action_space=SIMPLE_MOVEMENT, version="SuperMarioBros-v1")
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)


# Reset the environment
obs= env.reset()
done = False

while not done:
    # Preprocess the observation if necessary
    # obs = preprocess_observation(obs)

    # Predict the action
    action, _ = model.predict(obs, deterministic=True)

    # Take a step in the environment
    obs, reward, done,  info = env.step(action)
    # Render the environment
    env.render()
    time.sleep(0.03)  # Control the frame rate

# Close the environment
env.close()