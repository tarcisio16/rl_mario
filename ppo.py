import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import torch
import wandb
from utils import make_env, state_preprocessor
from stable_baselines3 import PPO
import cv2



if __name__ == "__main__":
    # Create and wrap the environment
    env = make_env(render_mode="rgb_array", action_space=SIMPLE_MOVEMENT, version="SuperMarioBros-v1")
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    # Initialize the PPO model
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=64,
        tensorboard_log="./ppo_mario_tb/",
        seed=42,
        device="cuda" if torch.cuda.is_available() else "mps",
    )
    # Train the agent
    model.learn(total_timesteps=1000000, tb_log_name="ppo_mario", reset_num_timesteps=False)
    model.save("ppo_mario")


            