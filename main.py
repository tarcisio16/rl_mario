import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from models.player import Player
from environment import wrap_env
import os
import torch
import numpy as np
from tqdm import tqdm
import time
import argparse

ENVIRONMENT   = "SuperMarioBros2-v1"
GAMMA         = 0.99
MOVEMENTS   = SIMPLE_MOVEMENT
NUM_EPISODES  = 300
BATCH_SIZE    = 8
EPSILON       = 1.0 
EPSILON_DECAY = 0.99999999975
EPSILON_MIN   = 0.1
LEARNING_RATE = 0.00025
BUFFER_SIZE   = 10_000
SYNC_RATE     = 10000
LOSS_FN       = torch.nn.SmoothL1Loss()
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

def main():

    parser = argparse.ArgumentParser(description="Mario DQN")
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        help="Set to True to train the model, False to play with a pre-trained model",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default="models/mario_model.pth",
        help="Path to the pre-trained model",
    )
    args = parser.parse_args()
    if args.train:
        base = gym_super_mario_bros.make(ENVIRONMENT, apply_api_compatibility=True)
    else:
        base = gym_super_mario_bros.make(ENVIRONMENT, apply_api_compatibility=True, render_mode="human")
    base = JoypadSpace(base, MOVEMENTS)
    env = wrap_env(base, idle_steps=500)
    env.reset()
    state_dims = env.observation_space.shape

    player = Player(
        state_dims    = state_dims,
        action_dims   = env.action_space.n,
        model         = "SCNN",
        learning_rate = LEARNING_RATE,
        gamma         = GAMMA,
        epsilon       = EPSILON,
        epsilon_decay = EPSILON_DECAY,
        epsilon_min   = EPSILON_MIN,
        buffer_size   = BUFFER_SIZE,
        batch_size    = BATCH_SIZE,
        sync_rate     = SYNC_RATE,
        loss_fn       = LOSS_FN,
        device        = DEVICE
    )

    if not args.train:
        player.load_model(args.load_model)
        print(f"✅ Model loaded from: {args.load_model}")

    next_obs, reward, term, trunc, _ = env.step(env.action_space.sample())
    done = term or trunc

    if args.train:
        for ep in tqdm(range(NUM_EPISODES), desc="Training", unit="episode"):
            done = False
            obs, _ = env.reset()
            tot_reward = 0
            while not done:
                action = player.select_action(obs)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                tot_reward += reward    
                player.store_experience(obs, action, reward, next_obs, done)
                player.learn(steps = 20)
                obs = next_obs

                if done and not args.train:
                    break
    else:
        obs , _ = env.reset()
        done = False
        while not done:
            env.render()
            time.sleep(0.05)
            action = player.select_action(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            obs = next_obs
            
            

    env.close()
    os.makedirs("models", exist_ok=True)
    player.save_model(os.path.join("models", "mario_model.pth"))
    print("Model saved to models/mario_model.pth")

if __name__ == "__main__":
    main()

    