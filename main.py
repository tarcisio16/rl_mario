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
from torch.utils.tensorboard import SummaryWriter
import signal


ENVIRONMENT   = "SuperMarioBros-v3"
GAMMA         = 0.99
MOVEMENTS   =  RIGHT_ONLY 
NUM_EPISODES  = 1000
EPSILON       = 1.0 
EPSILON_DECAY = 0.99995
EPSILON_MIN   = 0.01
LEARNING_RATE = 0.00005
BUFFER_SIZE   = 50_000
SYNC_RATE     = 10_000
IDLE_STEPS    = 2_000
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16, 
        help="Batch size for training",
    )
    args = parser.parse_args()
    if args.train:
        base = gym_super_mario_bros.make(ENVIRONMENT, apply_api_compatibility=True)
        base = JoypadSpace(base, MOVEMENTS)
        env = wrap_env(base, idle_steps=IDLE_STEPS if args.train else 0)
        
    else:
        base = gym_super_mario_bros.make(ENVIRONMENT, apply_api_compatibility=True, render_mode="human")
        base = JoypadSpace(base, MOVEMENTS)
        env = wrap_env(base, idle_steps=0, render_mode="human")

    env.reset()
    state_dims = env.observation_space.shape
    writer = SummaryWriter(
        log_dir = f"runs/{ENVIRONMENT}_{MOVEMENTS}_{args.batch_size}"
    )

    player = Player(
        state_dims    = state_dims,
        action_dims   = env.action_space.n,
        model         = "CNN",
        learning_rate = LEARNING_RATE,
        gamma         = GAMMA,
        epsilon       = EPSILON,
        epsilon_decay = EPSILON_DECAY,
        epsilon_min   = EPSILON_MIN,
        buffer_size   = BUFFER_SIZE,
        batch_size    = args.batch_size,
        sync_rate     = SYNC_RATE,
        loss_fn       = LOSS_FN,
        device        = DEVICE,
        writer      = writer,
    )

    if not args.train:
        player.load_model(args.load_model)
        print(f"✅ Model loaded from: {args.load_model}")

    print(f"Training with device: {DEVICE}")    
    if args.train:
        try:
            for ep in tqdm(range(NUM_EPISODES), desc="Training", unit="episode"):
                done = False
                obs, info = env.reset()
                tot_reward = 0
                prev_x = info.get("x_pos", 0)
                while not done:
                    action = player.select_action(obs)
                    next_obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc
                    tot_reward +=  0.01 * (info.get("x_pos", 0) - prev_x) + reward
                    prev_x = info.get("x_pos", 0)
                    
                    player.store_experience(obs, action, reward, next_obs, done)
                    player.learn(steps=20)
                    obs = next_obs

                # update epsilon

                writer.add_scalar("Episode/Reward", tot_reward, ep) 
                writer.add_scalar("Mario position", info.get("x_pos", 0), ep)
                writer.add_scalar("Epsilon", player.epsilon, ep)
                # write memory usage and disk usage
        except Exception as e:
            print("Training interrupted:", e)
        finally:
            os.makedirs("models", exist_ok=True)
            player.save_model(os.path.join("models", "mario_model.pth"))
            print("Model saved to models/mario_model.pth")
        os.makedirs("models", exist_ok=True)
        player.save_model(os.path.join("models", "mario_model.pth"))
        print("Model saved to models/mario_model.pth")
    else:
        for ep in tqdm(range(5), desc="Playing", unit="episode"):
            obs , info = env.reset()
            done = False
            while not done:
                env.render()
                action = player.select_action(obs)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                obs = next_obs
                time.sleep(0.1)
            
            

    env.close()


if __name__ == "__main__":
    main()

    