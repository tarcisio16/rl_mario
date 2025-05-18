import numpy as np
import gym
from gym.spaces import Box
from gym import Wrapper, ObservationWrapper
from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT



def wrap_env(env, skip=4, idle_steps=1000, clip_rewards=True, render_mode=None):
    env = MaxAndSkipEnv(env, skip=skip)               # 1. Skip and max-pool frames
    env = GrayScaleObservation(env)                   # 2. Convert to grayscale
    env = ResizeObservation(env, shape=84)            # 3. Resize to 84x84
    env = RepeatActionWrapper(env, repeat=4)       # 4. Repeat actions
    env = FrameStack(env, num_stack=4, lz4_compress=True)  # 4. Stack 4 frames
    if clip_rewards:
        env = ClipRewardEnv(env)                      # 5. Clip rewards
    env = EpisodicLifeMario(env)                      # 6. Make lives episodic
    #env = EnvironmentWrapper(env, idle_steps=idle_steps)  # 7. Idle check
    return env
        

class RepeatActionWrapper(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        # if action is not not noop
        if action == 0:
            obs , reward, terminated, truncated, info = self.env.step(action)
            return obs, reward, terminated, truncated, info
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class EnvironmentWrapper(Wrapper):
    def __init__(self, env, idle_steps=1000, delta = 2):
        super().__init__(env)
        self.idle_steps   = idle_steps
        self.last_x     = 0
        self.idle_counter = 0
        self.delta        = delta

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        assert info.get('x_pos', 0) is not None, "Environment must provide 'x_pos' in info"
        self.last_x = info.get('x_pos', 0)
        self.idle_counter  = 0
        return obs, info

    def step(self, action):
        for _ in range(self.idle_steps):
            obs, reward, terminated, truncated, info = self.env.step(action)
            current_x = info.get('x_pos', 0)
            if abs(current_x - self.last_x) < self.delta:
                self.idle_counter += 1
            else:
                self.idle_counter = 0
            self.last_x = current_x
            if self.idle_counter >= self.idle_steps:
                truncated = True

        return obs, reward, terminated, truncated, info


class EpisodicLifeMario(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def _get_lives(self):
        """Get the number of lives left in the game."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        if hasattr(env, '_life'):
            return env._life
        else:
            raise ValueError("Environment does not have _life attribute")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        env = self.env
        lives = self._get_lives()

        if lives < self.lives and lives > 0:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        info = {}
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ ,info= self.env.step(0)
        self.lives = self._get_lives()
        return obs, info 

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        terminated = False
        truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        # normalize reward between -1 and 1
        return np.sign(reward) * (np.sqrt(np.abs(reward)) - 1)

