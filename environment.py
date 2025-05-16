import numpy as np
import gym
from gym import Wrapper, ObservationWrapper
from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack


def wrap_env(env, skip=4, render_mode=None, idle_steps=1000):
    env = EpisodicLifeMario(env)
    env = EnvironmentWrapper(env, skip, idle_steps)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4, lz4_compress=False)
    return env

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
            obs, _, _, _ ,_= self.env.step(0)
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
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)



class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.np_random.integers(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs, info = None, {}
        for _ in range(noops):
            obs, _, terminated, truncated ,info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info 

    def step(self, ac):
        return self.env.step(ac)



class EnvironmentWrapper(Wrapper):
    def __init__(self, env, skip, idle_steps=1000, render_mode=None):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete), "Only discrete action spaces are supported"
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
            if self.idle_counter >= self.idle_steps and self.idle_steps > 0:
                self.idle_counter = 0
                truncated = True
                break
            self.last_position = current_position
            

        return obs, total_reward, terminated, truncated, info
