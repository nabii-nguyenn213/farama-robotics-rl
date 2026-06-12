from typing import Any, cast
import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper, RewardWrapper
from gymnasium.spaces import Box, Dict as DictSpace
from gymnasium.wrappers import TimeLimit

def flatten_goal_obs(obs_dict: dict[str, Any]) -> np.ndarray:
    obs = np.asarray(obs_dict["observation"], dtype=np.float32).reshape(-1)
    desired_goal = np.asarray(obs_dict["desired_goal"], dtype=np.float32).reshape(-1)
    return np.concatenate([obs, desired_goal], axis=-1).astype(np.float32)

class FlattenGoalObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, DictSpace):
            raise TypeError(
                f"FlattenGoalObservation expects Dict observation space, got {type(obs_space)}"
            )
        observation_space = cast(Box, obs_space.spaces["observation"])
        desired_goal_space = cast(Box, obs_space.spaces["desired_goal"])
        low = np.concatenate(
            [
                observation_space.low.reshape(-1),
                desired_goal_space.low.reshape(-1),
            ],
            axis=-1,
        ).astype(np.float32)
        high = np.concatenate(
            [
                observation_space.high.reshape(-1),
                desired_goal_space.high.reshape(-1),
            ],
            axis=-1,
        ).astype(np.float32)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        return flatten_goal_obs(observation)

class RewardScaler(RewardWrapper):
    def __init__(self, env, reward_scaler=1.0):
        super().__init__(env)
        self.reward_scaler = float(reward_scaler)

    def reward(self, reward):
        return float(reward) * self.reward_scaler


def make_env(
    env_name,
    max_episode_steps=0,
    reward_scaler=1.0,
    flatten_obs=True,
    **env_kwargs,
):
    env = gym.make(env_name, **env_kwargs)

    if flatten_obs and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenGoalObservation(env)

    if reward_scaler != 1.0:
        env = RewardScaler(env, reward_scaler)

    if max_episode_steps is not None and max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env
