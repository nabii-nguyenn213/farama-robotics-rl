import os 
import torch 
from omegaconf import OmegaConf
import gymnasium as gym 
import gymnasium_robotics
from gymnasium.spaces import Box, Discrete, Dict
from gymnasium.spaces.utils import flatdim
from envs.env import make_env

def loadConfig(configDir=None): 
    if configDir is None:
        configDir="configs/SAC.yaml"
    config = OmegaConf.load(configDir)
    OmegaConf.resolve(config)
    return config

def getObsActDim(env_name, max_episode_steps=0, reward_scaler=1.0, flatten_obs=True, **kwargs):
    env = make_env(
        env_name,
        max_episode_steps=max_episode_steps,
        reward_scaler=reward_scaler,
        flatten_obs=flatten_obs,
        **kwargs,
    )
    obs_space = env.observation_space
    act_space = env.action_space
    if isinstance(obs_space, Box):
        obs_dim = obs_space.shape[0]
    elif isinstance(obs_space, Discrete):
        obs_dim = obs_space.n
    else:
        env.close()
        raise NotImplementedError(f"Unsupported observation space: {type(obs_space)}")

    if isinstance(act_space, Box):
        act_dim = act_space.shape[0]
    elif isinstance(act_space, Discrete):
        act_dim = act_space.n
    else:
        env.close()
        raise NotImplementedError(f"Unsupported action space: {type(act_space)}")
    env.close()
    return obs_dim, act_dim
