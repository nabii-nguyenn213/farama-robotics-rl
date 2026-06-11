import os 
import torch 
from omegaconf import OmegaConf
import gymnasium as gym 
import gymnasium_robotics
from gymnasium.spaces import Box, Discrete

def loadConfig(configDir=None): 
    if configDir is None:
        configDir="configs/SAC.yaml"
    config = OmegaConf.load(configDir)
    OmegaConf.resolve(config)
    return config

def getObsActDim(env_name, **kwargs):
    env = gym.make(env_name, **kwargs)
    obs_space = env.observation_space
    act_space = env.action_space

    if isinstance(obs_space, Box):
        obs_dim = obs_space.shape[0]
    elif isinstance(obs_space, Discrete):
        obs_dim = obs_space.n
    else:
        raise NotImplementedError(f"Unsupported observation space: {type(obs_space)}")

    if isinstance(act_space, Box):
        act_dim = act_space.shape[0]
    elif isinstance(act_space, Discrete):
        act_dim = act_space.n
    else:
        raise NotImplementedError(f"Unsupported action space: {type(act_space)}")
    env.close()
    return obs_dim, act_dim
