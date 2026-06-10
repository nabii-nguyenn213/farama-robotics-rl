import numpy as np 
import gymnasium as gym 
from gymnasium.wrappers import TimeLimit
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

def make_env(env_name, max_episode_steps=0, reward_scaler=1.0, **env_kwargs): 
    env = gym.make(env_name, **env_kwargs) 
    if max_episode_steps and max_episode_steps > 0: 
        env = TimeLimit(env, max_episode_steps) 
    return env

def flatten_obs(obs_dict): 
    obs = np.asarray(obs_dict["observation"], dtype=np.float32)
    desired_goal = np.asarray(obs_dict["desired_goal"], dtype=np.float32)
    return np.concatenate([obs, desired_goal], axis=-1).astype(np.float32)

# def get_env_info(env_name, max_episode_steps=0, **env_kwargs): 
#     env = make_env(env_name, max_episode_steps, **env_kwargs)
#     try : 
#         obs_space = env.observation_space
#         act_space = env.action_space
#         obs_dim = int(np.prod(obs_space["observation"].shape) + np.prod(obs_space["desired_goal"].shape))
#         act_dim = int(np.prod(act_space.shape))
#         return {"obs_dim" : obs_dim, 
#                 "act_dim" : act_dim, 
#                 "action_low" : np.asarray(act_space.low, dtype=np.float32).copy(), 
#                 "action_high" : np.asarray(act_space.high, dtype=np.float32).copy()
#                }
#     finally: 
#         env.close()
