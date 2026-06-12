import os
import time 
from datetime import datetime
import random 
from abc import ABC, abstractmethod
import numpy as np 
import torch
from torch.utils import deterministic 
from agents.SAC import SAC_Agent
from envs.env import make_env

class BaseTrainer(ABC): 

    def __init__(self, config):
        super().__init__()
        self.config = config 

        env_name = config["env"].get("name", "FetchSlide-v4")
        env_kwargs = config["env"].get("kwargs", {})
        max_episode_steps = config["env"].get("max_episode_steps", 0)

        self.env = make_env(env_name, max_episode_steps, **env_kwargs)
        self.eval_env = make_env(env_name, max_episode_steps, **env_kwargs)
        self.agent =  None 
        
        if hasattr(self.agent, "device"): 
            assert self.agent is not None, "agent should not be None"
            self.device = self.agent.device
        else: 
            self.device = config["train"].get("device", "auto")
            self.device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device

        self.seed = config["train"].get("seed", 42)
        self.total_timesteps = config["train"].get("total_timesteps", 1_000_000)
        
        self.eval_every = config["eval"].get("eval_every", 1000)
        self.save_every = config["eval"].get("save_every", 1000)
        self.log_every = config["eval"].get("log_every", 1000)
        self.eval_episode = config["eval"].get("eval_episode", 10)

        self.global_step = 0 
        self.episode_num = 0 
        self.best_eval_return = -float("inf")

        self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = config["dir"].get("log", f"./logs/log/{env_name}")
        self.ckpt_dir = config["dir"].get("ckpt", f"./results/checkpoints/{env_name}")
        self.model_dir = config["dir"].get("model", f"./results/model/{env_name}")
        self.best_dir = config["dir"].get("best", f"./results/best/{env_name}")
        self.tb_dir = config["dir"].get("tensorboard", f"./logs/tensorboard_logs/{env_name}")
        
        self.learning_start = config["train"].get("learning_start", 10_000)
        self.gradient_step = config["train"].get("gradient_step", 1)
        self.batch_size = config["train"].get("batch_size", 64)

        self.seed_everything(self.seed)

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available(): 
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        try : 
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)
        except Exception: 
            pass 
        if self.eval_env is not None: 
            try: 
                self.eval_env.action_space.seed(seed)
                self.eval_env.observation_space.seed(seed)
            except Exception: 
                pass

    def make_dirs(self, run_name=None): 
        if run_name is None: 
            run_name = self.run_name
        os.makedirs(os.path.join(self.log_dir, run_name), exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_dir, run_name), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, run_name), exist_ok=True)
        os.makedirs(os.path.join(self.best_dir, run_name), exist_ok=True)
        os.makedirs(os.path.join(self.tb_dir, self.run_name), exist_ok=True)

    def reset_env(self, env=None): 
        if env is None: 
            env = self.env 
        output = env.reset()
        if isinstance(output, tuple): 
            obs, info = output
        else : 
            obs = output 
            info = {}
        return obs, info 
    
    def step_env(self, action, env=None): 
        if env is None: 
            env = self.env
        output = env.step(action)
        if len(output) == 5: 
            next_obs, reward, terminated, truncated, info = output 
            done = terminated or truncated 
        elif len(output) == 4: 
            next_obs, reward, done, info  = output 
            terminated = done 
            truncated = False 
        else: 
            raise RuntimeError(f"Unsupported env.step() output length: {len(output)}")
        return next_obs, reward, done, terminated, truncated, info

    @abstractmethod
    def train(self): 
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, num_episodes:int | None = None): 
        raise NotImplementedError
