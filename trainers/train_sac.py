import os 
import numpy as np 
import torch 
from typing import Any
from agents.SAC import SAC_Agent
from components.buffer import ReplayBuffer
from trainers.BaseTrainer import BaseTrainer
from utils.helper import getObsActDim
from utils.logger import Logger

class SACTrainer(BaseTrainer): 

    def __init__(self, config):
        super().__init__(config)
        self.agent = SAC_Agent(config)
        buffer_capacity = config["train"].get("memory_size", 1_000_000)
        env_name = config["env"].get("name", "FetchSlide-v4")
        env_kwargs = config["env"].get("kwargs", {})
        obs_dim, act_dim = getObsActDim(env_name,max_episode_steps=config["env"].get("max_episode_steps", 0),
                                        reward_scaler=config["env"].get("reward_scaler", 1.0),
                                        flatten_obs=config["env"].get("flatten_obs", True),**env_kwargs)
        self.replay_buffer = ReplayBuffer(max_size=buffer_capacity, obs_dim=obs_dim, act_dim=act_dim)


    def train(self): 
        self.dirs_resolve()
        self.make_dirs(run_name=self.run_name)
        self.logger = Logger(config=self.config, run_name=self.run_name, log_dir=self.log_dir, tb_dir=self.tb_dir)
        self.logger._init_files()
        self.logger.save_config()
        self.logger.info(f"Initialize SAC Trainer: run name={self.run_name}")
        
        obs, _ = self.reset_env(self.env)
        ep_return = 0.0 
        ep_len = 0 

        last_update_info = None 
        try: 
            for step in range(1, self.total_timesteps+1):
                self.global_step = step 
                if step < self.learning_start: 
                    action = self.env.action_space.sample()
                else: 
                    action = self.agent.act(obs, deterministic=False)
                next_obs, reward, done, terminated, truncated, info = self.step_env(action, self.env)
                self.replay_buffer.store_transition(obs, action, reward, next_obs, done)
                obs = next_obs 
                ep_return += float(reward)
                ep_len += 1 
                if done : 
                    self.episode_num += 1
                    self.logger.log_episode(self.episode_num, step, episodic_return=ep_return, episode_length=ep_len)
                    obs, _ = self.reset_env(self.env)
                    ep_return = 0.0
                    ep_len = 0 
                if step >= self.learning_start: 
                    for _ in range(self.gradient_step): 
                        batch = self.replay_buffer.sample_buffer(self.batch_size, device=self.agent.device)
                        last_update_info = self.agent.update(batch)
                    if step % self.log_every == 0 and last_update_info is not None: 
                        self.logger.log_train(step, last_update_info, print_to_console=True)
                if self.eval_every > 0 and step % self.eval_every == 0: 
                    avg_return = self.evaluate(self.eval_episode)
                    is_best = self.logger.log_eval(step, avg_return)
                    if is_best: 
                        self.save_best(step)
                if self.save_every > 0 and step % self.save_every == 0: 
                    self.save_checkpoint(step)
            self.logger.info("Finished SAC Training")
            self.save_checkpoint(self.global_step, filename="final.pt")
        finally: 
            self.logger.close()

    @torch.no_grad()
    def evaluate(self, num_episodes = None):
        if num_episodes is None : 
            num_episodes = self.eval_episode 
        returns = []
        for _ in range(num_episodes): 
            obs, _ = self.reset_env(self.eval_env)
            done = False 
            ep_return = 0.0 
            
            while not done: 
                action = self.agent.act(obs, deterministic=True)
                next_obs, reward, done, terminated, truncated, info = self.step_env(action, self.eval_env)
                ep_return += float(reward)
                obs = next_obs

            returns.append(ep_return)
        avg_return = sum(returns)/len(returns)
        return avg_return

    def save_checkpoint(self, step, filename=None):
        if filename is None:
            filename = f"sac_checkpoint_{step}.pt"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.ckpt_dir,
            filename,
        )
        checkpoint = {
            "step": step,
            "episode_num": self.episode_num,
            "config": self.config,
            "agent_state": self.get_agent_state(),
        }
        torch.save(checkpoint, ckpt_path)
        self.logger.log_checkpoint(
            path=ckpt_path,
            step=step,
            kind="checkpoint",
        )

    def save_best(self, step):
        os.makedirs(self.best_dir, exist_ok=True)
        best_path = os.path.join(
            self.best_dir,
            "sac_best.pt",
        )
        checkpoint = {
            "step": step,
            "episode_num": self.episode_num,
            "config": self.config,
            "agent_state": self.get_agent_state(),
        }
        torch.save(checkpoint, best_path)
        self.logger.log_checkpoint(
            path=best_path,
            step=step,
            kind="best model",
        )

    def save_model_only(self, step=None):
        if step is None:
            filename = "sac_model.pt"
        else:
            filename = f"sac_model_{step}.pt"
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(
            self.model_dir,
            filename,
        )
        torch.save(self.agent.net.state_dict(), model_path)
        self.logger.log_checkpoint(
            path=model_path,
            step=step,
            kind="model",
        )

    def get_agent_state(self):
        state: dict[str, Any] = {
            "net": self.agent.net.state_dict(),
            "target_critic1": self.agent.target_critic1.state_dict(),
            "target_critic2": self.agent.target_critic2.state_dict(),
            "actor_optimizer": self.agent.actor_optimizer.state_dict(),
        }
        if hasattr(self.agent, "critic1_optimizer"):
            state["critic1_optimizer"] = self.agent.critic1_optimizer.state_dict()
        if hasattr(self.agent, "critic2_optimizer"):
            state["critic2_optimizer"] = self.agent.critic2_optimizer.state_dict()
        alpha_optimizer = getattr(self.agent, "alpha_optimizer", None)
        if alpha_optimizer is not None:
            state["alpha_optimizer"] = alpha_optimizer.state_dict()
        log_alpha = getattr(self.agent, "log_alpha", None)
        if log_alpha is not None:
            if torch.is_tensor(log_alpha):
                state["log_alpha"] = log_alpha.detach().cpu()
            else:
                state["log_alpha"] = log_alpha
        alpha = getattr(self.agent, "alpha", None)
        if alpha is not None:
            if torch.is_tensor(alpha):
                state["alpha"] = alpha.detach().cpu()
            else:
                state["alpha"] = alpha
        return state
