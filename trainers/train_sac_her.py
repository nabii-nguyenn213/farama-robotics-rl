import os
import numpy as np
import torch
from typing import Any

from agents.SAC import SAC_Agent
from components.buffer import HER_ReplayBuffer
from trainers.BaseTrainer import BaseTrainer
from utils.logger import Logger
from utils.helper import flatten_goal_obs

class SACHERTrainer(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        self.agent = SAC_Agent(config)

        pretrained_checkpoint = self.config["train"].get("pretrained_checkpoint", None)

        if pretrained_checkpoint is not None:
            self.load_pretrained(pretrained_checkpoint)

        buffer_capacity = config["train"].get("memory_size", 1_000_000)
        her_ratio = config["train"].get("her_ratio", 0.8)

        self.replay_buffer = HER_ReplayBuffer(
            max_size = buffer_capacity,
            env = self.env,
            her_ratio = her_ratio,
        )

    def train(self):
        self.dirs_resolve()
        self.make_dirs(run_name=self.run_name)

        self.logger = Logger(
            config=self.config,
            run_name=self.run_name,
            log_dir=self.log_dir,
            tb_dir=self.tb_dir,
        )

        self.logger._init_files()
        self.logger.save_config()
        self.logger.info(f"Initialize SAC+HER Trainer: run name={self.run_name}")

        obs, _ = self.reset_env(self.env)
        ep_return = 0.0
        ep_len = 0
        last_update_info = None
        ep_success = 0.0

        initial_achieved_goal = obs["achieved_goal"].copy()

        ep_final_distance = float(np.linalg.norm(
            obs["achieved_goal"] - obs["desired_goal"]
        ))
        ep_object_displacement = 0.0
        try:
            for step in range(1, self.total_timesteps + 1):
                self.global_step = step

                if step < self.learning_start:
                    action = self.env.action_space.sample()
                
                else:
                    flat_obs = flatten_goal_obs(obs)
                    action = self.agent.act(flat_obs, deterministic=False)
                
                next_obs, reward, done, terminated, truncated, info = self.step_env(action, self.env)

                ep_success = max(
                    ep_success,
                    float(info.get("is_success", 0.0))
                )

                ep_final_distance = float(np.linalg.norm(
                    next_obs["achieved_goal"] - next_obs["desired_goal"]
                ))

                current_displacement = float(np.linalg.norm(
                    next_obs["achieved_goal"] - initial_achieved_goal
                ))

                ep_object_displacement = max(
                    ep_object_displacement,
                    current_displacement
                )

                self.replay_buffer.store_transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=terminated,
                    episode_done=terminated or truncated,
                    info=info,
                )

                obs = next_obs
                ep_return += float(reward)
                ep_len += 1

                if done:
                    self.episode_num += 1

                    self.logger.log_episode(
                        self.episode_num,
                        step,
                        episodic_return=ep_return,
                        episode_length=ep_len,
                    )

                    if self.logger.writer is not None:
                        self.logger.writer.add_scalar(
                            "episode/final_distance",
                            ep_final_distance,
                            step,
                        )

                        self.logger.writer.add_scalar(
                            "episode/object_displacement",
                            ep_object_displacement,
                            step,
                        )

                        self.logger.writer.add_scalar(
                            "episode/success",
                            ep_success,
                            step,
                        )

                    obs, _ = self.reset_env(self.env)
                    ep_return = 0.0
                    ep_len = 0
                    ep_success = 0.0

                    initial_achieved_goal = obs["achieved_goal"].copy()

                    ep_final_distance = float(np.linalg.norm(
                        obs["achieved_goal"] - obs["desired_goal"]
                    ))

                    ep_object_displacement = 0.0

                if step >= self.learning_start and self.replay_buffer.can_sample(self.batch_size):
                    for _ in range(self.gradient_step):
                        batch = self.replay_buffer.sample_buffer(
                            self.batch_size,
                            device=self.agent.device,
                        )
                        last_update_info = self.agent.update(batch)

                    if step % self.log_every == 0 and last_update_info is not None:
                        self.logger.log_train(step, last_update_info, print_to_console=True)
                
                if self.eval_every > 0 and step % self.eval_every == 0:
                    avg_return, success_rate = self.evaluate(self.eval_episode)

                    is_best = self.logger.log_eval(
                        step,
                        avg_return,
                        success_rate=success_rate,
                    )

                    if is_best:
                        self.save_best(step)
                
                if self.save_every > 0 and step % self.save_every == 0:
                    self.save_checkpoint(step)
                
            self.logger.info("Finished SAC+HER Training")
            self.save_checkpoint(self.global_step, filename="sacher_final.pt")
        finally:
            self.logger.close()
    @torch.no_grad()
    def evaluate(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.eval_episode

        returns = []
        successes = []

        for _ in range(num_episodes):
            obs, _ = self.reset_env(self.eval_env)
            done = False

            ep_return = 0.0
            ep_success = 0.0

            while not done:
                flat_obs = flatten_goal_obs(obs)
                action = self.agent.act(flat_obs, deterministic=True)

                next_obs, reward, done, terminated, truncated, info = self.step_env(
                    action,
                    self.eval_env,
                )

                ep_return += float(reward)

                ep_success = max(
                    ep_success,
                    float(info.get("is_success", 0.0)),
                )

                obs = next_obs

            returns.append(ep_return)
            successes.append(ep_success)

        avg_return = sum(returns) / len(returns)
        success_rate = sum(successes) / len(successes)

        return avg_return, success_rate
    
    def save_checkpoint(self, step, filename=None):
        if filename is None:
            filename = f"sacher_checkpoint_{step}.pt"

        os.makedirs(self.ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(self.ckpt_dir, filename)

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

        best_path = os.path.join(self.best_dir, "sacher_best.pt")

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
    
    def get_agent_state(self):
        state: dict[str, Any] = {
            "net": self.agent.net.state_dict(),
            "target_critic1": self.agent.target_critic1.state_dict(),
            "target_critic2": self.agent.target_critic2.state_dict(),
            "actor_optimizer": self.agent.actor_optimizer.state_dict(),
            "critic1_optimizer": self.agent.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.agent.critic2_optimizer.state_dict(),
        }

        alpha_optimizer = getattr(self.agent, "alpha_optimizer", None)
        if alpha_optimizer is not None:
            state["alpha_optimizer"] = alpha_optimizer.state_dict()
        
        log_alpha = getattr(self.agent, "log_alpha", None)
        if log_alpha is not None:
            state["log_alpha"] = log_alpha.detach().cpu()
        
        alpha = getattr(self.agent, "alpha", None)
        if alpha is not None:
            state["alpha"] = alpha.detach().cpu()
        
        return state
    
    def load_pretrained(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.agent.device,
            weights_only=False,
        )

        agent_state = checkpoint["agent_state"]

        self.agent.net.load_state_dict(agent_state["net"])

        self.agent.target_critic1.load_state_dict(
            agent_state["target_critic1"]
        )

        self.agent.target_critic2.load_state_dict(
            agent_state["target_critic2"]
        )

        if "log_alpha" in agent_state and hasattr(self.agent, "log_alpha"):
            self.agent.log_alpha.data.copy_(
                agent_state["log_alpha"].to(self.agent.device)
            )
            self.agent.alpha = self.agent.log_alpha.exp().detach()

        elif "alpha" in agent_state:
            alpha_value = agent_state["alpha"]
            if torch.is_tensor(alpha_value):
                alpha_value = alpha_value.item()

            self.agent.alpha = torch.tensor(
                float(alpha_value),
                device=self.agent.device,
            )

        print(f"Loaded pretrained checkpoint from: {checkpoint_path}")
        print(f"Pretrained step: {checkpoint.get('step', 'unknown')}")
        print(f"Pretrained episode: {checkpoint.get('episode_num', 'unknown')}")



