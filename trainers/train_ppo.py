import os
from typing import Any

import numpy as np
import torch

from agents.PPO import PPO_Agent
from trainers.BaseTrainer import BaseTrainer
from utils.logger import Logger


class PPOTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.agent = PPO_Agent(config)
        self.rollout_steps = int(config["train"].get("rollout_steps", 2048))
        self.eval_episode = config["eval"].get("eval_episode", config["eval"].get("eval_episodes", 10))

    def compute_gae(self, rewards, values, dones, last_value):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])

            delta = rewards[t] + self.agent.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.agent.gamma * self.agent.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.asarray(values, dtype=np.float32)
        return advantages, returns

    def train(self):
        self.dirs_resolve()
        self.make_dirs(run_name=self.run_name)
        self.logger = Logger(config=self.config, run_name=self.run_name, log_dir=self.log_dir, tb_dir=self.tb_dir)
        self.logger._init_files()
        self.logger.save_config()
        self.logger.info(f"Initialize PPO Trainer: run name={self.run_name}")

        obs, _ = self.reset_env(self.env)
        ep_return = 0.0
        ep_len = 0

        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []

        last_update_info = None

        try:
            for step in range(1, self.total_timesteps + 1):
                self.global_step = step

                action, log_prob, value = self.agent.act(obs, deterministic=False)
                action_for_env = np.clip(action, self.env.action_space.low, self.env.action_space.high)

                next_obs, reward, done, terminated, truncated, info = self.step_env(action_for_env, self.env)

                rollout_obs.append(obs)
                rollout_actions.append(action)
                rollout_log_probs.append(log_prob)
                rollout_values.append(value)
                rollout_rewards.append(float(reward))
                rollout_dones.append(done)

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
                    obs, _ = self.reset_env(self.env)
                    ep_return = 0.0
                    ep_len = 0

                should_update = len(rollout_rewards) >= self.rollout_steps or step == self.total_timesteps
                if should_update:
                    last_value = 0.0 if done else self.agent.value(obs)
                    advantages, returns = self.compute_gae(
                        rollout_rewards,
                        rollout_values,
                        rollout_dones,
                        last_value,
                    )

                    rollout = {
                        "obs": np.asarray(rollout_obs, dtype=np.float32),
                        "actions": np.asarray(rollout_actions, dtype=np.float32),
                        "log_probs": np.asarray(rollout_log_probs, dtype=np.float32),
                        "advantages": advantages,
                        "returns": returns,
                    }

                    last_update_info = self.agent.update(rollout)

                    rollout_obs.clear()
                    rollout_actions.clear()
                    rollout_log_probs.clear()
                    rollout_values.clear()
                    rollout_rewards.clear()
                    rollout_dones.clear()

                if step % self.log_every == 0 and last_update_info is not None:
                    self.logger.log_train(step, last_update_info, print_to_console=True)

                if self.eval_every > 0 and step % self.eval_every == 0:
                    avg_return = self.evaluate(self.eval_episode)
                    is_best = self.logger.log_eval(step, avg_return)
                    if is_best:
                        self.save_best(step)

                if self.save_every > 0 and step % self.save_every == 0:
                    self.save_checkpoint(step)

            self.logger.info("Finished PPO Training")
            self.save_checkpoint(self.global_step, filename="final.pt")
        finally:
            self.logger.close()

    @torch.no_grad()
    def evaluate(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.eval_episode

        returns = []
        for _ in range(num_episodes):
            obs, _ = self.reset_env(self.eval_env)
            done = False
            ep_return = 0.0

            while not done:
                action, _, _ = self.agent.act(obs, deterministic=True)
                action_for_env = np.clip(action, self.eval_env.action_space.low, self.eval_env.action_space.high)
                next_obs, reward, done, terminated, truncated, info = self.step_env(action_for_env, self.eval_env)
                ep_return += float(reward)
                obs = next_obs

            returns.append(ep_return)

        return sum(returns) / len(returns)

    def save_checkpoint(self, step, filename=None):
        if filename is None:
            filename = f"ppo_checkpoint_{step}.pt"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        checkpoint = {
            "step": step,
            "episode_num": self.episode_num,
            "config": self.config,
            "agent_state": self.get_agent_state(),
        }
        torch.save(checkpoint, ckpt_path)
        self.logger.log_checkpoint(path=ckpt_path, step=step, kind="checkpoint")

    def save_best(self, step):
        os.makedirs(self.best_dir, exist_ok=True)
        best_path = os.path.join(self.best_dir, "ppo_best.pt")
        checkpoint = {
            "step": step,
            "episode_num": self.episode_num,
            "config": self.config,
            "agent_state": self.get_agent_state(),
        }
        torch.save(checkpoint, best_path)
        self.logger.log_checkpoint(path=best_path, step=step, kind="best model")

    def save_model_only(self, step=None):
        if step is None:
            filename = "ppo_model.pt"
        else:
            filename = f"ppo_model_{step}.pt"
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, filename)
        torch.save(
            {
                "actor": self.agent.actor.state_dict(),
                "critic": self.agent.critic.state_dict(),
            },
            model_path,
        )
        self.logger.log_checkpoint(path=model_path, step=step, kind="model")

    def get_agent_state(self):
        state: dict[str, Any] = self.agent.get_state()
        return state
