import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

from components.networks import ActorNetwork, VCriticNetwork, MLP
from utils.helper import getObsActDim

class PPO_Agent: 
    def __init__(self, config):
        self.config = config
        self.device = config["train"].get("device", "auto")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device

        env_name = config["env"].get("name", "FetchReachDense-v4")
        env_kwargs = config["env"].get("kwargs", {})
        obs_dim, act_dim = getObsActDim(
            env_name,
            max_episode_steps=config["env"].get("max_episode_steps", 0),
            reward_scaler=config["env"].get("reward_scaler", 1.0),
            flatten_obs=config["env"].get("flatten_obs", True),
            **env_kwargs,
        )

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.gamma = float(config["train"].get("gamma", 0.99))
        self.gae_lambda = float(config["train"].get("gae_lambda", 0.95))
        self.clip_range = float(config["train"].get("clip_range", 0.2))
        self.entropy_coef = float(config["train"].get("entropy_coef", 0.0))
        self.value_coef = float(config["train"].get("value_coef", 0.5))
        self.max_grad_norm = float(config["train"].get("max_grad_norm", 0.5))
        self.n_epochs = int(config["train"].get("n_epochs", 10))
        self.batch_size = int(config["train"].get("batch_size", 64))

        hidden_size_actor = config["train"].get("hidden_size_actor", [256, 256])
        hidden_size_critic = config["train"].get("hidden_size_critic", [256, 256])

        actor_lr = float(config["train"]["optimizer"].get("actor_lr", 3e-4))
        critic_lr = float(config["train"]["optimizer"].get("critic_lr", 3e-4))

        self.actor = ActorNetwork(obs_dim, act_dim, hidden_size_actor).to(self.device)
        self.critic = VCriticNetwork(obs_dim, hidden_size_critic).to(self.device)

        opt_name = config["train"]["optimizer"].get("name", "Adam")
        if opt_name != "Adam":
            raise ValueError(f"Unsupported optimizer {opt_name}")

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _to_tensor(self, x):
        if torch.is_tensor(x):
            x = x.to(self.device, dtype=torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs_t = self._to_tensor(obs)

        if deterministic:
            action = self.actor.act_deterministic(obs_t)
            log_prob = self.actor.log_prob(obs_t, action)
        else:
            action, log_prob, _ = self.actor.sample(obs_t)

        value = self.critic(obs_t)

        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.squeeze().cpu().item()),
            float(value.squeeze().cpu().item()),
        )

    @torch.no_grad()
    def value(self, obs):
        obs_t = self._to_tensor(obs)
        return float(self.critic(obs_t).squeeze().cpu().item())

    def update(self, rollout):
        obs = torch.tensor(rollout["obs"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(rollout["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.tensor(rollout["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(rollout["advantages"], dtype=torch.float32, device=self.device)

        old_log_probs = old_log_probs.view(-1)
        returns = returns.view(-1)
        advantages = advantages.view(-1)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = obs.shape[0]
        last_info = {}

        for _ in range(self.n_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                new_log_probs = self.actor.log_prob(batch_obs, batch_actions).view(-1)
                values = self.critic(batch_obs).view(-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped_obj = ratio * batch_advantages
                clipped_obj = torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                ) * batch_advantages

                actor_loss = -torch.min(unclipped_obj, clipped_obj).mean()
                entropy_bonus = -new_log_probs.mean()
                critic_loss = F.mse_loss(values, batch_returns)

                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_bonus

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                approx_kl = (batch_old_log_probs - new_log_probs).mean().detach()
                clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().detach()

                last_info = {
                    "actor_loss": float(actor_loss.detach().cpu().item()),
                    "critic_loss": float(critic_loss.detach().cpu().item()),
                    "total_loss": float(total_loss.detach().cpu().item()),
                    "entropy": float(entropy_bonus.detach().cpu().item()),
                    "approx_kl": float(approx_kl.cpu().item()),
                    "clip_fraction": float(clip_fraction.cpu().item()),
                    "value_mean": float(values.mean().detach().cpu().item()),
                    "log_pi_mean": float(new_log_probs.mean().detach().cpu().item()),
                    # These keys keep compatibility with the current Logger class.
                    "q1_loss": 0.0,
                    "q2_loss": 0.0,
                    "q1_mean": 0.0,
                    "q2_mean": 0.0,
                }

        return last_info

    def get_state(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
