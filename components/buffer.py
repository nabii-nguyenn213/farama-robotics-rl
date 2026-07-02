import torch
import numpy as np 

class ReplayBuffer: 

    def __init__(self, max_size, obs_dim, act_dim): 
        obs_dim = (obs_dim, ) if isinstance(obs_dim, int) else tuple(obs_dim)
        self.memory_size = int(max_size)
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, *obs_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((self.memory_size, *obs_dim), dtype=np.float32)
        self.action_memory = np.zeros((self.memory_size, act_dim), dtype=np.float32)
        self.reward_memory = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.terminal_memory = np.zeros((self.memory_size, 1), dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done): 
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = np.asarray(state, dtype=np.float32)
        self.next_state_memory[index] = np.asarray(next_state, dtype=np.float32)
        self.action_memory[index] = np.asarray(action, dtype=np.float32)
        self.reward_memory[index] = np.asarray(reward, dtype=np.float32)
        self.terminal_memory[index] = float(done)
        self.memory_counter += 1

    def __len__(self): 
        return min(self.memory_counter, self.memory_size)

    def can_sample(self, batch_size):
        return len(self) >= batch_size

    def sample_buffer(self, batch_size, device="cpu"): 
        if not self.can_sample(batch_size):
            raise ValueError(f"Not enough samples in replay buffer {len(self)} < {batch_size}")
        batch = np.random.randint(0, len(self), size=batch_size) 
        return {
            "obs": torch.as_tensor(self.state_memory[batch], dtype=torch.float32, device=device),
            "act": torch.as_tensor(self.action_memory[batch], dtype=torch.float32, device=device),
            "rew": torch.as_tensor(self.reward_memory[batch], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_state_memory[batch], dtype=torch.float32, device=device),
            "done": torch.as_tensor(self.terminal_memory[batch], dtype=torch.float32, device=device),
            }

class RolloutBuffer: 
    pass

class HER_ReplayBuffer: 
    def __init__(self, max_size, env, her_ratio=0.8):
        self.max_size = int(max_size)
        self.env = env
        self.her_ratio = her_ratio

        self.episodes = []
        self.current_episode = []

    def _flatten_obs_goal(self, obs, goal):
        return np.concatenate(
            [
                np.asarray(obs, dtype=np.float32),
                np.asarray(goal, dtype=np.float32),
            ],
            axis=-1,
        ).astype(np.float32)
    
    def store_transition(self, obs, action, reward, next_obs, done, info=None):
        if info is None:
            info = {}

        transition = {
            "obs": np.asarray(obs["observation"], dtype=np.float32).copy(),
            "achieved_goal": np.asarray(obs["achieved_goal"], dtype=np.float32).copy(),
            "desired_goal": np.asarray(obs["desired_goal"], dtype=np.float32).copy(),

            "act": np.asarray(action, dtype=np.float32).copy(),
            "rew": float(reward),

            "next_obs": np.asarray(next_obs["observation"], dtype=np.float32).copy(),
            "next_achieved_goal": np.asarray(next_obs["achieved_goal"], dtype=np.float32).copy(),

            "done": float(done),
            "info": info,
        }

        self.current_episode.append(transition)

        if done:
            self.episodes.append(self.current_episode)
            self.current_episode = []

            if len(self.episodes) > self.max_size:
                self.episodes.pop(0)

    def __len__(self):
        return sum(len(ep) for ep in self.episodes)

    def can_sample(self, batch_size):
        return len(self.episodes) > 0 and len(self) >= batch_size
    
    def sample_buffer(self, batch_size, device='cpu'):
        if not self.can_sample(batch_size):
            raise ValueError(f"Not enough samples in HER buffer {len(self)} < {batch_size}")
        
        obs_batch = [] 
        act_batch = []
        rew_batch = []
        next_obs_batch = []
        done_batch = []

        for _ in range(batch_size):
            episode = self.episodes[np.random.randint(len(self.episodes))]
            t = np.random.randint(len(episode))

            transition = episode[t]

            obs = transition["obs"]
            desired_goal = transition["desired_goal"]
            action = transition["act"]
            reward = transition["rew"]
            next_obs = transition["next_obs"]
            next_achieved_goal = transition["next_achieved_goal"]
            done = transition["done"]
            info = transition["info"]

            if np.random.rand() < self.her_ratio:
                future_t = np.random.randint(t, len(episode))
                desired_goal = episode[future_t]["next_achieved_goal"]

                reward = self.env.unwrapped.compute_reward(
                    next_achieved_goal,
                    desired_goal,
                    info,
                )
            
            state_goal = self._flatten_obs_goal(obs, desired_goal)
            next_state_goal = self._flatten_obs_goal(next_obs, desired_goal)

            obs_batch.append(state_goal)
            act_batch.append(action)
            rew_batch.append([reward])
            next_obs_batch.append(next_state_goal)
            done_batch.append([done])
        
        return {
            "obs": torch.as_tensor(np.asarray(obs_batch), dtype=torch.float32, device=device),
            "act": torch.as_tensor(np.asarray(act_batch), dtype=torch.float32, device=device),
            "rew": torch.as_tensor(np.asarray(rew_batch), dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(np.asarray(next_obs_batch), dtype=torch.float32, device=device),
            "done": torch.as_tensor(np.asarray(done_batch), dtype=torch.float32, device=device),
        }

