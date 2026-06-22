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
    pass

