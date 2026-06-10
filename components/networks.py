import torch 
import torch.nn as nn
from torch.distributions import Normal

activation = {
    "Linear"    : nn.Identity(), 
    "ReLU"      : nn.ReLU(), 
    "ELU"       : nn.ELU(), 
    "Tanh"      : nn.Tanh(), 
    "LeakyReLU" : nn.LeakyReLU(), 
    "Sigmoid"   : nn.Sigmoid(), 
    "Softmax-1" : nn.Softmax(dim=-1),
    "Softmax0"  : nn.Softmax(dim=0),
    "Softmax1"  : nn.Softmax(dim=1),
    "Softmax2"  : nn.Softmax(dim=2)
}

EPS = 1e-6
LOG_STD_MIN = -20 
LOG_STD_MAX = 2 

def atanh(x): 
    x = torch.clamp(x, -1.0 + EPS, 1.0 - EPS) 
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

class MLP(nn.Module): 
    def __init__(self, layer_dims, hidden_act="ReLU", output_act="Linear"): 
        super().__init__()
        layers = []
        for i in range(len(layer_dims)-1): 
            act = hidden_act if i+2!=len(layer_dims) else output_act
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            layers.append(activation[act])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x): 
        return self.mlp(x)

class ActorNetwork(nn.Module): 

    def __init__(self, obs_dim, act_dim, hidden_size=[32, 32], 
                 hidden_act="ReLU", output_act="Linear"): 
        super().__init__()
        layer_dims = [obs_dim, *hidden_size]
        self.net = MLP(layer_dims, hidden_act, output_act)
        self.mu = nn.Linear(layer_dims[-1], act_dim)
        self.log_std = nn.Linear(layer_dims[-1], act_dim)

    def forward(self, obs): 
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std 
    
    @torch.no_grad()
    def act_deterministic(self, obs): 
        mu, _ = self.forward(obs)
        return torch.tanh(mu)

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std) 
        dist = Normal(mu, std) 
        u = dist.rsample() 
        a = torch.tanh(u)
        log_prob_u = dist.log_prob(u).sum(dim=-1, keepdim=True)
        log_det_jacobian = torch.log(1.0 - a.pow(2) + EPS).sum(dim=-1, keepdim=True)
        log_prob = log_prob_u - log_det_jacobian
        mu_action = torch.tanh(mu)
        return a, log_prob, mu_action

    def log_prob(self, obs, act): 
        if obs.ndim == 1: 
            obs = obs.unsqueeze(0)
        if act.ndim == 1: 
            act = act.unsqueeze(0)
        action = torch.clamp(act, -1.0 + EPS, 1.0 - EPS)
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        u = atanh(act)
        log_prob_u = dist.log_prob(u).sum(dim=-1, keepdim=True)
        log_det_jacobian = torch.log(1.0 - act.pow(2) + EPS).sum(dim=-1, keepdim=True)
        log_prob = log_prob_u - log_det_jacobian
        return log_prob

class QCriticNetwork(nn.Module): 

    def __init__(self, obs_dim, act_dim, hidden_size=[32, 32], 
                 hidden_act="ReLU", output_act="Linear"): 
        super().__init__()
        layer_dims = [obs_dim+act_dim, *hidden_size, 1]
        self.criticnet = MLP(layer_dims, hidden_act, output_act)

    def forward(self, obs, act): 
        if act.ndim==1: act = torch.unsqueeze(act, dim=1)
        x = torch.cat([obs, act], dim=1)
        return self.criticnet(x)

class ActorDoubleQCriticNetwork(nn.Module): 

    def __init__(self, obs_dim, act_dim, 
                 hidden_size_actor=[32, 32], hidden_size_critic=[32, 32], 
                 hidden_act="ReLU", output_act="Linear"):
        super().__init__()
        self.actor = ActorNetwork(obs_dim, act_dim, hidden_size_actor, hidden_act, output_act)
        self.critic1 = QCriticNetwork(obs_dim, act_dim, hidden_size_critic, hidden_act, output_act)
        self.critic2 = QCriticNetwork(obs_dim, act_dim, hidden_size_critic, hidden_act, output_act)

    def sample_action(self, obs): 
        action, log_pi, action_mean = self.actor.sample(obs)
        return action, log_pi, action_mean 

    def act_deterministic(self, obs):
        return self.actor.act_deterministic(obs)

    def getQvalues(self, obs, act): 
        q1 = self.critic1(obs, act) 
        q2 = self.critic2(obs, act) 
        return q1, q2
