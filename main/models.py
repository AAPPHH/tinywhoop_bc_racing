import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def ortho_init(module, gain=1.0):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-4):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('count', torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, x):
        if x.shape[0] < 2:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, correction=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)


NUM_BINS = 51

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.imu_rms = RunningMeanStd((10,))
        self.cv2_rms = RunningMeanStd((10,))
        self.cv2_proj = nn.Linear(10, 32)
        self.imu_proj = nn.Linear(40, 32)
        self.nav_proj = nn.Linear(6, 16)
        self.act_proj = nn.Linear(16, 16)
        self.head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 4 * NUM_BINS),
        )
        self.register_buffer('bins', torch.linspace(-1.0, 1.0, NUM_BINS))
        self.aux_head = nn.Linear(32, 8)

    def forward(self, obs, return_aux=False):
        imu = obs["imu"]
        cv2_feat = obs["cv2"]
        nav_feat = obs["nav"]
        act_hist = obs["actions"]
        B = cv2_feat.shape[0]
        if self.training:
            self.imu_rms.update(imu.reshape(-1, 10))
            self.cv2_rms.update(cv2_feat)
        imu_normed = self.imu_rms.normalize(imu)
        cv2_normed = self.cv2_rms.normalize(cv2_feat)
        cv2_f = F.relu(self.cv2_proj(cv2_normed))
        imu_f = self.imu_proj(imu_normed.reshape(B, -1))
        nav_f = F.relu(self.nav_proj(nav_feat))
        act_f = self.act_proj(act_hist.reshape(B, -1))
        combined = torch.cat([cv2_f, imu_f, nav_f, act_f], dim=-1)
        logits = self.head(combined).view(B, 4, NUM_BINS)
        logits = logits - logits.mean(dim=-1, keepdim=True)
        if return_aux:
            aux_pred = self.aux_head(cv2_f)
            return logits, aux_pred
        return logits

    def get_distribution(self, obs, return_aux=False):
        if return_aux:
            logits, aux_pred = self.forward(obs, return_aux=True)
            return Categorical(logits=logits), aux_pred
        return Categorical(logits=self.forward(obs))


class PopArtLayer(nn.Module):
    def __init__(self, in_features, out_features, beta=3e-4):
        super().__init__()
        self.beta = beta
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('sigma', torch.ones(1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        normalized = F.linear(x, self.weight, self.bias)
        return normalized * self.sigma + self.mu

    def forward_normalized(self, x):
        return F.linear(x, self.weight, self.bias)

    def normalize_targets(self, targets):
        return (targets - self.mu) / self.sigma

    @torch.no_grad()
    def update_stats(self, targets):
        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()
        new_mu = targets.mean()
        new_sigma = targets.std().clamp(min=1e-4)
        self.mu.copy_((1 - self.beta) * self.mu + self.beta * new_mu)
        self.sigma.copy_((1 - self.beta) * self.sigma + self.beta * new_sigma)
        self.weight.data.mul_(old_sigma / self.sigma)
        self.bias.data.copy_((old_sigma * self.bias.data + old_mu - self.mu) / self.sigma)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_rms = RunningMeanStd((38,))
        self.trunk = nn.Sequential(
            nn.Linear(38, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.popart = PopArtLayer(128, 1)

    def forward(self, obs, update_stats=True):
        state = obs["state"]
        if update_stats and self.training:
            self.state_rms.update(state)
        state_normed = self.state_rms.normalize(state)
        return self.popart(self.trunk(state_normed))

    def forward_normalized(self, obs, update_stats=False):
        state = obs["state"]
        if update_stats and self.training:
            self.state_rms.update(state)
        state_normed = self.state_rms.normalize(state)
        return self.popart.forward_normalized(self.trunk(state_normed))


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = Actor()
        self.critic = Critic()
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                ortho_init(m, gain=math.sqrt(2))
        ortho_init(self.actor.head[-1], gain=0.01)
        ortho_init(self.critic.popart, gain=1.0)

    def get_action(self, obs, deterministic=False):
        if deterministic:
            logits = self.actor(obs)
            probs = F.softmax(logits, dim=-1)
            action = (probs * self.actor.bins).sum(dim=-1)
            return action, torch.zeros(action.shape[0], device=action.device), None
        dist = self.actor.get_distribution(obs)
        indices = dist.sample()
        log_prob = dist.log_prob(indices).sum(dim=-1)
        action = self.actor.bins[indices]
        return action, log_prob, indices

    def get_value(self, obs):
        return self.critic(obs, update_stats=False)

    def evaluate(self, obs, indices):
        dist, aux_pred = self.actor.get_distribution(obs, return_aux=True)
        log_prob = dist.log_prob(indices).sum(dim=-1)
        entropy_per_motor = dist.entropy()
        entropy = entropy_per_motor.sum(dim=-1)
        value = self.critic(obs)
        normalized_value = self.critic.forward_normalized(obs)
        return log_prob, entropy, entropy_per_motor, value, normalized_value, aux_pred

    def update_popart(self, returns):
        self.critic.popart.update_stats(returns)

    def normalize_targets(self, returns):
        return self.critic.popart.normalize_targets(returns)
