import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, RelaxedOneHotCategorical
from sac_agent.sac_utils.utils import get_activation_fn
from sac_agent.networks.networks_common import \
     get_pos_shape, get_depth_network, get_img_network, \
     get_gripper_network, get_concat_features


# policy
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_space,
                 activation="relu", hidden_dim=256, **kwargs):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.sigma = nn.Linear(hidden_dim, action_dim)
        self._activation = get_activation_fn(activation)
        self.action_high = action_space.high[0]
        self.action_low = action_space.low[0]

    def forward(self, x):
        x = self._activation(self.fc1(x))
        x = self._activation(self.fc2(x))
        mu = self.mu(x)
        log_sigma = self.sigma(x)
        # avoid log_sigma to go to infinity
        sigma = torch.clamp(log_sigma, -20, 2).exp()
        return mu, sigma

    def scale_action(self, action):
        slope = (self.action_high - self.action_low) / 2
        action = self.action_low + slope * (action + 1)
        return action

    # return action scaled to env
    def act(self, curr_obs, deterministic=False, reparametrize=False):
        mu, sigma = self.forward(curr_obs)  # .squeeze(0)
        log_probs = None
        if(deterministic):
            action = torch.tanh(mu)
        else:
            dist = Normal(mu, sigma)
            if(reparametrize):
                sample = dist.rsample()
            else:
                sample = dist.sample()
            action = torch.tanh(sample)
            # For updating policy, Apendix of SAC paper
            # unsqueeze because log_probs is of dim (batch_size, action_dim)
            # but the torch.log... is (batch_size)
            log_probs = dist.log_prob(sample) - \
                torch.log((1 - action.square() + 1e-6))
            # +1e-6 to avoid no log(0)
            log_probs = log_probs.sum(-1)  # , keepdim=True)
        action = self.scale_action(action)
        return action, log_probs


class CNNPolicy(nn.Module):
    def __init__(self, obs_space, action_dim, action_space, affordance=None,
                 activation="relu", hidden_dim=256):
        super(CNNPolicy, self).__init__()
        self.action_high = torch.tensor(action_space.high).cuda()
        self.action_low = torch.tensor(action_space.low).cuda()
        _position_shape = get_pos_shape(obs_space)
        self.cnn_depth = get_depth_network(
                            obs_space,
                            out_feat=8,
                            activation=activation)
        self.cnn_img = get_img_network(
                            obs_space,
                            out_feat=8,
                            activation=activation,
                            affordance_cfg=affordance)
        self.cnn_gripper = get_gripper_network(
                            obs_space,
                            out_feat=8,
                            activation=activation,
                            affordance_cfg=affordance)
        out_feat = 0
        for net in [self.cnn_img, self.cnn_depth, self.cnn_gripper]:
            if(net is not None):
                out_feat += 8
        out_feat += _position_shape

        self.fc1 = nn.Linear(out_feat, hidden_dim)
        # Last dimension of action_dim is gripper_action
        self.mu = nn.Linear(hidden_dim, action_dim - 1)
        self.sigma = nn.Linear(hidden_dim, action_dim - 1)
        self.gripper_action = nn.Linear(hidden_dim, 2)  # open / close

    def forward(self, obs):
        features = get_concat_features(obs,
                                       self.cnn_img,
                                       self.cnn_depth,
                                       self.cnn_gripper)

        x = F.elu(self.fc1(features))
        mu = self.mu(x)
        log_sigma = self.sigma(x)
        gripper_a = F.softmax(self.gripper_action(x))
        # avoid log_sigma to go to infinity
        sigma = torch.clamp(log_sigma, -20, 2).exp()
        return mu, sigma, gripper_a

    def scale_action(self, action):
        slope = (self.action_high - self.action_low) / 2
        action = self.action_low + slope * (action + 1)
        return action

    # return action scaled to env
    def act(self, curr_obs, deterministic=False, reparametrize=False):
        mu, sigma, gripper = self.forward(curr_obs)  # .squeeze(0)
        log_probs = None
        if(deterministic):
            action = torch.tanh(mu)
            gripper_a = torch.argmax(gripper)
        else:
            dist = Normal(mu, sigma)
            gripper_dist = GumbelSoftmax(0.5)
            if(reparametrize):
                sample = dist.rsample()
                gripper_a = gripper_dist.rsample()
            else:
                sample = dist.sample()
                gripper_a = gripper_dist.sample()
            action = torch.tanh(sample)
            # For updating policy, Apendix of SAC paper
            # unsqueeze because log_probs is of dim (batch_size, action_dim)
            # but the torch.log... is (batch_size)
            log_probs = dist.log_prob(sample) -\
                torch.log((1 - action.square() + 1e-6))
            log_probs = log_probs.sum(-1)  # , keepdim=True)
            log_prob_a = gripper_dist.log_prob(gripper_a)

        # add gripper action
        action = torch.cat((action, gripper_a), -1)
        action = self.scale_action(action)

        # add gripper action to log_probs
        log_probs = torch.cat((log_probs, log_prob_a), -1)
        return action, log_probs


class GumbelSoftmax(RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size()):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)