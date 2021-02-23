import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from sac_agent.sac_utils.utils import get_activation_fn
from sac_agent.networks.networks_common import get_pos_shape, get_depth_network, get_concat_features, get_img_network


# policy
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_max,
                 activation="relu", hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.sigma = nn.Linear(hidden_dim, action_dim)
        self._activation = get_activation_fn(activation)
        self.action_max = action_max

    def forward(self, x):
        x = self._activation(self.fc1(x))
        x = self._activation(self.fc2(x))
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))
        # ensure sigma is between "0" and 1
        return mu, sigma

    # return action scaled to env
    def act(self, curr_obs, deterministic=False, reparametrize=False):
        mu, sigma = self.forward(curr_obs)  # .squeeze(0)
        log_probs = None
        if(deterministic):
            action = torch.tanh(mu) * self.action_max
        else:
            dist = Normal(mu, sigma)
            if(reparametrize):
                sample = dist.rsample()
            else:
                sample = dist.sample()
            action = torch.tanh(sample) * self.action_max
            # For updating policy, Apendix of SAC paper
            # unsqueeze because log_probs is of dim (batch_size, action_dim)
            # but the torch.log... is (batch_size)
            log_probs = dist.log_prob(sample) - \
                torch.log((1 - torch.tanh(sample).pow(2) + 1e-6))
            # +1e-6 to avoid no log(0)
            log_probs = log_probs.sum(-1)  # , keepdim=True)

        return action, log_probs


class CNNPolicy(nn.Module):
    def __init__(self, obs_space, action_dim, action_max,
                 activation="relu", hidden_dim=256):
        super(CNNPolicy, self).__init__()
        self.action_max = action_max
        _position_shape = get_pos_shape(obs_space)
        self.cnn_depth = get_depth_network(
                            obs_space,
                            out_feat=8,
                            activation=activation)
        self.cnn_img = get_img_network(
                            obs_space,
                            out_feat=8,
                            activation=activation)

        out_feat = 16 if self.cnn_depth is not None else 8
        out_feat += _position_shape

        self.fc1 = nn.Linear(out_feat, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.sigma = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        features = get_concat_features(obs, self.cnn_img, self.cnn_depth)

        x = F.elu(self.fc1(features))
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))
        return mu, sigma

    # return action scaled to env
    def act(self, curr_obs, deterministic=False, reparametrize=False):
        mu, sigma = self.forward(curr_obs)  # .squeeze(0)
        log_probs = None
        if(deterministic):
            action = torch.tanh(mu) * self.action_max
        else:
            dist = Normal(mu, sigma)
            if(reparametrize):
                sample = dist.rsample()
            else:
                sample = dist.sample()
            action = torch.tanh(sample) * self.action_max
            # For updating policy, Apendix of SAC paper
            # unsqueeze because log_probs is of dim (batch_size, action_dim)
            # but the torch.log... is (batch_size)
            log_probs = dist.log_prob(sample) -\
                torch.log((1 - torch.tanh(sample).pow(2) + 1e-6))
            log_probs = log_probs.sum(-1)  # , keepdim=True)
        return action, log_probs


class legacy_CNNPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, action_max, hidden_dim=256, activation=F.relu):
        super(legacy_CNNPolicy, self).__init__()
        self.action_max = action_max
        self.non_linearity = F.relu

        _history_length = state_dim['img_obs'].shape[0]
        self.conv1 = nn.Conv2d(_history_length, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        _img_size = state_dim['img_obs'].shape[-1]
        w, h = self.calc_out_size(_img_size, _img_size, 8, 0, 4)
        w, h = self.calc_out_size(w, h, 4, 0, 2)
        w, h = self.calc_out_size(w, h, 3, 0, 1)

        self.fc1 = nn.Linear(w*h*64, 16)
        _position_shape = state_dim['robot_obs'].shape[0]
        _position_shape = 0
        self.fc2 = nn.Linear(_position_shape + 16, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.sigma = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        img = x['img_obs']
        # pos = x['robot_obs']
        # cv2.waitKey(1)
        if(len(img.shape) == 3):
            img = img.unsqueeze(0)
        batch_size = img.shape[0]
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(batch_size, -1))).squeeze()  # bs, 16
        # print("Img features", x)
        # x = torch.cat((pos, x), dim=-1)
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))
        return mu, sigma

    # return action scaled to env
    def act(self, curr_obs, deterministic=False, reparametrize=False):
        mu, sigma = self.forward(curr_obs)  # .squeeze(0)
        log_probs = None
        if(deterministic):
            action = torch.tanh(mu) * self.action_max
        else:
            dist = Normal(mu, sigma)
            if(reparametrize):
                sample = dist.rsample()
            else:
                sample = dist.sample()
            action = torch.tanh(sample) * self.action_max
            log_probs = dist.log_prob(sample) - \
                torch.log((1 - torch.tanh(sample).pow(2) + 1e-6))
            log_probs = log_probs.sum(-1)  # , keepdim=True)
        return action, log_probs

    def calc_out_size(self, w, h, kernel_size, padding, stride):
        width = (w - kernel_size + 2*padding)//stride + 1
        height = (h - kernel_size + 2*padding)//stride + 1
        return width, height
