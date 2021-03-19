import torch
import torch.nn as nn
import torch.nn.functional as F
from sac_agent.sac_utils.utils import get_activation_fn
from sac_agent.networks.networks_common import \
     get_pos_shape, get_depth_network, get_img_network, \
     get_gripper_network, get_concat_features


# q function
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,
                 activation="relu", hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
        self._activation = get_activation_fn(activation)

    def forward(self, states, actions):
        x = torch.cat((states, actions), -1)
        x = self._activation(self.fc1(x))
        x = self._activation(self.fc2(x))
        return self.q(x).squeeze()


class CNNCritic(nn.Module):
    def __init__(self, obs_space, action_dim,
                 hidden_dim=256, activation="relu"):
        super(CNNCritic, self).__init__()
        _position_shape = get_pos_shape(obs_space)
        self.cnn_depth = get_depth_network(
                        obs_space,
                        out_feat=8,
                        activation=activation)
        self.cnn_img = get_img_network(
                        obs_space,
                        out_feat=8,
                        activation=activation)
        self.cnn_gripper = get_gripper_network(
                        obs_space,
                        out_feat=8,
                        activation=activation)
        out_feat = 0
        for net in [self.cnn_img, self.cnn_depth, self.cnn_gripper]:
            if(net is not None):
                out_feat += 8
        out_feat += _position_shape + action_dim

        self._activation = activation
        self.fc1 = nn.Linear(out_feat, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        features = get_concat_features(states,
                                       self.cnn_img,
                                       self.cnn_depth,
                                       self.cnn_gripper)
        x = torch.cat((features, actions), -1)
        x = F.elu(self.fc1(x))
        x = self.q(x).squeeze()
        return x


class legacy_CNNCritic(nn.Module):
    def __init__(self, state_dim, action_dim,
                 activation=F.relu, hidden_dim=256):
        super(legacy_CNNCritic, self).__init__()
        _history_length = state_dim['img_obs'].shape[0]
        self.conv1 = nn.Conv2d(_history_length, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 5)

        _img_size = state_dim['img_obs'].shape[-1]
        w, h = self.calc_out_size(_img_size, _img_size, 5, 0, 1)
        w, h = self.calc_out_size(w, h, 3, 0, 1)
        w, h = self.calc_out_size(w, h, 5, 0, 1)

        self.fc1 = nn.Linear(w*h*16, 16)
        _position_shape = state_dim['robot_obs'].shape[0]
        self._q = CriticNetwork(_position_shape + 16, action_dim,
                                activation=activation,
                                hidden_dim=hidden_dim)

    def forward(self, states, actions):
        img = states['img_obs']
        pos = states['robot_obs']

        batch_size = img.shape[0]
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(batch_size, -1)))  # bs, 16

        b_states = torch.cat((pos, x), dim=-1)
        x = self._q(b_states, actions)
        return x

    def calc_out_size(self, w, h, kernel_size, padding, stride):
        width = (w - kernel_size + 2*padding)//stride + 1
        height = (h - kernel_size + 2*padding)//stride + 1
        return width, height
