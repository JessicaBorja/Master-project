import numpy as np
from collections import deque, namedtuple
from sac_agent.sac_utils.utils import tt


class ReplayBuffer:
    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, max_size, dict_state=False):
        self._transition = namedtuple("transition",
                                      ["states", "actions", "rewards",
                                       "next_states", "terminal_flags"])
        # self._data = self._data(states=[], actions=[], rewards=[],
        #                         next_states=[], terminal_flags=[])
        self._data = deque(maxlen=int(max_size))
        self._max_size = max_size
        self.dict_state = dict_state

    def __len__(self):
        return len(self._data)

    def add_transition(self, state, action, reward, next_state, done):
        transition = self._transition(state, action, reward, next_state, done)
        self._data.append(transition)

    def sample(self, batch_size):
        batch_indices = np.random.choice(len(self._data), batch_size)
        (batch_states, batch_actions, batch_rewards,
         batch_next_states, batch_terminal_flags) = \
            zip(*[self._data[i] for i in batch_indices])

        batch_actions = np.array(batch_actions)
        batch_rewards = np.array(batch_rewards)
        batch_terminal_flags = np.array(batch_terminal_flags).astype('uint8')
        if(self.dict_state):
            v = {k: np.array([dic[k] for dic in batch_states])
                 for k in batch_states[0]}
            batch_states = v
            v = {k: np.array([dic[k] for dic in batch_next_states])
                 for k in batch_next_states[0]}
            batch_next_states = v

        return tt(batch_states), tt(batch_actions), tt(batch_rewards),\
            tt(batch_next_states), tt(batch_terminal_flags)
