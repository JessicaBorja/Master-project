import numpy as np
from collections import deque, namedtuple
from vapo.sac_agent.sac_utils.utils import tt
from pathlib import Path


class ReplayBuffer:
    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, max_size, dict_state=False, logger=None):
        self._transition = namedtuple("transition",
                                      ["state", "action", "reward",
                                       "next_state", "terminal_flag"])
        self._data = deque(maxlen=int(max_size))
        self._max_size = max_size
        self.dict_state = dict_state
        self.last_saved_idx = 0
        self.logger = logger

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

    def save(self, path="./replay_buffer"):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        num_entries = len(self._data)
        for i in range(self.last_saved_idx, num_entries):
            transition = self._data[i]
            file_name = "%s/transition_%d.npz" % (path, i)
            np.savez(file_name,
                     state=transition.state,
                     action=transition.action,
                     next_state=transition.next_state,
                     reward=transition.reward,
                     terminal_flag=transition.terminal_flag)
        self.logger.info("Saved transitions with indices : %d - %d" % (self.last_saved_idx, i))
        self.last_saved_idx = i

    def load(self, path="./replay_buffer"):
        p = Path(path)
        if p.is_dir():
            p = p.glob('*.npz')
            files = [x for x in p if x.is_file()]
            if len(files) > 0:
                for file in files:
                    data = np.load(file, allow_pickle=True)
                    transition = self._transition(data['state'].item(),
                                                  data['action'],
                                                  data['next_state'].item(),
                                                  data['reward'].item(),
                                                  data['terminal_flag'].item())
                    self._data.append(transition)
                self.last_saved_idx = len(files)
                self.logger.info("Replay buffer loaded successfully")
            else:
                self.logger.info("No files were found in path %s" % (path))
        else:
            self.logger.info("Path %s does not have an appropiate directory address" % (path))