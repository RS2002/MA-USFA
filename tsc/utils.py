import numpy as np
import random
import os
import torch
import xml.etree.ElementTree as ET


def set_seed(seed, deterministic_torch=True):
    """Seed every RNG so a run is bit-for-bit reproducible.

    Covers the four stochastic sources in this codebase:
      * Python ``random``      -- replay-buffer sampling (Buffer.get)
      * NumPy                  -- PPO minibatch indices (np.random.randint)
      * PyTorch (CPU+CUDA)     -- weight init and action sampling
                                  (torch.multinomial)
    The SUMO vehicle stream is seeded separately at env.reset(seed=...) by the
    caller, because it lives in the environment, not here.

    With ``deterministic_torch`` we also force cuDNN into deterministic mode.
    Because k_hop/h_hop only change the S^A / S^R matrices (never the model
    architecture or the rollout), seeding here guarantees that, for a fixed
    (method, seed), the *first* episode's rollout is identical across all hop
    settings -- the sanity check the user asked for.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Make Adam/matmul reductions order-stable where CUDA allows it.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def parse_tripinfo(path):
    """Parse a SUMO tripinfo XML into standard evaluation metrics.

    SUMO writes one ``<tripinfo>`` element per *completed* trip when the sim is
    launched with ``--tripinfo-output <path>`` and then closed. This gives the
    two metrics almost every traffic-signal-control paper reports but that are
    not available online: average travel time and throughput.

    Returns a dict with:
        travel_time : mean trip ``duration`` (s)          -- primary TSC metric
        time_loss   : mean ``timeLoss`` (s, delay vs free-flow)
        wait_time   : mean ``waitingTime`` (s)
        throughput  : number of completed trips
    Missing/empty/parse-failing files return NaNs and 0 throughput, so callers
    never crash on a run where no vehicle finished.
    """
    empty = {'travel_time': float('nan'), 'time_loss': float('nan'),
             'wait_time': float('nan'), 'throughput': 0}
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return empty
    try:
        durations, losses, waits = [], [], []
        # iterparse keeps memory flat on large networks.
        for _, elem in ET.iterparse(path, events=("end",)):
            if elem.tag == "tripinfo":
                durations.append(float(elem.get("duration", "nan")))
                losses.append(float(elem.get("timeLoss", "nan")))
                waits.append(float(elem.get("waitingTime", "nan")))
                elem.clear()
        if not durations:
            return empty
        return {
            'travel_time': float(np.nanmean(durations)),
            'time_loss': float(np.nanmean(losses)),
            'wait_time': float(np.nanmean(waits)),
            'throughput': len(durations),
        }
    except ET.ParseError:
        # File was still being written / truncated; treat as unavailable.
        return empty


class Buffer:
    """Rollout buffer for CMAT.

    Design notes
    ------------
    * ``next_states`` is passed in explicitly by the caller (``Agent.step``)
      rather than being back-patched from the following ``append``. This
      removes two bugs present in the previous implementation:
        (1) the last transition's next_state pointing to itself, and
        (2) cross-episode contamination, where the first transition of a new
            episode overwrote the terminal next_state of the previous episode.
    * The GAE/critic bootstrap already masks terminal transitions with
      ``(1 - done)``, so the stored terminal next_state is never actually used
      for value propagation; storing a copy just keeps array shapes uniform.
    """

    def __init__(self, buffer_size=10000):
        self.buffer_size = buffer_size
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []
        self.consensus = []

    @property
    def num(self):
        return len(self.states)

    def append(self, states, actions, consensus, reward, values, log_probs, done, next_states):
        self.states.append(states)
        self.actions.append(actions)
        self.consensus.append(consensus)
        self.rewards.append(reward)
        self.values.append(values)
        self.log_probs.append(log_probs)
        self.dones.append(done)
        self.next_states.append(next_states)

        if self.num > self.buffer_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.consensus.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.dones.pop(0)
            self.next_states.pop(0)

    def _stack(self, indices=None):
        if indices is None:
            sel = lambda buf: np.stack(buf)
            dones = np.array(self.dones)
        else:
            sel = lambda buf: np.stack([buf[i] for i in indices])
            dones = np.array([self.dones[i] for i in indices])
        return (
            sel(self.states),       # [T, N, s]
            sel(self.actions),      # [T, N]
            sel(self.rewards),      # [T, N]
            sel(self.values),       # [T, N]
            sel(self.log_probs),    # [T, N]
            dones,                  # [T]
            sel(self.consensus),    # [T, N, c]
            sel(self.next_states),  # [T, N, s]
        )

    def get(self, batch_size=None):
        if batch_size is None:
            return self._stack()
        # Sample with replacement over all stored transitions. next_states are
        # now correct for every index (including terminals), so no index guard
        # is required.
        indices = random.choices(range(self.num), k=batch_size)
        return self._stack(indices)


class ReplayBuffer:
    """Off-policy replay buffer for IDDQN.

    Each stored transition is a full multi-agent step:
        state      [N, s]   observation (+timestep) of every intersection
        action     [N]      discrete action chosen at every intersection
        reward     [N]      per-agent reward
        next_state [N, s]   observation after the environment step
        done       scalar   episode-termination flag (shared across agents)

    A fixed-capacity ring buffer (oldest transitions overwritten) with uniform
    random minibatch sampling -- the standard DQN experience replay. Unlike the
    on-policy PPO ``Buffer`` this is NOT cleared each episode; transitions from
    many episodes coexist so the Q-network trains on decorrelated samples.
    """

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self._pos = 0

    @property
    def num(self):
        return len(self.states)

    def append(self, state, action, reward, next_state, done):
        if self.num < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
        else:
            # Overwrite the oldest slot (ring buffer).
            self.states[self._pos] = state
            self.actions[self._pos] = action
            self.rewards[self._pos] = reward
            self.next_states[self._pos] = next_state
            self.dones[self._pos] = done
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size):
        """Uniformly sample a minibatch. Returns numpy stacks:
        (states [B,N,s], actions [B,N], rewards [B,N], next_states [B,N,s],
         dones [B])."""
        indices = random.choices(range(self.num), k=batch_size)
        states = np.stack([self.states[i] for i in indices])
        actions = np.stack([self.actions[i] for i in indices])
        rewards = np.stack([self.rewards[i] for i in indices])
        next_states = np.stack([self.next_states[i] for i in indices])
        dones = np.array([self.dones[i] for i in indices])
        return states, actions, rewards, next_states, dones