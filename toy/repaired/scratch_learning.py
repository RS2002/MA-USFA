"""Audited target-specific scratch DDQN, restricted to the direct-only path.

Numerical operations are extracted from the original validated train_dqn
implementation. No source-conditioned or residual selector roles are exposed.
"""
from __future__ import annotations
import copy, dataclasses, hashlib, json, os, signal, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .legacy_world import SFWorldExact
from .exact import evaluate_policy
from .task_protocol import canonical_json

@dataclasses.dataclass
class LearningConfig:
    horizon: int = 40
    gamma: float = .95
    kappa: float = 1.
    hidden: int = 64
    lr: float = 1e-3
    batch: int = 64
    replay_capacity: int = 50000
    warmup: int = 256
    update_every: int = 4
    target_every: int = 1000
    checkpoint_every: int = 10000
    validation_every: int = 20000
    epsilon_final: float = .05
    epsilon_decay_fraction: float = .8
    grad_clip: float = 10.
    threads: int = 2
    source_steps: int = 100000  # per source, four sources
    sf_steps: int = 160000
    composer_steps: int = 80000
    no_sf_steps: int = 240000
    direct_steps: int = 640000

    def validate(self):
        if self.horizon != 40 or self.gamma != .95 or self.kappa != 1.:
            raise ValueError('This implementation is the declared N2/H40/gamma.95/kappa1 protocol')
        if not 0 < self.replay_capacity <= 50000:
            raise ValueError('Replay capacity must be bounded by 50000')
        for key in ('source_steps', 'sf_steps', 'composer_steps', 'no_sf_steps', 'direct_steps'):
            if getattr(self, key) <= 0 or getattr(self, key) % self.horizon:
                raise ValueError(f'{key} must be a positive multiple of H=40')
        if not 1 <= self.threads <= 4 or self.batch > self.replay_capacity:
            raise ValueError('Local thread/batch guard violated')
        for key in ('hidden', 'batch', 'warmup', 'update_every', 'target_every',
                    'checkpoint_every', 'validation_every'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if self.checkpoint_every % self.horizon or self.validation_every % self.horizon:
            raise ValueError('Checkpoint and validation intervals must align with full episodes')
        if not 0 < self.epsilon_final <= 1 or not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError('Invalid declared epsilon schedule')


def configure_torch(config, seed):
    config.validate()
    if config.threads > int(os.environ.get('MA_USFA_THREAD_BUDGET', 4)):
        raise ValueError('Training config exceeds the resource guardian thread budget')
    torch.set_num_threads(config.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def model_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        x = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(x.dtype).encode())
        h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device='cpu')


def observation(states, times, horizon=40):
    s = np.asarray(states, dtype=np.int64)
    t = np.broadcast_to(np.asarray(times), s.shape)
    cells = np.stack([s // 25, s % 25], -1)
    coords = np.stack([cells[..., 0] // 5, cells[..., 0] % 5,
                       cells[..., 1] // 5, cells[..., 1] % 5], -1) / 4.
    return np.concatenate([coords, ((horizon-t)/horizon)[..., None]], -1).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=64, zero_final=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, output_dim))
        if zero_final:
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)



class Replay:
    fields = {'state': (np.int16, ()), 'time': (np.uint8, ()),
              'next_state': (np.int16, ()), 'action': (np.int16, ()),
              'reward': (np.float32, ()), 'done': (np.float32, ()),
              'weights': (np.float32, (2, 4)), 'context': (np.int8, (2,)),
              'features': (np.float32, (2, 5))}

    def __init__(self, capacity):
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.data = {key: np.zeros((capacity,)+shape, dtype=dtype)
                     for key, (dtype, shape) in self.fields.items()}

    def append(self, **item):
        for key in self.fields:
            self.data[key][self.cursor] = item.get(key, 0)
        self.cursor = (self.cursor+1) % self.capacity
        self.size = min(self.capacity, self.size+1)

    def sample(self, rng, batch):
        ix = rng.choice(self.size, batch, replace=False)
        return {key: value[ix] for key, value in self.data.items()}

    def state_dict(self):
        return {'capacity': self.capacity, 'size': self.size, 'cursor': self.cursor,
                'data': {key: x[:self.size].copy() for key, x in self.data.items()}}

    def load(self, state):
        if state['capacity'] != self.capacity:
            raise ValueError('Resume replay capacity differs')
        self.size, self.cursor = state['size'], state['cursor']
        for key, value in state['data'].items():
            self.data[key][:self.size] = value


class StopAtEpisode:
    def __init__(self):
        self.requested = False
        self.previous = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.signal(sig, lambda *_: setattr(self, 'requested', True))
        return self

    def __exit__(self, *args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


class Recorder:
    def __init__(self, directory, config, seed, stage, protocol):
        self.path = Path(directory); self.path.mkdir(parents=True, exist_ok=True)
        self.config, self.seed, self.stage = config, seed, stage
        self.meta = {'config': dataclasses.asdict(config), 'seed': seed, 'stage': stage,
                     'protocol_sha256': protocol, 'device': 'cpu',
                     'resume': 'exact at saved episode boundary; latest includes bounded ring replay'}
        meta_path = self.path / 'run.json'
        if meta_path.exists() and json.loads(meta_path.read_text()) != self.meta:
            raise RuntimeError('Existing run identity differs; use a new run directory')
        meta_path.write_text(canonical_json(self.meta)+'\n')
        self.start = time.monotonic()

    def log(self, **values):
        values.update(stage=self.stage, wall_seconds=time.monotonic()-self.start)
        with (self.path / 'progress.jsonl').open('a') as f:
            f.write(canonical_json(values)+'\n')
        print(canonical_json(values), flush=True)

    def save(self, name, payload):
        final = self.path / (name+'.pt'); temp = self.path / (name+'.pt.tmp')
        torch.save({'identity': self.meta, **payload}, temp)
        os.replace(temp, final)

    def load_latest(self):
        p = self.path / 'latest.pt'
        if not p.exists():
            return None
        state = torch.load(p, map_location='cpu', weights_only=False)
        if state['identity'] != self.meta:
            raise RuntimeError('Resume identity mismatch')
        return state


def _rng_state(rng):
    return {'numpy': rng.get_state(), 'torch': torch.get_rng_state()}


def _restore_rng(rng, state):
    rng.set_state(state['numpy']); torch.set_rng_state(state['torch'])


def _frozen_input_hashes(sources=None, sf_cache=None):
    return {'source_hash': None if sources is None else hashlib.sha256(sources.tobytes()).hexdigest(),
            'sf_cache_hash': None if sf_cache is None else hashlib.sha256(sf_cache.tobytes()).hexdigest()}


def _check_frozen_inputs(payload, sources=None, sf_cache=None):
    expected = _frozen_input_hashes(sources, sf_cache)
    if any(key not in payload or payload[key] != value for key, value in expected.items()):
        raise RuntimeError('Checkpoint is bound to different frozen source/SF input artifacts')



def q_inputs(role, states, times, weights, sources=None, sf_cache=None):
    if role != 'direct' or sources is not None or sf_cache is not None:
        raise ValueError('This module exposes only the unconstrained direct scratch policy')
    obs = observation(states, times)
    return tensor(np.concatenate([obs, weights.reshape(-1, 8)], -1)), None


def q_values(model, role, states, times, weights, sources=None, sf_cache=None):
    x, base = q_inputs(role, states, times, weights, sources, sf_cache)
    out = model(x)
    return out if base is None else out+base



def policy_from_model(model, role, world, weights, sources=None, sf_cache=None, **_):
    policy = np.empty((world.horizon, world.S), dtype=np.int64)
    states = np.arange(world.S)
    w = np.broadcast_to(weights, (world.S, 2, 4))
    with torch.no_grad():
        for t in range(world.horizon):
            q = q_values(model, role, states, np.full(world.S, t), w, sources, sf_cache)
            policy[t] = q.argmax(-1).numpy()
    return policy


def evaluate_model(model, role, world, tasks, sources=None, sf_cache=None):
    before = model_hash(model)
    rows = []
    for task in tasks:
        w = np.asarray(task['weights'], dtype=np.float32)
        policy = policy_from_model(model, role, world, w, sources, sf_cache)
        rows.append({'task_id': task['id'], 'kind': task['kind'],
                     'return': evaluate_policy(world, policy, w)['initial_return']})
    if before != model_hash(model):
        raise RuntimeError('Evaluation changed model parameters')
    return {'mean_return': float(np.mean([r['return'] for r in rows])), 'rows': rows,
            'model_hash_before': before, 'model_hash_after': model_hash(model),
            'evaluation_optimizer_updates': 0, 'normalization': 'fixed coordinate/time scaling'}


def _dqn_batch(model, target, role, batch, optimizer, cfg, sources, sf_cache):
    q = q_values(model, role, batch['state'], batch['time'], batch['weights'], sources, sf_cache)
    pred = q.gather(1, tensor(batch['action'], torch.long)[:, None]).squeeze(1)
    next_time = np.minimum(batch['time'].astype(int)+1, cfg.horizon-1)
    with torch.no_grad():
        next_online = q_values(model, role, batch['next_state'], next_time,
                               batch['weights'], sources, sf_cache)
        a = next_online.argmax(-1)
        next_target = q_values(target, role, batch['next_state'], next_time,
                               batch['weights'], sources, sf_cache)
        bootstrap = next_target.gather(1, a[:, None]).squeeze(1)
        y = tensor(batch['reward'])+cfg.gamma*(1-tensor(batch['done']))*bootstrap
    loss = nn.functional.smooth_l1_loss(pred, y)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); optimizer.step()
    return float(loss.detach())


def train_dqn(directory, cfg, seed, role, budget, tasks, validation_tasks,
              protocol_hash, sources=None, sf_cache=None, source_index=None):
    """Train the direct scratch policy. Never opens the test objective file."""
    if role != 'direct' or sources is not None or sf_cache is not None or source_index is not None:
        raise ValueError('Only the direct unconstrained scratch baseline is published here')
    configure_torch(cfg, seed)
    world = SFWorldExact(horizon=cfg.horizon, gamma=cfg.gamma, kappa=cfg.kappa, seed=seed)
    input_dim = 5 if role == 'source' else 13 if role == 'direct' else 29
    output_dim = 25 if role in ('source', 'direct') else 16
    model = MLP(input_dim, output_dim, cfg.hidden, zero_final=role in ('full', 'masked'))
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    replay = Replay(cfg.replay_capacity)
    rng = np.random.RandomState(seed+17)
    recorder = Recorder(directory, cfg, seed, role, protocol_hash)
    frozen_inputs = _frozen_input_hashes(sources, sf_cache)
    steps, updates, episodes, best, last_loss = 0, 0, 0, -float('inf'), None
    existing = recorder.load_latest()
    if existing:
        if existing['budget'] != budget or existing.get('source_index') != source_index:
            raise RuntimeError('Resume budget/source identity differs')
        _check_frozen_inputs(existing, sources, sf_cache)
        model.load_state_dict(existing['model']); target.load_state_dict(existing['target'])
        optimizer.load_state_dict(existing['optimizer']); replay.load(existing['replay'])
        steps, updates, episodes, best = (existing[k] for k in ('steps', 'updates', 'episodes', 'best'))
        _restore_rng(rng, existing['rng'])
        if existing.get('complete'):
            if not (Path(directory)/'final.pt').exists():
                recorder.save('final', {'model': model.state_dict(), 'steps': steps,
                    'updates': updates, 'rng': existing['rng'], 'complete': True,
                    'model_sha256': model_hash(model), **frozen_inputs})
            best_state = torch.load(Path(directory)/'best.pt', map_location='cpu', weights_only=False)
            _check_frozen_inputs(best_state, sources, sf_cache)
            model.load_state_dict(best_state['model'])
            return freeze(model), existing

    def payload(complete=False):
        return {'model': model.state_dict(), 'target': target.state_dict(),
                'optimizer': optimizer.state_dict(), 'replay': replay.state_dict(),
                'rng': _rng_state(rng), 'steps': steps, 'updates': updates,
                'episodes': episodes, 'best': best, 'budget': budget,
                'source_index': source_index, 'complete': complete,
                **frozen_inputs}

    with StopAtEpisode() as stop:
        while steps < budget:
            task = tasks[int(rng.randint(len(tasks)))]
            w = np.asarray(task['weights'], dtype=np.float32)
            state = world.reset(rng=rng)
            for t in range(cfg.horizon):
                eps = 1-(1-cfg.epsilon_final)*min(1., steps/(budget*cfg.epsilon_decay_fraction))
                if rng.rand() < eps:
                    selected = int(rng.randint(output_dim))
                else:
                    with torch.no_grad():
                        selected = int(q_values(model, role, np.array([state]), np.array([t]),
                            w[None], sources, sf_cache).argmax(-1)[0])
                if role in ('source', 'direct'):
                    action = selected
                else:
                    a0 = int(sources[selected//4, t, state, 0])
                    a1 = int(sources[selected % 4, t, state, 1])
                    action = a0*5+a1
                    actual = world.joint_actions[action]
                    expected = np.array([sources[selected//4, t, state, 0],
                                         sources[selected % 4, t, state, 1]])
                    if not np.array_equal(actual, expected):
                        raise RuntimeError('Candidate membership violation')
                nxt, phi, _ = world.step(state, action, rng=rng)
                done = t+1 == cfg.horizon
                replay.append(state=state, time=t, next_state=nxt, action=selected,
                              reward=world.reward(phi, w), done=done, weights=w, features=phi)
                steps += 1
                if replay.size >= max(cfg.warmup, cfg.batch) and steps % cfg.update_every == 0:
                    last_loss = _dqn_batch(model, target, role, replay.sample(rng, cfg.batch),
                                           optimizer, cfg, sources, sf_cache)
                    updates += 1
                if steps % cfg.target_every == 0:
                    target.load_state_dict(model.state_dict())
                state = nxt
            episodes += 1
            should_val = steps % cfg.validation_every == 0 or steps == budget
            if should_val:
                result = evaluate_model(model, role, world, validation_tasks, sources, sf_cache)
                score = result['mean_return']
                if score > best:
                    best = score
                    recorder.save('best', {'model': model.state_dict(), 'steps': steps,
                                          'updates': updates, 'validation': result,
                                          'source_index': source_index,
                                          'model_sha256': model_hash(model), **frozen_inputs})
                recorder.log(steps=steps, updates=updates, episodes=episodes,
                             loss=last_loss, epsilon=eps, validation=result)
            elif steps % 1000 == 0:
                recorder.log(steps=steps, updates=updates, episodes=episodes, loss=last_loss, epsilon=eps)
            if steps % cfg.checkpoint_every == 0 or steps == budget or stop.requested:
                recorder.save('latest', payload(complete=steps == budget))
            if stop.requested and steps < budget:
                recorder.log(status='interrupted_at_episode_boundary', steps=steps, updates=updates)
                return None, payload()
    recorder.save('final', {'model': model.state_dict(), 'steps': steps, 'updates': updates,
                            'rng': _rng_state(rng), 'complete': True,
                            'model_sha256': model_hash(model), **frozen_inputs})
    best_state = torch.load(Path(directory)/'best.pt', map_location='cpu', weights_only=False)
    _check_frozen_inputs(best_state, sources, sf_cache)
    model.load_state_dict(best_state['model'])
    return freeze(model), {'steps': steps, 'updates': updates, 'episodes': episodes,
                           'best_validation': best, 'complete': True, 'model_hash': model_hash(model)}



@dataclasses.dataclass
class SweepConfig(LearningConfig):
    """LearningConfig with the audited kappa sweep enabled."""

    def validate(self):
        if self.horizon != 40 or self.gamma != .95 or not 0. <= self.kappa <= 1.:
            raise ValueError('Retrain diagnostic requires H=40, gamma=.95, 0<=kappa<=1')
        if not 0 < self.replay_capacity <= 50000:
            raise ValueError('Replay capacity must be bounded by 50000')
        for key in ('source_steps', 'sf_steps', 'composer_steps', 'no_sf_steps', 'direct_steps'):
            if getattr(self, key) <= 0 or getattr(self, key) % self.horizon:
                raise ValueError(f'{key} must be a positive multiple of H=40')
        if not 1 <= self.threads <= 2 or self.batch > self.replay_capacity:
            raise ValueError('Retrain diagnostic requires one or two CPU threads')
        for key in ('hidden', 'batch', 'warmup', 'update_every', 'target_every',
                    'checkpoint_every', 'validation_every'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if self.checkpoint_every % self.horizon or self.validation_every % self.horizon:
            raise ValueError('Checkpoint and validation intervals must align with episodes')
        if not 0 < self.epsilon_final <= 1 or not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError('Invalid epsilon schedule')
        guard = os.environ.get('MA_USFA_THREAD_BUDGET')
        if guard is None or int(guard) < self.threads:
            raise ValueError('Run retraining through the declared thread budget')


