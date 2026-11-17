"""Shared observation, fingerprint and graceful-stop helpers.

These helpers are extracted unchanged from the audited synthetic runner.
"""
import hashlib
import signal
import numpy as np
def model_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        x = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(x.dtype).encode())
        h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()



def observation(states, times, horizon=40):
    s = np.asarray(states, dtype=np.int64)
    t = np.broadcast_to(np.asarray(times), s.shape)
    cells = np.stack([s // 25, s % 25], -1)
    coords = np.stack([cells[..., 0] // 5, cells[..., 0] % 5,
                       cells[..., 1] // 5, cells[..., 1] % 5], -1) / 4.
    return np.concatenate([coords, ((horizon-t)/horizon)[..., None]], -1).astype(np.float32)



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



