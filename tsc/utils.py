"""Current utilities; tripinfo is read only after its SUMO episode closes."""
from tsc_repaired.rollout_evaluate import parse_tripinfo
from tsc_repaired.config import sha256_file

def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
