"""Predeclared objective splits; training loaders cannot open test by default."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ASSIGNMENTS = ((0, 3), (3, 0), (1, 2), (2, 1))


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def source_weights():
    out = np.full((4, 2, 4), .15, dtype=np.float32)
    for k, goals in enumerate(ASSIGNMENTS):
        for i, goal in enumerate(goals):
            out[k, i, goal] = 1.3
    return out


def reference_context(weights):
    """Own dominant harvest goal -> unique matching source component index."""
    w = np.asarray(weights)
    goals = w.argmax(-1)
    lookup = np.empty((2, 4), dtype=np.int64)
    for k, pair in enumerate(ASSIGNMENTS):
        for i, goal in enumerate(pair):
            lookup[i, goal] = k
    return np.stack([lookup[i, goals[..., i]] for i in range(2)], -1)


def _draw(rng, goals, kind):
    rows = rng.uniform(.1, .35, size=(2, 4))
    if kind == 'overlap':
        secondary = rng.choice([j for j in range(4) if j != goals[0]], 2, replace=False)
    else:
        secondary = [rng.choice([j for j in range(4) if j != goal]) for goal in goals]
    for i, goal in enumerate(goals):
        rows[i, goal] = rng.uniform(1.1, 1.4)
        rows[i, secondary[i]] = rng.uniform(.7, .95)
    if kind == 'homo':
        rows[1] = rows[0]
    return rows.astype(np.float32)


def create_protocol(directory, seed=11001):
    """Writes test weights once; later training only reads train/validation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'manifest.json').exists():
        return verify_protocol(directory)
    rng = np.random.RandomState(seed)
    pairs = list(itertools.permutations(range(4), 2))
    val_pairs = [pairs[i] for i in (0, 3, 6, 9, 1, 10)]
    specs = {
        'train': ([g for g in range(4) for _ in range(4)], pairs * 2,
                  [g for g in range(4) for _ in range(6)]),
        'validation': (list(range(4)), val_pairs, [0, 1, 2, 3, 0, 1]),
        'test': ([g for g in range(4) for _ in range(2)], pairs,
                 [g for g in range(4) for _ in range(3)]),
    }
    seen, files, counts = set(), {}, {}
    for split, (homo, distinct, overlap) in specs.items():
        entries = []
        for kind, goals_list in [('homo', [(g, g) for g in homo]),
                                 ('distinct', distinct),
                                 ('overlap', [(g, g) for g in overlap])]:
            for goals in goals_list:
                w = _draw(rng, goals, kind)
                key = w.tobytes()
                if key in seen:
                    raise RuntimeError('Duplicate objective in frozen split generation')
                seen.add(key)
                entries.append({'id': f'{split}_{kind}_{len(entries):03d}', 'kind': kind,
                                'weights': w.tolist(), 'reference': reference_context(w).tolist()})
        payload = {'split': split, 'split_seed': seed, 'objectives': entries}
        filename = 'test.sealed.json' if split == 'test' else split + '.json'
        (directory / filename).write_text(canonical_json(payload) + '\n')
        files[split] = {'file': filename, 'sha256': digest(payload)}
        counts[split] = {kind: sum(e['kind'] == kind for e in entries)
                         for kind in ('homo', 'distinct', 'overlap')}
    manifest = {'version': 'e2-v1', 'split_seed': seed, 'files': files, 'counts': counts,
                'source_weights': source_weights().tolist(), 'assignments': ASSIGNMENTS,
                'test_status': 'sealed_until_explicit_eval_test',
                'scope': 'unseen weights; all reference profiles covered in training',
                'mapping': 'own dominant harvest goal; first-index ties; fixed source lookup'}
    (directory / 'manifest.json').write_text(canonical_json(manifest) + '\n')
    return verify_protocol(directory)


def verify_protocol(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for info in manifest['files'].values():
        # Only fingerprint bytes here; never parse sealed test objective values.
        raw = (directory / info['file']).read_bytes().rstrip(b'\n')
        if hashlib.sha256(raw).hexdigest() != info['sha256']:
            raise RuntimeError('Frozen objective file fingerprint changed')
    # Validation of sealed bytes is not a training read of test objective values.
    return manifest


def load_tasks(directory, split, allow_test=False):
    if split == 'test' and not allow_test:
        raise PermissionError('Test objectives stay sealed; use explicit eval-test command')
    if split not in ('train', 'validation', 'test'):
        raise ValueError(split)
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    info = manifest['files'][split]
    payload = json.loads((directory / info['file']).read_text())
    if digest(payload) != info['sha256']:
        raise RuntimeError('Frozen split fingerprint mismatch')
    return payload['objectives']
