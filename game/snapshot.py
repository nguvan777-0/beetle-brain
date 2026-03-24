"""Save and load world snapshots to/from disk."""
import os
import numpy as np
from sim.config import N_HIDDEN
from sim.population.genome import decode

SNAPSHOT_PATH = "snapshot.npz"


def save_snapshot(pop, food, tick, history, hall_fame):
    hist_arr = np.array(history, dtype=np.float32) if history else np.empty((0, 7), dtype=np.float32)
    np.savez_compressed(SNAPSHOT_PATH,
        x=pop['x'], y=pop['y'], angle=pop['angle'], energy=pop['energy'],
        W_body=pop['W_body'], W1=pop['W1'], W2=pop['W2'],
        h_state=pop['h_state'],
        generation=pop['generation'], age=pop['age'], eaten=pop['eaten'],
        food=food, tick=np.array([tick], dtype=np.int32),
        hist=hist_arr)
    print(f"[saved] {len(pop['x'])} organisms → {SNAPSHOT_PATH}  (tick {tick})")


def load_snapshot(rng):
    if not os.path.exists(SNAPSHOT_PATH):
        return None, None, 0, [], []
    d      = np.load(SNAPSHOT_PATH, allow_pickle=True)
    W_body = d['W_body'].astype(np.float32)
    t      = decode(W_body)
    pop = {
        'x':          d['x'].astype(np.float32),
        'y':          d['y'].astype(np.float32),
        'angle':      d['angle'].astype(np.float32),
        'energy':     d['energy'].astype(np.float32),
        'W_body':     W_body,
        'W1':         d['W1'].astype(np.float32),
        'W2':         d['W2'].astype(np.float32),
        **t,
        'generation': d['generation'].astype(np.int32),
        'age':        d['age'].astype(np.int32),
        'eaten':      d['eaten'].astype(np.int32),
        'h_state':    (d['h_state'].astype(np.float32) if 'h_state' in d
                       else np.zeros((len(d['x']), N_HIDDEN), dtype=np.float32)),
    }
    history = [tuple(row) for row in d['hist']] if d['hist'].ndim == 2 and len(d['hist']) else []
    print(f"[loaded] {len(pop['x'])} organisms ← {SNAPSHOT_PATH}  (tick {int(d['tick'][0])})")
    return pop, d['food'], int(d['tick'][0]), history, []
