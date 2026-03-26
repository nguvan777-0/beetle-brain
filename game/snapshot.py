"""Save and load world snapshots to/from disk."""
import json
import os
import numpy as np
from sim.config import N_HIDDEN, N_INPUTS, N_OUTPUTS
from sim.population.genome import decode, N_BODY
from sim.vents import make_vents
from sim import phylo

SNAPSHOT_PATH = "snapshot.npz"


def save_snapshot(world, tick, history, hall_fame, stats=None):
    pop      = world['pop']
    food     = world['food']
    vents    = world['vents']
    hist_arr = np.array(history, dtype=np.float32) if history else np.empty((0, 7), dtype=np.float32)
    phylo_state = world['phylo']
    extra = {}
    if stats is not None:
        payload = {
            'samples':          stats.samples,
            'hall_fame':        stats.hall_fame,
            'run_meta':         stats.run_meta,
            '_lineage_series':  stats._lineage_series,
            '_lineage_hues':    stats._lineage_hues,
            '_lineage_first_tick': stats._lineage_first_tick,
            '_lineage_parent_map': stats._lineage_parent_map,
            '_tick_start':      stats._tick_start,
        }
        extra['stats_json'] = np.frombuffer(json.dumps(payload).encode(), dtype=np.uint8)
    np.savez_compressed(SNAPSHOT_PATH,
        x=pop['x'], y=pop['y'], angle=pop['angle'], energy=pop['energy'],
        W_body=pop['W_body'], W1=pop['W1'], W2=pop['W2'], Wh=pop['Wh'], b1=pop['b1'], b2=pop['b2'],
        h_state=pop['h_state'],
        generation=pop['generation'], age=pop['age'], eaten=pop['eaten'],
        lineage_id=pop['lineage_id'], individual_id=pop['individual_id'],
        food=food, vents=vents, tick=np.array([tick], dtype=np.int32),
        hist=hist_arr,
        phylo_parent=phylo_state['parent'],
        phylo_hue=phylo_state['hue'],
        phylo_next_id=np.array([phylo_state['next_id']], dtype=np.int32),
        seed=np.array([world.get('seed', 0)], dtype=np.int64),
        **extra)
    print(f"[saved] {len(pop['x'])} organisms → {SNAPSHOT_PATH}  (tick {tick})")


def _migrate_w1(w1):
    """Zero-pad W1 along the input axis if N_INPUTS grew since this snapshot was saved."""
    if w1.shape[1] < N_INPUTS:
        pad = np.zeros((w1.shape[0], N_INPUTS - w1.shape[1], w1.shape[2]), dtype=np.float32)
        w1 = np.concatenate([w1, pad], axis=1)
    return w1


def load_snapshot(rng):
    if not os.path.exists(SNAPSHOT_PATH):
        return None, 0, [], [], None
    d      = np.load(SNAPSHOT_PATH, allow_pickle=True)
    W_body = d['W_body'].astype(np.float32)
    t      = decode(W_body)
    n      = len(d['x'])
    pop = {
        'x':           d['x'].astype(np.float32),
        'y':           d['y'].astype(np.float32),
        'angle':       d['angle'].astype(np.float32),
        'energy':      d['energy'].astype(np.float32),
        'W_body':      W_body,
        'W1':          _migrate_w1(d['W1'].astype(np.float32)),
        'W2':          d['W2'].astype(np.float32),
        'Wh':          (d['Wh'].astype(np.float32) if 'Wh' in d
                        else np.zeros((n, N_HIDDEN, N_HIDDEN), dtype=np.float32)),
        'b1':          (d['b1'].astype(np.float32) if 'b1' in d
                        else np.zeros((n, N_HIDDEN),  dtype=np.float32)),
        'b2':          (d['b2'].astype(np.float32) if 'b2' in d
                        else np.zeros((n, N_OUTPUTS), dtype=np.float32)),
        **t,
        'generation':  d['generation'].astype(np.int32),
        'age':         d['age'].astype(np.int32),
        'eaten':       d['eaten'].astype(np.int32),
        'h_state':     (d['h_state'].astype(np.float32) if 'h_state' in d
                        else np.zeros((n, N_HIDDEN), dtype=np.float32)),
        'lineage_id':  (d['lineage_id'].astype(np.int32) if 'lineage_id' in d
                        else np.zeros(n, dtype=np.int32)),
        'individual_id': (d['individual_id'].astype(np.int32) if 'individual_id' in d
                          else np.arange(n, dtype=np.int32)),
    }
    vents = d['vents'].astype(np.float32) if 'vents' in d else make_vents()
    if 'phylo_parent' in d:
        phylo_state = {'parent':  d['phylo_parent'].astype(np.int32),
                       'hue':     d['phylo_hue'].astype(np.float32) if 'phylo_hue' in d else None,
                       'next_id': int(d['phylo_next_id'][0])}
        if phylo_state['hue'] is None:
            phylo_state['hue'] = np.zeros(phylo.M, dtype=np.float32)
    else:
        phylo_state = phylo.from_snapshot(pop['individual_id'])
    seed        = int(d['seed'][0]) if 'seed' in d else None
    world       = {'pop': pop, 'food': d['food'].astype(np.float32),
                   'vents': vents, 'phylo': phylo_state, 'seed': seed}
    history     = [tuple(row) for row in d['hist']] if d['hist'].ndim == 2 and len(d['hist']) else []
    stats = None
    if 'stats_json' in d:
        from sim.stats import StatsCollector
        payload = json.loads(d['stats_json'].tobytes().decode())
        stats = StatsCollector()
        stats.samples              = payload['samples']
        stats.hall_fame            = payload['hall_fame']
        stats.run_meta             = payload['run_meta']
        stats._lineage_series      = {int(k): v for k, v in payload['_lineage_series'].items()}
        stats._lineage_hues        = {int(k): v for k, v in payload['_lineage_hues'].items()}
        stats._lineage_first_tick  = {int(k): v for k, v in payload['_lineage_first_tick'].items()}
        stats._lineage_parent_map  = {int(k): v for k, v in payload['_lineage_parent_map'].items()}
        stats._tick_start          = payload['_tick_start']
    print(f"[loaded] {len(pop['x'])} organisms ← {SNAPSHOT_PATH}  (tick {int(d['tick'][0])}, seed {seed})")
    return world, int(d['tick'][0]), history, [], stats
