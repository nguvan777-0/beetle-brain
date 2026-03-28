"""Stats collector: samples population state every N ticks for post-run reporting."""
import numpy as np
from sim.config import (
    DRAIN_SCALE, SPEED_TAX, SIZE_TAX, SENSING_TAX, BRAIN_TAX,
    SPEED_MIN, SPEED_MAX, FOV_MIN, FOV_MAX, RAY_MIN, RAY_MAX,
    SIZE_MIN, SIZE_MAX, MOUTH_MIN, MOUTH_MAX,
    PRED_RATIO_MIN, PRED_RATIO_MAX,
    MUTATION_RATE_MIN, MUTATION_RATE_MAX,
    MUTATION_SCALE_MIN, MUTATION_SCALE_MAX,
    EPIGENETIC_MIN, EPIGENETIC_MAX,
    WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX,
    HGT_EAT_MIN, HGT_EAT_MAX, HGT_CONTACT_MIN, HGT_CONTACT_MAX,
    BREED_AT_MIN, BREED_AT_MAX, CLONE_WITH_MIN, CLONE_WITH_MAX,
    N_RAYS, N_HIDDEN,
)
from sim import phylo as _phylo

SAMPLE_EVERY = 500

# gene metadata: (key_in_pop, display_name, min, max)
GENES = [
    ('speed',            'speed',        SPEED_MIN,        SPEED_MAX),
    ('fov',              'fov',          FOV_MIN,          FOV_MAX),
    ('ray_len',          'ray_len',      RAY_MIN,          RAY_MAX),
    ('size',             'size',         SIZE_MIN,         SIZE_MAX),
    ('r',                'r',            40,               255),
    ('g',                'g',            40,               255),
    ('b',                'b',            40,               255),
    ('turn_s',           'turn_s',       0.05,             0.30),
    ('breed_at',         'breed_at',     BREED_AT_MIN,     BREED_AT_MAX),
    ('clone_with',       'clone_with',   CLONE_WITH_MIN,   CLONE_WITH_MAX),
    ('mutation_rate',    'mut_rate',     MUTATION_RATE_MIN,MUTATION_RATE_MAX),
    ('mutation_scale',   'mut_scale',    MUTATION_SCALE_MIN,MUTATION_SCALE_MAX),
    ('epigenetic',       'epigenetic',   EPIGENETIC_MIN,   EPIGENETIC_MAX),
    ('weight_decay',     'wt_decay',     WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX),
    ('mouth',            'mouth',        MOUTH_MIN,        MOUTH_MAX),
    ('pred_ratio',       'pred_ratio',   PRED_RATIO_MIN,   PRED_RATIO_MAX),
    ('hgt_eat_rate',     'hgt_eat',      HGT_EAT_MIN,      HGT_EAT_MAX),
    ('hgt_contact_rate', 'hgt_contact',  HGT_CONTACT_MIN,  HGT_CONTACT_MAX),
    ('n_rays',           'n_rays',       0,                N_RAYS),
    ('active_neurons',   'active_neur',  0,                N_HIDDEN),
]
GENE_NAMES = [g[1] for g in GENES]


class StatsCollector:
    def __init__(self):
        self.samples    = []   # one dict per sample tick
        self.hall_fame  = {
            'longest': None,   # max age
            'hunter':  None,   # max hunts (predation)
            'grazer':  None,   # max grazed (food)
            'eldest':  None,   # max generation
        }
        self.run_meta    = {}
        self._tick_start = None   # set on first record; all ticks stored relative to this
        # lineage river: {ancestor_id: [(tick, count), ...]}
        self._lineage_series     = {}
        self._lineage_hues       = {}
        self._lineage_first_tick = {}
        self._lineage_parent_map = {}   # {child_uid: parent_uid}

    # ── sampling ─────────────────────────────────────────────────────────────

    def record(self, tick, pop, phylo_state=None):
        N = len(pop['x'])
        if N == 0:
            return
        if self._tick_start is None:
            self._tick_start = tick
        tick = tick - self._tick_start

        # ── drain components ──────────────────────────────────────────────────
        speeds    = pop.get('speed', np.zeros(N))
        kleiber   = DRAIN_SCALE * pop['size'] ** 0.75
        spd_tax   = speeds ** 2 * SPEED_TAX
        sz_tax    = pop['size'] ** 2 * SIZE_TAX
        sen_tax   = pop['ray_len'] * pop['fov'] * SENSING_TAX
        an        = pop.get('active_neurons', np.zeros(N))
        brn_tax   = an ** 1.5 * BRAIN_TAX

        # ── normalized genome heatmap row ──────────────────────────────────────
        gene_row = []
        for key, _, lo, hi in GENES:
            vals = pop.get(key)
            if vals is None:
                gene_row.append(0.5)
                continue
            span = hi - lo
            gene_row.append(float((vals.mean() - lo) / span) if span > 0 else 0.5)

        # ── lineage counts ────────────────────────────────────────────────────
        lineage_counts = {}
        if phylo_state is not None:
            depth   = max(4, int(pop['generation'].max()) // 3)
            anc_ids = _phylo.ancestor_at(pop['individual_id'], depth, phylo_state)
            uids, counts = np.unique(anc_ids, return_counts=True)
            for uid, cnt in zip(uids.tolist(), counts.tolist()):
                lineage_counts[uid] = cnt
                if uid not in self._lineage_hues:
                    self._lineage_hues[uid]       = float(phylo_state['hue'][uid % _phylo.M])
                    self._lineage_first_tick[uid]  = tick
                    # find parent lineage: ancestor of uid's immediate parent at same depth
                    imm_parent = int(phylo_state['parent'][uid % _phylo.M])
                    if imm_parent >= 0:
                        plid = int(_phylo.ancestor_at(np.array([imm_parent], dtype=np.int32), depth, phylo_state)[0])
                        self._lineage_parent_map[uid] = plid
            for uid, cnt in lineage_counts.items():
                self._lineage_series.setdefault(uid, []).append((tick, cnt))

        self.samples.append({
            'tick':             tick,
            'pop':              N,
            'max_gen':          int(pop['generation'].max()),
            'max_age':          int(pop['age'].max()),
            'max_grazed':       int(pop['grazed'].max()),
            'max_hunts':        int(pop['hunts'].max()),
            # per-trait means
            'size_mean':        float(pop['size'].mean()),
            'size_min':         float(pop['size'].min()),
            'size_max':         float(pop['size'].max()),
            'speed_mean':       float(speeds.mean()),
            'fov_mean_deg':     float(np.degrees(pop['fov'].mean())),
            'fov_min_deg':      float(np.degrees(pop['fov'].min())),
            'ray_mean':         float(pop['ray_len'].mean()),
            'ray_min':          float(pop['ray_len'].min()),
            'pred_ratio_mean':  float(pop['pred_ratio'].mean()),
            'mutation_mean':    float(pop['mutation_rate'].mean()),
            'hgt_eat_mean':     float(pop['hgt_eat_rate'].mean()),
            'hgt_contact_mean':     float(pop['hgt_contact_rate'].mean()),
            'n_rays_mean':          float(pop['n_rays'].mean()) if 'n_rays' in pop else 0.0,
            'n_rays_min':           float(pop['n_rays'].min())  if 'n_rays' in pop else 0.0,
            'active_neurons_mean':  float(an.mean()),
            # drain breakdown (per tick, per wight)
            'drain_kleiber':    float(kleiber.mean()),
            'drain_speed':      float(spd_tax.mean()),
            'drain_size':       float(sz_tax.mean()),
            'drain_sensing':    float(sen_tax.mean()),
            'drain_brain':      float(brn_tax.mean()),
            # genome heatmap row (20 values, each 0-1 normalized)
            'genes_norm':       gene_row,
            # final-snapshot data for scatter + per-lineage traits
            'size_all':         pop['size'].tolist(),
            'speed_all':        speeds.tolist(),
            'pred_ratio_all':   pop['pred_ratio'].tolist(),
            'n_rays_all':       pop['n_rays'].tolist() if 'n_rays' in pop else [0] * N,
            'active_neurons_all': an.tolist(),
            'lineage_hues_all': ([float(self._lineage_hues.get(
                                    int(_phylo.ancestor_at(np.array([iid]), max(4, int(pop['generation'].max()) // 3), phylo_state)[0]),
                                    0.0))
                                  for iid in pop['individual_id'].tolist()]
                                 if phylo_state is not None else [0.0] * N),
        })

        self._update_hall_fame(pop, phylo_state)

    def _update_hall_fame(self, pop, phylo_state):
        for key, idx_fn in [
            ('longest', lambda: int(pop['age'].argmax())),
            ('hunter',  lambda: int(pop['hunts'].argmax())),
            ('grazer',  lambda: int(pop['grazed'].argmax())),
            ('eldest',  lambda: int(pop['generation'].argmax())),
        ]:
            i = idx_fn()
            entry = self._snapshot_wight(pop, i, phylo_state)
            cur = self.hall_fame[key]
            if cur is None or entry['sort_val'] > cur['sort_val']:
                self.hall_fame[key] = entry

    def _snapshot_wight(self, pop, i, phylo_state):
        hue = 0.0
        if phylo_state is not None:
            depth = max(4, int(pop['generation'].max()) // 3)
            anc   = int(_phylo.ancestor_at(np.array([pop['individual_id'][i]]), depth, phylo_state)[0])
            hue   = float(phylo_state['hue'][anc % _phylo.M])
        return {
            'age':          int(pop['age'][i]),
            'grazed':       int(pop['grazed'][i]),
            'hunts':        int(pop['hunts'][i]),
            'generation':   int(pop['generation'][i]),
            'speed':        float(pop['speed'][i]),
            'size':         float(pop['size'][i]),
            'fov_deg':      float(np.degrees(pop['fov'][i])),
            'ray_len':      float(pop['ray_len'][i]),
            'pred_ratio':   float(pop['pred_ratio'][i]),
            'mutation_rate':float(pop['mutation_rate'][i]),
            'hgt_eat_rate': float(pop['hgt_eat_rate'][i]),
            'r':            int(pop['r'][i]),
            'g':            int(pop['g'][i]),
            'b':            int(pop['b'][i]),
            'lineage_hue':  hue,
            'sort_val':     max(int(pop['age'][i]), (int(pop['hunts'][i]) + int(pop['grazed'][i])) * 10, int(pop['generation'][i]) * 100),
        }

    # ── finalize ─────────────────────────────────────────────────────────────

    def finalize(self, tick, elapsed, pop=None, phylo_state=None, extinct=False, seed=None):
        if pop is not None and not extinct:
            self.record(tick, pop, phylo_state)
        session_ticks = tick - (self._tick_start or tick)
        self.run_meta = {
            'ticks':         session_ticks,
            'final_tick':    tick,
            'elapsed':       elapsed,
            'tps':           session_ticks / elapsed if elapsed > 0 else 0,
            'extinct':       extinct,
            'final_pop':     len(pop['x']) if pop is not None and not extinct else 0,
            'final_max_gen': int(pop['generation'].max()) if pop is not None and len(pop.get('generation', [])) > 0 else 0,
            'seed':          seed,
        }
