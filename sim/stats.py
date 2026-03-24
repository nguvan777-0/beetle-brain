"""Stats collector: samples population state every N ticks for post-run reporting."""
import numpy as np
from sim.config import DRAIN_SCALE


SAMPLE_EVERY = 500


class StatsCollector:
    def __init__(self):
        self.samples   = []   # list of dicts, one per sample
        self.hall_fame = {
            'longest':  None,   # max age
            'killer':   None,   # max eaten
            'eldest':   None,   # max generation
        }
        self.run_meta  = {}   # filled by finalize()

    # ── sampling ─────────────────────────────────────────────────────────────

    def record(self, tick, pop):
        """Call every SAMPLE_EVERY ticks."""
        N = len(pop['x'])
        if N == 0:
            return

        drain   = DRAIN_SCALE * pop['size'] ** 0.75
        sensing = pop['ray_len'] * pop['fov']

        self.samples.append({
            'tick':             tick,
            'pop':              N,
            'max_gen':          int(pop['generation'].max()),
            'max_age':          int(pop['age'].max()),
            'max_eaten':        int(pop['eaten'].max()),
            'size_mean':        float(pop['size'].mean()),
            'size_min':         float(pop['size'].min()),
            'size_max':         float(pop['size'].max()),
            'speed_mean':       float(pop['speed'].mean()),
            'fov_mean':         float(np.degrees(pop['fov'].mean())),
            'fov_min':          float(np.degrees(pop['fov'].min())),
            'ray_mean':         float(pop['ray_len'].mean()),
            'ray_min':          float(pop['ray_len'].min()),
            'pred_ratio_mean':  float(pop['pred_ratio'].mean()),
            'mutation_mean':    float(pop['mutation_rate'].mean()),
            'hgt_eat_mean':     float(pop['hgt_eat_rate'].mean()),
            'hgt_contact_mean': float(pop['hgt_contact_rate'].mean()),
            'drain_mean':       float(drain.mean()),
            'sensing_mean':     float(sensing.mean()),
        })

        self._update_hall_fame(pop)

    def _update_hall_fame(self, pop):
        N = len(pop['x'])
        for key, idx_fn in [
            ('longest', lambda: int(pop['age'].argmax())),
            ('killer',  lambda: int(pop['eaten'].argmax())),
            ('eldest',  lambda: int(pop['generation'].argmax())),
        ]:
            i = idx_fn()
            entry = self._snapshot_wight(pop, i)
            cur = self.hall_fame[key]
            if cur is None:
                self.hall_fame[key] = entry
            elif entry['value'] > cur['value']:
                self.hall_fame[key] = entry

    def _snapshot_wight(self, pop, i):
        return {
            'age':          int(pop['age'][i]),
            'eaten':        int(pop['eaten'][i]),
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
            'value':        max(int(pop['age'][i]), int(pop['eaten'][i]), int(pop['generation'][i])),
        }

    # ── finalize ─────────────────────────────────────────────────────────────

    def finalize(self, tick, elapsed, pop=None, extinct=False):
        """Call once at end of run."""
        self.run_meta = {
            'ticks':      tick,
            'elapsed':    elapsed,
            'tps':        tick / elapsed if elapsed > 0 else 0,
            'extinct':    extinct,
            'final_pop':  len(pop['x']) if pop is not None and not extinct else 0,
            'final_max_gen': int(pop['generation'].max()) if pop is not None and not extinct else 0,
        }
        if pop is not None and not extinct:
            self.record(tick, pop)
