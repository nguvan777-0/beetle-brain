"""
run_headless.py — run beetle-brain at max speed, print analysis
    uv run --with numpy python run_headless.py
"""
import numpy as np
import time
import sys

# ── copy constants + classes from world.py without pygame ─────────────────────
WIDTH, HEIGHT   = 900, 900
N_FOOD          = 200
N_START         = 80
MAX_ORGANISMS   = 300
SPEED_MAX       = 4.0;  SPEED_MIN  = 0.3
FOV_MAX         = np.pi*0.9; FOV_MIN = np.pi*0.15
RAY_MAX         = 180.0; RAY_MIN   = 30.0
SIZE_MAX        = 9.0;   SIZE_MIN  = 3.0
DRAIN_MAX       = 0.18;  DRAIN_MIN = 0.02
ENERGY_START    = 120.0; ENERGY_MAX = 200.0
ENERGY_FOOD     = 55.0;  ENERGY_BREED = 160.0; ENERGY_CLONE = 80.0
N_RAYS          = 7
N_INPUTS        = N_RAYS * 2 + 1
N_HIDDEN        = 12
N_OUTPUTS       = 2
N_BODY          = 9
MUTATION_RATE   = 0.12
MUTATION_SCALE  = 0.15

def sig(x): return 1.0 / (1.0 + np.exp(-float(x)))

def decode_body(W):
    speed  = SPEED_MIN  + sig(W[0]) * (SPEED_MAX  - SPEED_MIN)
    fov    = FOV_MIN    + sig(W[1]) * (FOV_MAX    - FOV_MIN)
    ray    = RAY_MIN    + sig(W[2]) * (RAY_MAX    - RAY_MIN)
    size   = SIZE_MIN   + sig(W[3]) * (SIZE_MAX   - SIZE_MIN)
    drain  = DRAIN_MIN  + sig(W[4]) * (DRAIN_MAX  - DRAIN_MIN)
    turn   = 0.05       + sig(W[8]) * 0.25
    return speed, fov, ray, size, drain, turn

def make_genome(rng):
    return (rng.standard_normal(N_BODY).astype(np.float32),
            rng.standard_normal((N_INPUTS, N_HIDDEN)).astype(np.float32) * 0.8,
            rng.standard_normal((N_HIDDEN, N_OUTPUTS)).astype(np.float32) * 0.8)

def mutate(Wb, W1, W2):
    rng = np.random.default_rng()
    def _m(W):
        m = rng.random(W.shape) < MUTATION_RATE
        W = W.copy(); W[m] += rng.standard_normal(m.sum()).astype(np.float32) * MUTATION_SCALE
        return W
    return _m(Wb), _m(W1), _m(W2)

class Org:
    _id = 0
    def __init__(self, x, y, angle, Wb, W1, W2, energy=None, generation=0):
        self.x = float(x); self.y = float(y); self.angle = float(angle)
        self.Wb = Wb; self.W1 = W1; self.W2 = W2
        self.energy = energy or ENERGY_START
        self.generation = generation; self.age = 0; self.eaten = 0
        Org._id += 1; self.id = Org._id
        self.speed, self.fov, self.ray, self.size, self.drain, self.turn_s = decode_body(Wb)

    def sense(self, food, others):
        inp = np.zeros(N_INPUTS, dtype=np.float32)
        for i, a in enumerate(np.linspace(-self.fov/2, self.fov/2, N_RAYS) + self.angle):
            dx, dy = np.cos(a), np.sin(a)
            for j, (pos, rad) in enumerate([(food, 3.0), (others, self.size*2)]):
                best = 1.0
                if len(pos):
                    fx=pos[:,0]-self.x; fy=pos[:,1]-self.y
                    pr=fx*dx+fy*dy; pe=np.abs(fx*dy-fy*dx)
                    m=(pr>0)&(pr<self.ray)&(pe<rad)
                    if m.any(): best=float(pr[m].min())/self.ray
                inp[i*2+j] = 1.0 - best
        inp[-1] = self.energy / ENERGY_MAX
        return inp

    def tick(self, food, others):
        h = np.tanh(self.sense(food, others) @ self.W1)
        out = np.tanh(h @ self.W2)
        turn  = float(out[0]) * self.turn_s
        speed = (float(out[1])+1.0)*0.5*self.speed
        self.angle += turn
        self.x = (self.x + np.cos(self.angle)*speed) % WIDTH
        self.y = (self.y + np.sin(self.angle)*speed) % HEIGHT
        self.energy -= self.drain + speed*0.01 + self.size*0.002
        self.age += 1
        return speed

    def eat(self, food):
        if not len(food): return []
        d = np.sqrt((food[:,0]-self.x)**2+(food[:,1]-self.y)**2)
        hit = np.where(d < self.size+3.0)[0].tolist()
        if hit: self.energy = min(ENERGY_MAX, self.energy+ENERGY_FOOD*len(hit)); self.eaten+=len(hit)
        return hit

    def predate(self, others):
        for o in others:
            if o.size > self.size*0.75: continue
            if (o.x-self.x)**2+(o.y-self.y)**2 < (self.size+o.size)**2:
                self.energy = min(ENERGY_MAX, self.energy+o.energy*0.7); self.eaten+=1; return o
        return None

    def clone(self):
        if self.energy < ENERGY_BREED: return None
        self.energy = ENERGY_CLONE
        a = self.angle+np.pi+np.random.uniform(-0.5,0.5)
        Wb,W1,W2 = mutate(self.Wb,self.W1,self.W2)
        return Org(self.x+np.cos(a)*(self.size*2+2), self.y+np.sin(a)*(self.size*2+2),
                   a, Wb, W1, W2, ENERGY_CLONE, self.generation+1)

# ── SIMULATION ────────────────────────────────────────────────────────────────
TICKS   = 10000
REPORT  = 1000

rng = np.random.default_rng(42)
orgs = [Org(rng.uniform(0,WIDTH), rng.uniform(0,HEIGHT),
            rng.uniform(0,2*np.pi), *make_genome(rng)) for _ in range(N_START)]
food = rng.uniform(0,[WIDTH,HEIGHT],size=(N_FOOD,2)).astype(np.float32)

print(f"{'tick':>6} {'pop':>4} {'maxGen':>7} {'maxAte':>7} "
      f"{'avgSpd':>7} {'avgSz':>6} {'avgFov':>7} {'avgDrn':>7}")
print("─"*65)

t0 = time.time()

for tick in range(1, TICKS+1):
    new_orgs = []; eaten_ids = set()

    for org in orgs:
        if id(org) in eaten_ids: continue
        live = [o for o in orgs if o is not org and id(o) not in eaten_ids]
        others = np.array([[o.x,o.y] for o in live],dtype=np.float32) if live else np.empty((0,2))
        org.tick(food, others)
        for i in org.eat(food): eaten_ids.add(i)   # food indices handled below
        prey = org.predate(live)
        if prey: eaten_ids.add(id(prey))
        if len(orgs)+len(new_orgs) < MAX_ORGANISMS:
            c = org.clone()
            if c: new_orgs.append(c)

    # remove eaten food by index — need to track food indices separately
    food_eaten = set()
    for org in orgs:
        if id(org) in eaten_ids: continue
    # simpler: just re-run eat tracking via positions
    new_food_mask = np.ones(len(food), bool)
    for org in orgs:
        if id(org) in eaten_ids: continue
        d = np.sqrt((food[:,0]-org.x)**2+(food[:,1]-org.y)**2)
        new_food_mask &= d >= (org.size+3.0)
    food = food[new_food_mask]
    short = N_FOOD - len(food)
    if short > 0:
        food = np.vstack([food, rng.uniform(0,[WIDTH,HEIGHT],size=(short,2)).astype(np.float32)]) if len(food) else rng.uniform(0,[WIDTH,HEIGHT],size=(N_FOOD,2)).astype(np.float32)

    orgs = [o for o in orgs if o.energy > 0 and id(o) not in eaten_ids] + new_orgs
    if len(orgs) < 10:
        for _ in range(20):
            orgs.append(Org(rng.uniform(0,WIDTH),rng.uniform(0,HEIGHT),
                            rng.uniform(0,2*np.pi),*make_genome(rng)))

    if tick % REPORT == 0 and orgs:
        mg  = max(o.generation for o in orgs)
        ma  = max(o.eaten      for o in orgs)
        asp = np.mean([o.speed for o in orgs])
        asz = np.mean([o.size  for o in orgs])
        afv = np.degrees(np.mean([o.fov for o in orgs]))
        adr = np.mean([o.drain for o in orgs])
        elapsed = time.time()-t0
        print(f"{tick:6d} {len(orgs):4d} {mg:7d} {ma:7d} "
              f"{asp:7.2f} {asz:6.1f} {afv:7.1f} {adr:7.3f}  ({elapsed:.1f}s)")

print()
print("TOP 5 SURVIVORS:")
for o in sorted(orgs, key=lambda o:-o.eaten)[:5]:
    print(f"  gen={o.generation:4d}  age={o.age:6d}  eaten={o.eaten:4d}  "
          f"speed={o.speed:.2f}  size={o.size:.1f}  "
          f"fov={np.degrees(o.fov):.0f}°  drain={o.drain:.3f}")
