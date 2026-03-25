# growing a brain from the ground up

A wight is its weights. The weights are the organism — body and brain, all 222 floats.
Mutate them, cross them over, let selection filter. No fitness function, no reward
signal, no loss. Just survival.

We already did the first part. Wights go from aimless to hunters. That's proven.

---

## what we've proven

Start: 12 random wights, random RNN weights, random movement.

The brain is an Elman RNN — no bias, no architecture tricks:

```python
h_t   = tanh(x_t @ W1 + h_{t-1})   # (N_HIDDEN,) = (N_INPUTS,) @ (N_INPUTS, N_HIDDEN)
out_t = tanh(h_t @ W2)              # (N_OUTPUTS,) — turn and speed
```

222 floats total. 18 body, 180 for W1, 24 for W2. That's the whole organism.

Within a few thousand ticks, some wights are tracking and killing prey. W1 and W2
evolved to make hunting the output of that recurrence. The hidden state is carrying
something real across ticks. You can watch it happen.

---

## what follows from a working substrate

**Niche partitioning.** Competing lineages split the resource space to avoid direct
competition — different prey sizes, different vent territories, different hunting
strategies. A 12-dimensional h_state can only encode one strategy well. Once two
predator lineages both evolve pursuit, selection pushes them apart. Character
displacement — not designed, selected.

**Cognitive speciation.** HGT transplants brain weights across lineages. But as W1/W2
diverge, transplants become incoherent — a recipient brain can't use donor output
weights built for a different input encoding. At some divergence threshold, gene flow
ceases. Speciation from cognitive incompatibility, not geography.

**The Red Queen.** Prey evolve evasion circuits. Predators evolve better pursuit
circuits. Each improvement degrades the other's fitness — continuous co-evolution with
no stable endpoint. A predator can absorb the prey's evasion circuit via HGT and use
it offensively.

**The Baldwin effect.** Behaviors carried in h_state via epigenetic inheritance can
become hardwired in W1/W2 — the RNN learns to produce the behavior without needing the
inherited state. `epigenetic` should evolve down when that happens. Learned → instinct,
visible in the genome.

---

## what h_state actually encodes

Selection is sculpting W1/W2 to read and write h_state usefully. 12 floats per wight
— opaque by default. The probe:

```python
# does any h_state dimension correlate with bearing to nearest vent?
bearing = np.arctan2(vents[:, 1] - pop['y'][:, None],
                     vents[:, 0] - pop['x'][:, None]).min(axis=1)
np.corrcoef(pop['h_state'].T, bearing)  # (12, N) vs (N,)
```

If a dimension correlates with vent bearing at gen 100 but not gen 10, the brain
learned spatial memory. That's measurable.

---

## HGT as the propagation mechanism

```python
g = np.concatenate([W_body, W1.flatten(), W2.flatten()])  # (222,)
cut = rng.integers(1, 222)
g_new = np.where(np.arange(222) >= cut, g_donor, g_recipient)
```

204 of 222 weights are brain. A predator that kills a wight with a better circuit
absorbs part of it immediately — not through reproduction, through the kill. Cognitive
adaptations spread horizontally across lineages as well as vertically through descent.

---

## can selection grow a larger brain?

The brain is fixed at N_HIDDEN=12. In biology, brain size is under selection — complex
environments grow large brains because the cognitive payoff outweighs the metabolic
cost. Make N_HIDDEN evolvable, penalize it metabolically, and watch whether selection
drives encephalization when the world demands it.

What happens when the environment varies — vents that move, food that cycles? What
happens when the world is complex enough to require multiple strategies — does the
population partition into cognitive niches, speciate around incompatible brain
encodings, run the Red Queen indefinitely?

We have a system where these questions have real answers.

Where do we go from here?
