# Growing a brain from the ground up

The wight is defined entirely by its weights. Both body and brain consist of 222 floats. 
These values are mutated, crossed over, and filtered purely by selection. There is no fitness function, reward signal, or loss calculation—only survival.

The first phase is complete: wights reliably evolve from aimless movement to active hunting.

---

## Observed behaviors

Starting state: 12 random wights with randomized RNN weights and movement.

The brain uses a simple Elman RNN with no biases or architectural tricks:

```python
h_t   = np.tanh(x_t @ W1 + h_prev)  # (N_HIDDEN,) = (N_INPUTS,) @ (N_INPUTS, N_HIDDEN)
out_t = np.tanh(h_t @ W2)           # (N_OUTPUTS,) — turn and speed
```

The organism is parameterized by exactly 222 floats: 18 for the body, 180 for W1, and 24 for W2.

Within a few thousand ticks, tracking and hunting behaviors emerge. The weights W1 and W2 evolve to map inputs to pursuit outputs, utilizing the hidden state to retain information across ticks.

---

## Emergent dynamics

**Niche partitioning.** Competing lineages naturally subdivide the resource space to minimize direct competition, adopting different prey targets, vent territories, or hunting strategies. Because a 12-dimensional hidden state has limited capacity, selection drives competing predator lineages apart into distinct strategies (character displacement).

**Cognitive speciation.** Horizontal Gene Transfer (HGT) moves brain weights between lineages. As W1 and W2 diverge between lineages, these transplants become incoherent. A recipient brain cannot process donor output weights that expect a different input encoding. Once divergence reaches a critical threshold, gene flow stops, resulting in speciation driven by cognitive incompatibility rather than geography.

**Red Queen dynamics.** Prey evolve evasion circuits, prompting predators to evolve better pursuit circuits. Each adaptation decreases the opposing lineage's survival rate, driving continuous co-evolution. Predators can also acquire prey evasion circuits via HGT and repurpose them defensively or offensively.

**The Baldwin effect.** Behaviors initially maintained in the hidden state via epigenetic inheritance can become permanently hardwired into W1 and W2. When the RNN evolves to produce the behavior natively, reliance on inherited state (`epigenetic`) decreases. This shifts learned behavior into instinct encoded directly in the genome.

---

## Decoding the hidden state

Selection optimizes W1 and W2 to effectively read and write to the hidden state. While the 12 floats are opaque by default, they can be probed for correlations:

```python
# Check if any hidden state dimension correlates with the bearing to the nearest vent:
dx = vents[:, 0] - pop['x'][:, None]
dy = vents[:, 1] - pop['y'][:, None]

# Find the vent with the minimum distance squared
nearest_idx = np.argmin(dx**2 + dy**2, axis=0)

# Calculate bearing to that specific nearest vent
bearing = np.arctan2(dy[nearest_idx, np.arange(len(pop))], 
                     dx[nearest_idx, np.arange(len(pop))])

np.corrcoef(pop['h_state'].T, bearing)  # (N_HIDDEN, N) vs (N,)
```

Measuring these correlations across generations (e.g., gen 10 vs. gen 100) provides direct metrics for the development of spatial memory and other cognitive traits.

---

## HGT as a propagation mechanism

```python
g = np.concatenate([W_body, W1.flatten(), W2.flatten()])  # Entire genome
cut = rng.integers(1, len(g))
g_new = np.where(np.arange(len(g)) >= cut, g_donor, g_recipient)
```

Brain weights make up 204 of the 222 total parameters. When a predator kills another wight, it immediately absorbs portions of the victim's genome. This allows advantageous cognitive circuits to spread horizontally across the population via predation, rather than strictly vertically through reproduction.

---

## Evolutionary scaling of brain size

Currently, the brain is fixed at `N_HIDDEN=12`. In biological systems, brain size is subject to selection: complex environments favor larger brains because the cognitive advantages outweigh the metabolic costs.

By making `N_HIDDEN` evolvable and applying a metabolic penalty to it, the system can model encephalization. Future experiments can test whether dynamic environments (e.g., moving vents, cyclical food sources) or complex multi-agent interactions drive the population to evolve larger brain capacities, partition into cognitive niches, and sustain open-ended Red Queen co-evolution.
