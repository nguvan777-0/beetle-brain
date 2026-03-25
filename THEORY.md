# Growing a brain from the ground up

The wight is defined by its weights. Body and brain together are ~2,294 floats — mutated, crossed over, and filtered purely by selection. No fitness function, no reward signal, no loss calculation. Only survival.

The first phase is complete: wights reliably evolve from aimless movement to active hunting. A coastline can split a population. Grazers emerge. Predators emerge. The brain shrinks when the environment lets it and grows when the environment demands it.

The brain is an Elman RNN — hidden state carries across ticks, two outputs, turn and speed.

```
h_new = tanh(x @ W1 + h_prev @ Wh + b1)
out   = tanh(h_new @ W2 + b2)
```

`x` is the sensory input: up to 7 rays × 5 channels (food distance, organism distance, r/g/b) plus energy — 36 floats. `h_prev` is whatever the wight was thinking last tick. `W1`, `W2`, `Wh`, `b1`, `b2` all evolve. The recurrent connection is the memory.

---

## What we've seen

Size tips fast. FOV narrows. Grazers lose vision when food is free. A coastline can split a population — one lineage predatory with a full brain, one slow and nearly mindless running on sunlight.

The brain shrinks when the environment lets it. It grows when the environment demands it.

---

## The brain grows when it has to

`active_neurons` is an evolvable gene — it sets how many of the 32 hidden neurons are live each tick. A wight with `active_neurons=4` runs a tiny brain; one with `active_neurons=32` runs the full RNN. The cost scales superlinearly with brain size:

```python
drain = DRAIN_SCALE * pop['size'] ** 0.75   # Kleiber's law
pop['energy'] -= (drain
                  + speeds**2               * SPEED_TAX
                  + np.abs(turns) * pop['size'] * TURN_TAX
                  + pop['size']**2          * SIZE_TAX
                  + pop['n_rays'] * pop['ray_len'] * pop['fov'] * SENSING_TAX
                  + pop['active_neurons']**1.5  * BRAIN_TAX)
```

Sight, motion, size, turning, thought — all on the same energy budget. A grazer parked near a vent needs almost nothing. A predator tracking prey across a coastline needs more. Every neuron has to earn its place.

---

## Boundaries

Energy flows at edges. Coastlines, thermal vents, sunlight gradients — anywhere two regimes meet, selection sharpens. We place those edges deliberately. The steeper the gradient, the faster things diverge.

---

## HGT

On each kill, the predator has a chance to absorb a slice of the victim's genome — brain weights included. `hgt_eat_rate` is an evolvable gene ranging from 0.5–15% per kill:

```python
g_r = np.concatenate([wb_r, w1_r.reshape(n, -1), w2_r.reshape(n, -1),
                       wh_r.reshape(n, -1), b1_r, b2_r], axis=1)
cuts = rng.integers(1, L, size=n)
mask = np.arange(L)[None, :] >= cuts[:, None]   # True → take from donor
g_new = np.where(mask, g_d, g_r)
```

A random cut point splits both genomes. The predator keeps everything before the cut and takes the prey's genome from cut to end. Useful circuits spread sideways through the population, not just down through offspring.

Early in a run this pulls everything toward the winner. Later, as lineages diverge, transplanted weights stop making sense in a foreign brain. Gene flow stops. The speciation barrier isn't geography — it's cognitive incompatibility.

---

## Evolutionary scaling of brain size

The current ceiling is 32 hidden neurons. That's a config value, not a law.

`active_neurons` is already evolvable — selection sets the size, not us. The cost curve is superlinear (`active_neurons^1.5`), so every neuron has to earn its place. A world that doesn't demand complexity gets brains that shrink to zero. A world that does gets brains that grow until the energy runs out.

The bet is that if the world is complex and diverse enough — more terrain, boundaries, efficient energy flow, moving gradients, prey that evade, predators that pursue — the brain scales on its own. We design the pressure, not the architecture.

---

## What we're watching

- The hidden state saturates at ±1 early. What is it encoding? When does it start encoding something we'd recognize?
- When does a wight stop reacting to other wights and start predicting them?
- Bigger world, more terrain, seasonal vents, moving coastlines, a dozen ways to capture energy — what grows when the pressure never stops?
