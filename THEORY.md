# long-run analysis — where wights end up

You can't predict the exact path. Tiny mutations compound.
But you can predict where the attractors are.

---

## why exact prediction is impossible

The simulation is chaotic. Two populations that differ by one mutation bit
diverge completely within a few thousand ticks. This is not a fixable problem —
it's the mechanism. Evolution is a random walk biased by selection.

What you CAN predict: the stable states the walk converges to.

---

## ESS for size

An Evolutionarily Stable Strategy is a trait value no mutant can invade.

**The predation rule:** size_predator > 1.25 × size_prey

This means: in a population where everyone has size s*, a mutant with
size s > 1.25 × s* eats everyone and can't be eaten. A mutant with
s < s*/1.25 gets eaten by everyone.

There is no direct cost to being large. Drain is independently encoded
in W_body[:,4]. Size and drain are orthogonal weights.

**Therefore the ESS for size is SIZE_MAX = 9.0.** It's a corner solution.
The only thing that can stop it is frequency-dependent selection:
if everyone is max size, no one can eat each other, and small food-eaters
can survive. That's when you get stable size dimorphism — big predators
coexisting with small food-eaters. Natural ecosystems exactly.

**Why run 001 shows 7.8, not 9.0:**

The size weight w_s (W_body[:,3]) is pulled toward 0 by decay (δ = 0.00002)
and jittered by mutation (rate 0.12, scale 0.15). At neutral weight (w=0),
sigmoid(0) = 0.5, which maps to:

```
size_neutral = 3.0 + 0.5 × (9.0 - 3.0) = 6.0
```

The observed 7.8 corresponds to sigmoid⁻¹((7.8 - 3) / 6) = sigmoid⁻¹(0.8)
= ln(4) ≈ 1.39. Selection hasn't finished pushing the weight up yet.
7,891 ticks is early.

---

## weight decay as an Ornstein-Uhlenbeck process

Each weight follows this dynamic each tick:

```
w → w × (1 - δ)   +   mutation noise   +   selection gradient
```

Without selection, this is an OU process with stationary variance:

```
D  = p_mut × σ_mut² / 2  =  0.12 × 0.15² / 2  =  0.00135 per tick
σ² = D / δ               =  0.00135 / 0.00002  =  67.5
σ  = 8.22
```

sigmoid(8.22) ≈ 0.9997. The mutation noise alone covers nearly the entire
trait range. Decay is ~400× weaker than mutation at the scale of these weights.

**What this means:** weight decay barely constrains trait values. It keeps
the population from becoming perfectly fixed (which would end evolution),
but selection dominates the equilibrium position.

The weight decay half-life is:

```
t½ = ln(2) / δ = 0.693 / 0.00002 = 34,650 ticks
```

After ~35,000 ticks (≈ 2 minutes headless), the founding wight's original
weights have decayed to half. After ~170,000 ticks, they're at ~3%.
All genetic continuity with the first wight is gone well within an hour of
headless runtime.

---

## phase transition: when does size monoculture break?

There's a critical food density ρ* where food-eating outcompetes predation.
Below ρ*, predation wins → size → 9. Above ρ*, food-eating wins → size shrinks.

Rough estimate with current params (ray_len ≈ 90, fov ≈ 64° ≈ 1.12 rad,
speed ≈ 2.4, drain ≈ 0.08):

```
area swept per tick ≈ ray_len × fov × speed  =  90 × 1.12 × 2.4  ≈  242 units
food density  =  200 / (900 × 900)  ≈  0.000247 per unit²
food/tick     ≈  242 × 0.000247  ≈  0.06 pellets
energy/tick   ≈  0.06 × 55  -  0.08  ≈  +3.2 net
```

Food-eating alone is viable. This is why color could eventually diverge —
a small, fast, dim food-eater is a valid alternative strategy if the big
predators aren't scanning the right areas.

**To push the sim into size dimorphism:** reduce food_count below ~80.
Below that threshold, pure food-eating can't sustain a wight at drain=0.08,
and predation becomes mandatory. That creates runaway size selection, which
collapses diversity. Above ~80, there's room for small wights. The current
200 is well into the coexistence zone.

---

## FOV: where it's headed

We observed: 73° → 64° in 7,891 ticks. It's narrowing.

The ESS FOV depends on what the dominant energy source is. In a food-scarce
world, wide FOV catches more — you need the peripheral scan. In a predator
world, narrow FOV tracks prey more precisely — 7 rays concentrated forward
beats 7 rays spread over 162°.

As population density stays high (300 cap), predation is the primary food
source for large wights. Selection pressure: narrower FOV → better prey
tracking → more predation success.

Expected long-term floor: **~35–50°**. That's where a 7-ray system can
usefully resolve a target moving at the speed observed (2.4 units/tick) at
detection range (~90 units). Below that, the prey crosses between rays
faster than the wight can turn.

---

## color: the most uncertain prediction

Color is the hardest to predict long-term because it's a co-evolutionary
arms race. Both the camouflage strategy and the detection strategy
evolve simultaneously.

Current pressure: bright wights (high r+g+b) have a larger predation
detection radius (up to +9 units). This cuts both ways:
- Bright predators find prey from further away (good for predators)
- Bright prey are spotted from further away (bad for prey)

The stable states are:
1. **Uniform dim** — everyone dark, detection radius collapses to size alone
2. **Arms race to bright** — everyone bright, detection radius inflated for all,
   net effect neutral but energetically wasteful
3. **Dimorphism** — dim prey that evade detection + bright predators that hunt

Which one the sim lands in is path-dependent and unpredictable. It depends
which mutation hits first in which lineage.

---

## h_state: when does memory do something real?

The recurrent hidden state (N_HIDDEN=12) persists across ticks and is
partially inherited (25%). In run 001 it was nearly saturated (mean |h| = 0.70)
but the population was only 7 generations deep.

For h_state to encode something interpretable (fear, hunger, momentum),
the weights W1/W2 need to evolve to read and write it meaningfully.
That requires selection pressure on behaviors that span multiple ticks —
e.g., a wight that remembers which direction food was last found survives
longer than one that doesn't.

Rough estimate: this becomes visible at **gen 50–200**. Below that, h_state
is mostly noise that evolution hasn't had time to sculpt.

At current rates (~7 gen per 30s headless), gen 100 is ~7 minutes of runtime.

---

## long-run prediction summary

| trait       | current (run 001) | predicted long-run     | mechanism              |
|-------------|-------------------|------------------------|------------------------|
| size        | 7.8               | 8.5–9.0                | corner ESS, no cost    |
| FOV         | 64°               | 35–50°                 | predator tracking      |
| speed       | 2.4               | near current           | energy sweet spot      |
| drain       | 0.08              | near current           | energy sweet spot      |
| color       | neutral ~140      | unpredictable          | arms race              |
| h_state     | active, unsculpted| meaningful at gen 100+ | selection on memory    |
| dimorphism  | none              | possible if food > 80  | frequency-dependence   |

---

## what "1 billion ticks" actually means

At 263 ticks/sec headless, 1 billion ticks = ~44 days of runtime.

The founding wight's genome is irrelevant after ~170,000 ticks (11 minutes).
By 1 billion ticks, the population has thermalized completely — the only
surviving information is what selection kept. The weights will be distributed
around the ESS attractors above, jittered by mutation, with variance set by
the OU equilibrium.

You can't predict which individual wight wins. You can predict the distribution
they're drawn from.
