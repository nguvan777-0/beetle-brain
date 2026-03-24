# long-run analysis — where wights end up

This is not a simulation of evolution as a metaphor. It's evolution running. The same
forces that shaped every living thing — selection, drift, mutation, horizontal gene
transfer — are operating here, on a genome you can read, at a speed you can watch.

The questions are real. Will sex emerge from HGT? Will predators evolve to eat their
prey's adaptations wholesale? Will memory crystallize in the RNN after enough generations?
Biology spent 4 billion years answering these. We get to watch them unfold in minutes.

You can't predict the exact path. Two populations that differ by one mutation bit
diverge completely within a few thousand ticks — that's not a bug, it's the mechanism.
Evolution is a random walk biased by selection.

But you can predict the attractors — the stable states the population gets pulled toward
and stays in, like a ball rolling into a valley.

---

## ESS for size

An Evolutionarily Stable Strategy is a trait value no mutant can invade.

**The predation rule:** size_predator > pred_ratio × size_prey (pred_ratio ∈ [1.05, 2.0])

In a monomorphic population at size s*, a mutant with size > pred_ratio × s* eats everyone.
A mutant with size < s*/pred_ratio gets eaten by everyone.

**Cost of being large:** drain = `0.015 × size^0.75` (Kleiber). But energy capacity
= `10 × size²`. Reserve time ∝ `size² / size^0.75 = size^1.25`. Larger wights have
proportionally *more* energy buffer. There is no size cost that outweighs the predation
advantage. **The ESS for size is SIZE_MAX = 9.0.** Corner solution.

The only brake is frequency-dependence: if everyone is max size, no one can eat each other,
and small food-eaters survive. That's size dimorphism — large predators coexisting with
small food-eaters. Natural ecosystems exactly.

**To force dimorphism:** reduce `food_count` below ~80. Below that, pure food-eating can't
sustain a wight, predation becomes mandatory, runaway size selection collapses diversity.
The default 100 is in the coexistence zone.

---

## neutral genes and drift

`weight_decay` carries no selection pressure — aging is metabolic, not genetic.
It drifts freely under mutation, unconstrained by fitness. Over long runs it distributes
uniformly across its range.

This is useful. A neutral gene is a clock. If `weight_decay` is flat after 100k ticks
but `mutation_rate` is sharply peaked, you're seeing selection in action on `mutation_rate`
against a clean drift baseline. Real population genetics uses neutral markers exactly
this way — microsatellites, synonymous substitutions. `weight_decay` is that here.

---

## ESS for HGT rates

`hgt_eat_rate` and `hgt_contact_rate` are genome genes. Their ESS depends on genetic
diversity in the population.

**Cost:** crossover with a random genome is usually destructive. A well-adapted wight
incorporating a stranger's W1/W2 likely breaks its foraging behavior. Cost is proportional
to genetic distance from the donor.

**Benefit:** HGT can spread beneficial mutations across lineage boundaries — faster than
waiting for the same mutation to arise twice independently.

In a genetically uniform population, HGT is pure noise. Both rates drift toward zero.
In stable predator-prey dimorphism, prey evolve novel evasion strategies. Predators that
incorporate prey genome via predation literally eat the prey's adaptations — Red Queen
dynamics without parasites. The predation arms race has a metabolic shortcut.

Expected long-run: `hgt_eat_rate` stabilizes somewhere in [0.01, 0.05] for carnivores.
`hgt_contact_rate` stays near the minimum unless dense clustering emerges around vents.

**The proto-sex question:** sex is HGT with mate choice. If `hgt_contact_rate` evolves
upward and wights cluster near genetically diverse populations to exchange genes, that's
proto-conjugation. Whether it emerges here is the experiment.

---

## phase transition: when does size monoculture break?

There's a critical food density ρ* where food-eating outcompetes predation.
Below ρ*, predation wins → size → 9. Above ρ*, food-eating wins → size shrinks.

Rough estimate with current params (ray_len ≈ 90, fov ≈ 90° ≈ 1.57 rad, speed ≈ 2.2):

```
area swept per tick ≈ ray_len × fov × speed  =  90 × 1.57 × 2.2  ≈  311 units
food density  =  100 / (700 × 700)  ≈  0.000204 per unit²
food/tick     ≈  311 × 0.000204  ≈  0.063 pellets
energy/tick   ≈  0.063 × 4  -  0.06  ≈  +0.19 net
```

Food-eating alone is barely viable. Current 100 food_count is close to the threshold —
the coexistence zone is narrow. This is why color divergence is possible: a small, dim
food-eater is a valid strategy if the big predators aren't scanning the right areas.

---

## FOV: where it's headed

Narrow FOV tracks a known target better. Wide FOV catches peripheral movement.

As predation dominates energy intake, selection pushes toward narrower FOV — 7 rays
concentrated forward beats 7 rays spread over 162° for tracking moving prey.

Expected long-term floor: **~35–50°**. Below that, prey moving at ~2 units/tick at
~90 unit detection range crosses between adjacent rays faster than the wight can turn.

In a food-dominated world (high food_count), wide FOV wins. Watch it toggle.

---

## color: the most uncertain prediction

Color is the hardest to predict long-term because it's a co-evolutionary arms race.
Both the camouflage strategy and the detection strategy evolve simultaneously.

Current pressure: bright wights (high r+g+b) have a larger predation detection radius
(up to +9 units). This cuts both ways — bright predators find prey from further away,
bright prey are spotted from further away.

The stable states are:
1. **Uniform dim** — everyone dark, detection radius collapses to size alone
2. **Arms race to bright** — everyone bright, detection symmetric, net effect neutral
3. **Dimorphism** — dim prey that evade detection + bright predators that hunt

Which one the sim lands in is path-dependent. It depends which mutation hits first
in which lineage. Now that founders start at their lineage hue, the initial color
distribution is a rainbow — which state it converges to from that start is the question.

---

## h_state: when does memory do something real?

The recurrent hidden state (N_HIDDEN=12) persists across ticks and is partially
inherited via `epigenetic` (∈ [0, 1]).

For h_state to encode something interpretable — hunger, momentum, fear — the weights
W1/W2 need to evolve to read and write it meaningfully. That requires selection on
behaviors that span multiple ticks: remembering which direction food was last found,
persisting a pursuit trajectory.

This is also where HGT is most interesting. Crossover at the W1/W2 boundary produces
a wight with one wight's input encoding and another's output mapping. Most are incoherent.
The rare coherent ones are cognitively novel.

Rough estimate: visible at **gen 50–200**. Below that, h_state is mutation noise.
At ~600 ticks/sec headless, gen 100 is reachable in minutes.

---

## long-run prediction summary

| trait            | predicted long-run          | mechanism                        |
|------------------|-----------------------------|----------------------------------|
| size             | 8.5–9.0 (corner)            | reserve time ∝ size^1.25         |
| FOV              | 35–50°                      | prey tracking at observed speed  |
| speed            | near current                | quadratic energy cost sweet spot |
| hgt_eat_rate     | 0.01–0.05 for carnivores    | Red Queen via predation          |
| hgt_contact_rate | near min unless clustering  | weaker signal, noisy             |
| weight_decay     | flat / neutral              | vestigial, no selection pressure |
| color            | path-dependent              | arms race from rainbow start     |
| h_state          | meaningful at gen 100+      | selection on multi-tick behavior |

---

## what "1 billion ticks" actually means

At ~600 ticks/sec headless on Apple Silicon, 1 billion ticks = ~19 days of runtime.

The founding genome is irrelevant after ~170,000 ticks (a few minutes). By 1 billion
ticks, the population has thermalized — the only surviving information is what selection
kept. Weights distribute around the ESS attractors above, jittered by mutation and
shuffled by HGT.

You can't predict which individual wight wins. You can predict the distribution
they're drawn from.
