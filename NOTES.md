# observations

---

## run 003 — 2026-03-25

**setup:** seed 680501, coastline_x=350 (land/sea split at world midpoint), sunlight=0.25, drain_scale=0.010 — same terrain config as run 002. b1/b2 bias weights added to genome this run, Wh and RGB color vision also active.

**snapshot:** tick 15,627 · pop 643 · max gen 29 · max age 5,327 · max ate 251

**what happened:**

population stabilized around 600–650, no extinction events. a single organism survived 5,327 ticks — roughly 34% of the entire run — indicating a near-immortal grazer that found a stable niche and compounded it.

no land/sea speciation visible by tick 15,627, unlike run 002. the right half (land) is noticeably sparser but not empty — some wights are on land, but no distinct land clade has split off yet. run 002 saw speciation earlier; the difference may be seed (vent positions relative to coastline) or that b1/b2 starting at zero slowed early adaptation.

**ray gene converged to near-zero.** this is the dominant signal. most wights evolved to be nearly blind — 0 or 1 active rays. food density is high enough that random wandering beats paying the sensing tax. the RGB color vision machinery (3 channels per ray, full genome cost) is going unused. this is a food-density problem: vents are too generous to force investment in sight.

hall of fame is a monoculture: all top 5 eaters are gen 18, spd 2.0, sz 8.8 — identical phenotype. strategy-space PCA shows one tight cluster with sparse outliers. fast convergence on a medium-size, medium-speed grazer; no size arms race, no predation pressure.

mouth and predx both near zero. this is a pure herbivore world — no wight-on-wight predation worth measuring. hgt-con is wide, suggesting active contact gene transfer, but it's shuffling the same winning genotype rather than introducing new diversity.

b1/b2 were loaded as zeros (new feature, fresh genome). 29 generations is too few to see bias weights pull neuron resting states away from zero.

**open questions:**
- will land/sea speciation emerge later in this run, or does this seed's vent layout prevent the tidal-pool bridge that enabled run 002?
- does food scarcity force the ray gene off zero, or do wights evolve other strategies first?
- will b1/b2 bias develop visible effect by gen 100+?
- the near-immortal wight at age 5,327 — is it on land (sunlight income) or parked next to a vent?

---

## run 002 — 2026-03-24

**setup:** 300 wights, coastline_x=350 (land/sea split), sunlight=0.25, drain_scale=0.010, turn_tax=0.01, vents spawn up to shoreline (tidal pools)

**what happened:**

speciation confirmed. 79% of the population colonized land within the run. land lineage evolved significantly lower speed and shrank brains down to ~3 active neurons — converging on a plant-like phenotype. sea wights retained fast speed and larger brains (~full neuron budget), staying predatory.

tidal pool vents acted as a bridge. early colonizers could exploit vent energy near the shoreline before photosynthesis pressure fully kicked in. without this, land was a death zone — all wights starved instantly because sunlight (0.05 at the time) couldn't cover basal drain.

the "valley of death" was solved by: (1) buffing sunlight 5x to 0.25, (2) lowering drain_scale from 0.015 to 0.010, (3) placing vents adjacent to the coast. once plants could survive long enough to reproduce, selection pressure sculpted them rapidly.

**turning mechanics (analysis):**

wights have analog steering. `out[:, 0]` from the RNN is a tanh output in `[-1.0, 1.0]` and is applied each tick as `pop['angle'] += turns` where `turns = out[:, 0] * pop['turn_s']`. this is cumulative — a wight holding `+0.5` will spiral continuously.

however, the RNN carries hidden state (`h_prev`) between ticks. if a wight stabilizes its hidden state and settles its turning output toward `0.0`, it locks in a heading and travels straight. this is energetically favorable now that `TURN_TAX = np.abs(turns) * pop['size'] * 0.01` penalizes every tick of rotation. evolution should reward wights that scan briefly, commit to a heading, and glide — rather than spinning indefinitely.

**open questions:**
- does the land clade eventually lose all motility (speed → 0)?
- will sea predators evolve to cross the coastline to raid the dense plant clusters?
- does h_state in the plant clade go dormant (near-zero activations), or does it track something environmental like day/vent proximity?
- at what depth does turn_tax visibly flatten angular velocity distributions in the population?

---

## run 001 — 2026-03-23

**setup:** 300 wights, 200 food, 900×900 world, recurrent brain (Elman RNN), camouflage active
**duration:** ~30s headless, 7,891 ticks, 263 ticks/sec

**what happened:**

population hits 300 within the first few hundred ticks and never leaves. the cap is the pressure. no cap, no evolution — just growth.

size converged hard and fast. average size went from random toward 7.8 (max is 9) and stayed there. once one big wight exists it eats the small ones and breeds. the children are big. the small ones die. it's not gradual — it tips.

FOV narrowed steadily the whole run: 73° → 64°. this is the most interesting signal so far. wide FOV = more peripheral input, but the brain has to do something useful with it. narrow FOV = focused forward. the survivors are hunters, not scanners.

speed and drain found a stable equilibrium fast — speed ~2.4, drain ~0.08. not maxed. the energy budget has a sweet spot and the weights found it.

color barely moved (140.4 → 141.6). camouflage pressure exists — bright prey have a larger predation detection radius — but size selection is so dominant it swamps everything else. color is nearly neutral at this timescale.

hidden state (h) is very active — mean absolute value 0.70, nearly saturated at ±1. the brain is writing strongly into h every tick. whether it's doing something useful — fear, anticipation, momentum — takes more generations to tell. only 7 generations in 30s. the dominant wights live long and breed slowly.

**open questions:**
- does FOV keep narrowing or does it hit a floor?
- at what generation does h_state start doing something interpretable?
- will color ever diverge, or does size just win forever?
- what breaks the size monoculture? (a fast small organism that evades?)
