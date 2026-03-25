# observations

---

## a787e90:753463 · 2026-03-25 · the size lock

**setup:** coastline_x=350, sunlight=0.25, drain_scale=0.010. all 5 vents on the sea side (x: 184–245). b1/b2, Wh, RGB color vision active.

**screenshot:** tick 7,937 · pop 296 · max gen 9 · max age 6,952 · max eaten 20

**snapshot:** tick 8,854 · pop 497 · max gen 10 · max age 7,869 · max eaten 20

![screenshot](https://github.com/nguvan777-0/beetle-brain/releases/download/screenshots/screenshot_s753463_0007937.png)

**what happened:**

size is the dominant story. mean 8.61, max 8.75 out of 9.0 — 93% of the range, reached by gen 9. once one big wight exists it eats the small ones and breeds big. it tips, it doesn't drift.

the immortal grazer. max age 6,952 at tick 7,937 — that organism has been alive since roughly tick 985, before gen 5 existed. it found something and never lost it. K-strategist profile: breed_at mean 138/180, slow speed 1.77, conservative clone_with 43/85.

**n_rays collapsed, ray_len didn't.** mean n_rays 1.63 (min 1), but ray_len held at mean 87/180. wights are paying the sensing tax for long rays they mostly don't cast. ray_len alone isn't punishing enough to drive it to zero — only the count collapsed. all 5 vents are on the sea side (left, x: 184–245) with ~500 wights packed around them. at that density a wight with one ray still collides with food by proximity alone.

**brain held at 54% capacity.** mean 17/32 active neurons. not shrinking despite no predation and minimal vision. the hidden state is carrying something useful tick to tick — energy tracking, momentum, turn history.

**population grew from 296 to 497 between tick 7,937 and 8,854.** nearly doubled in ~900 ticks. the run is still expanding.

predation absent: mean eaten 0.3, max 20. pred_ratio mean 1.37 — the gene didn't collapse, just waiting for prey. no land/sea speciation at gen 10 — land (right, x > 350) has sunlight but no vents, and nothing has survived long enough on sunlight alone to establish a land clade.

---

## b5aa402:680501 · 2026-03-25 · the blind run

**setup:** coastline_x=350 (land/sea split at world midpoint), sunlight=0.25, drain_scale=0.010. b1/b2 bias weights added to genome this run, Wh and RGB color vision also active.

**snapshot:** tick 15,627 · pop 643 · max gen 29 · max age 5,327 · max ate 251

**what happened:**

population stabilized around 600–650, no extinction events. a single organism survived 5,327 ticks — roughly 34% of the entire run — indicating a near-immortal grazer that found a stable niche and compounded it.

no land/sea speciation visible by tick 15,627, unlike the speciation run. the right half (land) is noticeably sparser but not empty — some wights are on land, but no distinct land clade has split off yet. the difference may be seed (vent positions relative to coastline) or that b1/b2 starting at zero slowed early adaptation.

**ray gene converged to near-zero.** this is the dominant signal. most wights evolved to have near-zero vision — 0 or 1 active rays. food density is high enough that random wandering beats paying the sensing tax. the RGB color vision machinery (3 channels per ray, full genome cost) is going unused. this is a population-density problem: at 600–650 wights packed around a handful of vents, a wight with no vision collides with food and prey by proximity alone — sight adds no marginal value. sight evolves when prey is worth tracking across real distance; that distance never materializes here.

hall of fame is a monoculture: all top 5 eaters are gen 18, spd 2.0, sz 8.8 — identical phenotype. strategy-space PCA shows one tight cluster with sparse outliers. fast convergence on a medium-size, medium-speed grazer; no size arms race, no predation pressure.

mouth and predx both near zero. this is a pure herbivore world — no wight-on-wight predation worth measuring. hgt-con is wide, suggesting active contact gene transfer, but it's shuffling the same winning genotype rather than introducing new diversity.

b1/b2 were loaded as zeros (new feature, fresh genome). 29 generations is too few to see bias weights pull neuron resting states away from zero.

---

## 8ecd0a7:seed42 · 2026-03-24 · speciation

**setup:** 300 wights, coastline_x=350 (land/sea split), sunlight=0.25, drain_scale=0.010, turn_tax=0.01, vents spawn up to shoreline (tidal pools)

**what happened:**

speciation confirmed. 79% of the population colonized land within the run. land lineage evolved significantly lower speed and shrank brains down to ~3 active neurons — converging on a plant-like phenotype. sea wights retained fast speed and larger brains (~full neuron budget), staying predatory.

tidal pool vents acted as a bridge. early colonizers could exploit vent energy near the shoreline before photosynthesis pressure fully kicked in. without this, land was a death zone — all wights starved instantly because sunlight (0.05 at the time) couldn't cover basal drain.

the "valley of death" was solved by: (1) buffing sunlight 5x to 0.25, (2) lowering drain_scale from 0.015 to 0.010, (3) placing vents adjacent to the coast. once plants could survive long enough to reproduce, selection pressure sculpted them rapidly.

**turning mechanics (analysis):**

wights have analog steering. `out[:, 0]` from the RNN is a tanh output in `[-1.0, 1.0]` and is applied each tick as `pop['angle'] += turns` where `turns = out[:, 0] * pop['turn_s']`. this is cumulative — a wight holding `+0.5` will spiral continuously.

however, the RNN carries hidden state (`h_prev`) between ticks. if a wight stabilizes its hidden state and settles its turning output toward `0.0`, it locks in a heading and travels straight. this is energetically favorable now that `TURN_TAX = np.abs(turns) * pop['size'] * 0.01` penalizes every tick of rotation. evolution should reward wights that scan briefly, commit to a heading, and glide — rather than spinning indefinitely.

---

## 07352b5:seed42 · 2026-03-23 · size monoculture

**setup:** 300 wights, 200 food, 900×900 world, recurrent brain (Elman RNN), camouflage active
**duration:** ~30s headless, 7,891 ticks, 263 ticks/sec

**what happened:**

population hits 300 within the first few hundred ticks and never leaves. the cap is the pressure. no cap, no evolution — just growth.

size converged hard and fast. average size went from random toward 7.8 (max is 9) and stayed there. once one big wight exists it eats the small ones and breeds. the children are big. the small ones die. it's not gradual — it tips.

FOV narrowed steadily the whole run: 73° → 64°. this is the most interesting signal so far. wide FOV = more peripheral input, but the brain has to do something useful with it. narrow FOV = focused forward. the survivors are hunters, not scanners.

speed and drain found a stable equilibrium fast — speed ~2.4, drain ~0.08. not maxed. the energy budget has a sweet spot and the weights found it.

color barely moved (140.4 → 141.6). camouflage pressure exists — bright prey have a larger predation detection radius — but size selection is so dominant it swamps everything else. color is nearly neutral at this timescale.

hidden state (h) is very active — mean absolute value 0.70, nearly saturated at ±1. the brain is writing strongly into h every tick. whether it's doing something useful — fear, anticipation, momentum — takes more generations to tell. only 7 generations in 30s. the dominant wights live long and breed slowly.
