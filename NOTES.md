# observations

---

## 9b99bae:334770 · 2026-03-25 · crash-and-lock

**setup:** 3 vents (sea side, x: 154–349), coastline default, drain_scale default. fresh 12 wights.

**screenshot:** tick 13,701 · pop dense blue monoculture, lineage river nearly flat, strategy space collapsed

**snapshot:** tick 12,943 · pop 1,634 · max gen 19 · max age 12,500 · max eaten 600

![screenshot](https://github.com/nguvan777-0/beetle-brain/releases/download/screenshots/screenshot_s334770_0013701.png)

**what happened:**

two-phase run with a hard break at tick ~6,500.

**phase 1 (ticks 0–6,000):** textbook slow growth. pop climbed 12→234, brains healthy at ~21 neurons, rays low (2.4 — vision already collapsing under density), speed steady at 2.0. then the population crashed: 234→154 in a single 500-tick window. no flag fired except brain pruning (-2.2 neurons). with only 3 vents, food couldn't support the density.

**phase 2 (ticks 6,500–12,943): the lock.** the crash was a selection event. what survived it: large body (size 6.1→8.1, now 85% of range), wide vision (rays 2.4→4.0 — vision *expanded* as crowding dropped), slower speed (2.0→1.78), lean brain (21→12 neurons). this morphotype then boomed: 103 wights at tick 7,500 to 1,634 at exit.

**mutation rate collapsed to 13%.** this is the most striking signal. once the crash-survivors locked in, they stopped exploring. the winning genotype is being copied with high fidelity. the population is converging with near-zero drift.

the oldest wight has been alive since tick ~2,885 — survived the crash, survived the boom. gen 0. ate only 9 things. a low-key grazer that outlasted everything.

lineage 7 holds 29% of the population. it and its near-clones (lineages 64, 83, 574 — all size 8.14, ~4.0 rays, 11 neurons) are essentially the same wight, slightly relabeled by time of birth.

**only 3 vents mattered.** less food forced a harder selection event than the 5-vent runs. the crash happened earlier relative to population size, and the lock was sharper.

```
============================================================
  beetle-brain · run report
  12,443 ticks  ·  1.0s  ·  12,268 t/s  ·  seed 334770  ·  pop 1634
============================================================

  max gen        19
  max age    12,500
  max eaten     600

  oldest wight born ~tick 2,885  (survived 10,058 ticks)

  food on map : 94
  vents       : 3
    vent 1  x=349  y=531  (sea)
    vent 2  x=154  y=494  (sea)
    vent 3  x=228  y=279  (sea)

────────────────────────────────────────────────────────────
  trait                  mean      min      max   range%
────────────────────────────────────────────────────────────
  speed                  1.78     0.75     2.53      40%
  fov°                  91.64    68.56   138.57      57%
  ray_len              109.40    68.58   160.67      61%
  size                   8.09     7.09     8.55      85%
  breed_at             124.77    96.64   169.23      39%
  clone_with            62.07    35.38    79.40      62%
  mutation_rate          0.07     0.04     0.14      13%
  mutation_scale         0.24     0.11     0.30      45%
  epigenetic             0.75     0.59     0.84      75%
  weight_decay           0.00     0.00     0.00      29%
  mouth                  2.72     1.95     4.82      25%
  pred_ratio             1.61     1.19     1.86      59%
  hgt_eat_rate           0.10     0.02     0.13      67%
  hgt_contact            0.01     0.01     0.02      69%
  active_neurons         11.9        6       24      37%
  n_rays                 3.96        1        6      57%
────────────────────────────────────────────────────────────

  mean color  r=47  g=123  b=227

────────────────────────────────────────────────────────────
  hall of fame
────────────────────────────────────────────────────────────
  longest survivor
    age 12,500  ·  ate 9  ·  gen 0
    size 8.14  speed 1.89  fov 90.0°  pred 1.63
  most kills
    age 7,863  ·  ate 72  ·  gen 1
    size 8.14  speed 1.89  fov 90.0°  pred 1.83
  eldest lineage
    age 173  ·  ate 9  ·  gen 19
    size 7.81  speed 1.89  fov 88.2°  pred 1.83

────────────────────────────────────────────────────────────
  trajectory  (tick 0 → 6,500 → 12,443)
────────────────────────────────────────────────────────────
  pop             16.0 →  134.0 → 1634.0          ▁▁▁▁▁▁  ▁▁▁▁▁▂▂▃▅█
  max_gen          1.0 →    9.0 →   19.0    ▁▁▂▂▂▃▃▃▃▄▄▄▄▅▅▅▄▄▅▅▅▆▇█
  size             6.4 →    6.9 →    8.1  ▂▁▁   ▁▁▁▁▁▁▂▄▆▇▇▇▇▇██████
  speed            2.0 →    1.6 →    1.8  ▆▇▇█▇▇▇▆▆▇▇▆▄▃  ▁▁▁▁▂▃▃▃▄▄
  fov°            87.3 →  102.8 →   91.6   ▂▆▃▅▆▆▆█▇▇▇██▇▆▅▅▅▄▄▃▃▃▂▂
  pred_ratio       1.5 →    1.6 →    1.6  ▁▄▄▄▆▆▆▇████▅▃  ▁▂▂▂▃▄▄▄▄▄
  mut_rate         0.1 →    0.1 →    0.1  ▆█▆▆▅▄▄▄▄▄▄▄▃▃▁▁▁▁▁▁      
  hgt_eat          0.1 →    0.1 →    0.1  ▅▇▇▇█▇▇████▇▅▃  ▁▃▃▄▅▆▇▇██
  hgt_contact      0.0 →    0.0 →    0.0   ▁▄▄▇▇▆▇▇▇▇▇▇█████████████
  n_rays           3.2 →    3.9 →    4.0  ▃▂▂▁▁       ▃▅███▇▇▇▆▆▆▅▅▅
  active_neur     17.8 →   17.0 →   11.9  ▅▅▇▇▇██████▇▆▄▁▁▁▁▁       

  drain / tick (mean at end)
    kleiber      0.0480  ▂▁▁   ▁▁▁▁▁▁▂▄▆▇▇▇▇▇██████
    speed        0.0131  ▇▇▆█▇▆▆▆▆▆▆▆▄▂  ▁▁▁▁▁▂▃▃▃▃
    size         0.0197  ▂▁▁   ▁▁▁▁▁▁▂▄▆▇▇▇▇▇██████
    sensing      0.0070   ▁▂▁▃▃▂▃▃▃▃▃▅▆██▇▆▆▅▅▄▃▃▂▂
    brain        0.0084  ▅▅▇▇▇█▇▇███▇▆▄▁▁▁▁▁       

────────────────────────────────────────────────────────────
  genome at exit  (position within each gene's range)
────────────────────────────────────────────────────────────
  speed             ████░░░░░░   40%
  fov               █████▋░░░░   57%
  ray_len           ██████▏░░░   61%
  size              ████████▌░   85%
  r                 ▍░░░░░░░░░    3%
  g                 ███▉░░░░░░   39%
  b                 ████████▊░   87%
  turn_s            ████████▍░   83%
  breed_at          ███▉░░░░░░   39%
  clone_with        ██████▏░░░   62%
  mut_rate          █▍░░░░░░░░   13%
  mut_scale         ████▌░░░░░   45%
  epigenetic        ███████▌░░   75%
  wt_decay          ██▉░░░░░░░   29%
  mouth             ██▌░░░░░░░   25%
  pred_ratio        █████▉░░░░   59%
  hgt_eat           ██████▋░░░   67%
  hgt_contact       ██████▉░░░   69%
  n_rays            █████▋░░░░   57%
  active_neur       ███▊░░░░░░   37%

────────────────────────────────────────────────────────────
  lineage river  (134 total)
────────────────────────────────────────────────────────────
  7           born tick      0  share   29%  final   557                ▁▁▁▁▁▁▁▂▂▂▄█
             size 8.15  speed 1.86  pred 1.60  n_rays 4.1  neurons 11
  1           born tick      0  share    8%  final     1    ▁▁▂▃▄▅▆▇█▇▃▁    
  64          born tick  6,000  share    7%  final   165        ▁▁▂▂▂▅█
             size 8.14  speed 1.96  pred 1.61  n_rays 4.0  neurons 11
  83          born tick  8,000  share    6%  final   146       ▂▂▃▆█
             size 8.14  speed 1.98  pred 1.60  n_rays 4.0  neurons 11
  9           born tick      0  share    5%  final    44         ▁▁▁▁▂▃▃▃▃▂▂▃▃▄▃▃▃▅█
             size 7.64  speed 0.94  pred 1.48  n_rays 4.5  neurons 11
  574         born tick 10,000  share    4%  final    92   ▁▃▅▆█
             size 8.12  speed 1.99  pred 1.59  n_rays 4.0  neurons 12
  384         born tick  6,000  share    2%  final    32       ▁▁▂▂▃▃▄▆█
             size 8.11  speed 0.94  pred 1.58  n_rays 4.3  neurons 10
  91          born tick  5,500  share    2%  final    21   ▁▃▃▃▃▃▃▃▃▃▃▃▅█
             size 7.64  speed 0.99  pred 1.54  n_rays 4.2  neurons 11

────────────────────────────────────────────────────────────
  strategy spread at final snapshot  — tight cluster, monoculture
────────────────────────────────────────────────────────────
  size        mean  8.09  std 0.19  range 7.09–8.55
  pred_ratio  mean  1.61  std 0.11  range 1.19–1.86
  speed       mean  1.78
```


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
