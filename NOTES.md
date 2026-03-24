# observations

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

