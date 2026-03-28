"""
report.py — generate a self-contained HTML report from a StatsCollector.

    uv run --with numpy --with plotly python report.py   # regenerate from snapshot

Or call generate(stats, path) directly from run_headless.py / game/main.py.
"""
import colorsys
import subprocess


def _report_stem(stats):
    """Build report filename stem: reports/report_{commit}_{seed}_{tick:07d}"""
    import os
    meta = stats.run_meta
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = 'unknown'
    seed = meta.get('seed', 0)
    tick = meta.get('final_tick', meta.get('ticks', 0))
    os.makedirs("reports", exist_ok=True)
    return f"reports/report_{commit}_{seed}_{tick:07d}"


def _hue_to_rgb_css(hue, s=0.85, v=0.9):
    r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def _hue_to_rgba_css(hue, alpha, s=0.85, v=0.9):
    r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


def _lineage_trait_means(stats, samples):
    """Return {uid: {size, speed, pred, n_rays, neurons}} for all lineages with wights at final sample."""
    if not samples:
        return {}
    last       = samples[-1]
    hues_all   = last.get('lineage_hues_all', [])
    sz_all     = last.get('size_all', [])
    sp_all     = last.get('speed_all', [])
    pr_all     = last.get('pred_ratio_all', [])
    nr_all     = last.get('n_rays_all', [])
    an_all     = last.get('active_neurons_all', [])
    hue_to_uid = {v: k for k, v in stats._lineage_hues.items()}
    buckets    = {}
    for i, h in enumerate(hues_all):
        uid = hue_to_uid.get(h)
        if uid is None:
            continue
        buckets.setdefault(uid, []).append(i)
    result = {}
    for uid, idxs in buckets.items():
        n = len(idxs)
        result[uid] = {
            'size':    sum(sz_all[i] for i in idxs) / n,
            'speed':   sum(sp_all[i] for i in idxs) / n,
            'pred':    sum(pr_all[i] for i in idxs) / n,
            'n_rays':  sum(nr_all[i] for i in idxs) / n if nr_all else 0,
            'neurons': sum(an_all[i] for i in idxs) / n if an_all else 0,
            'count':   n,
        }
    return result


def _sparkline(series, width=28):
    import math
    BLOCKS = " ▁▂▃▄▅▆▇█"
    lo, hi = min(series), max(series)
    if hi == lo:
        return BLOCKS[4] * width
    norm  = [(v - lo) / (hi - lo) for v in series]
    idxs  = [round(v * (len(BLOCKS) - 1)) for v in norm]
    step  = max(1, len(idxs) // width)
    sampled = idxs[::step][:width]
    return "".join(BLOCKS[i] for i in sampled)


def generate_text(stats, path=None, world=None, tick=None):
    import numpy as np
    samples = stats.samples
    meta    = stats.run_meta
    hof     = stats.hall_fame

    lines = []
    w = lines.append

    w("=" * 60)
    w("  beetle-brain · run report")
    ticks   = meta.get('ticks', 0)
    elapsed = meta.get('elapsed', 0)
    tps     = meta.get('tps', 0)
    seed    = meta.get('seed', '?')
    status  = "EXTINCT" if meta.get('extinct') else f"pop {meta.get('final_pop', '?')}"
    w(f"  {ticks:,} ticks  ·  {elapsed:.1f}s  ·  {tps:,.0f} t/s  ·  seed {seed}  ·  {status}")
    w("=" * 60)

    if samples:
        max_age   = max(s['max_age']   for s in samples)
        max_hunts  = max(s['max_hunts']  for s in samples)
        max_grazed = max(s['max_grazed'] for s in samples)
        max_gen   = max(s['max_gen']   for s in samples)
        w(f"\n  max gen    {max_gen:>6}")
        w(f"  max age    {max_age:>6,}")
        w(f"  max hunts  {max_hunts:>6}")
        w(f"  max grazed {max_grazed:>6}")

    # snapshot traits, vents, oldest wight — only when pop is available
    if world is not None and tick is not None:
        from sim.config import (
            SPEED_MIN, SPEED_MAX, FOV_MIN, FOV_MAX, RAY_MIN, RAY_MAX,
            SIZE_MIN, SIZE_MAX, MOUTH_MIN, MOUTH_MAX,
            BREED_AT_MIN, BREED_AT_MAX, CLONE_WITH_MIN, CLONE_WITH_MAX,
            MUTATION_RATE_MIN, MUTATION_RATE_MAX,
            MUTATION_SCALE_MIN, MUTATION_SCALE_MAX,
            EPIGENETIC_MIN, EPIGENETIC_MAX,
            WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX,
            HGT_EAT_MIN, HGT_EAT_MAX, HGT_CONTACT_MIN, HGT_CONTACT_MAX,
            PRED_RATIO_MIN, PRED_RATIO_MAX, N_RAYS, N_HIDDEN,
            COASTLINE_X,
        )
        pop   = world['pop']
        food  = world.get('food', [])
        vents = world.get('vents', [])
        N     = len(pop['x'])

        def _pct(val, lo, hi):
            return 100 * (val - lo) / (hi - lo) if hi > lo else 0.0

        born_at = tick - int(pop['age'].max()) if N > 0 else 0
        w(f"\n  oldest wight born ~tick {born_at:,}  (survived {int(pop['age'].max()):,} ticks)")
        w(f"\n  food on map : {len(food)}")
        w(f"  vents       : {len(vents)}")
        for i, v in enumerate(vents):
            side = "sea" if v[0] < COASTLINE_X else "land"
            w(f"    vent {i+1}  x={v[0]:.0f}  y={v[1]:.0f}  ({side})")

        TRAITS = [
            ('speed',          pop['speed'],             SPEED_MIN,         SPEED_MAX,   ''),
            ('fov°',           np.degrees(pop['fov']),   np.degrees(FOV_MIN), np.degrees(FOV_MAX), ''),
            ('ray_len',        pop['ray_len'],            RAY_MIN,           RAY_MAX,    ''),
            ('size',           pop['size'],               SIZE_MIN,          SIZE_MAX,   ''),
            ('breed_at',       pop['breed_at'],           BREED_AT_MIN,      BREED_AT_MAX, ''),
            ('clone_with',     pop['clone_with'],         CLONE_WITH_MIN,    CLONE_WITH_MAX, ''),
            ('mutation_rate',  pop['mutation_rate'],      MUTATION_RATE_MIN, MUTATION_RATE_MAX, ''),
            ('mutation_scale', pop['mutation_scale'],     MUTATION_SCALE_MIN,MUTATION_SCALE_MAX, ''),
            ('epigenetic',     pop['epigenetic'],         EPIGENETIC_MIN,    EPIGENETIC_MAX, ''),
            ('weight_decay',   pop['weight_decay'],       WEIGHT_DECAY_MIN,  WEIGHT_DECAY_MAX, ''),
            ('mouth',          pop['mouth'],              MOUTH_MIN,         MOUTH_MAX,  ''),
            ('pred_ratio',     pop['pred_ratio'],         PRED_RATIO_MIN,    PRED_RATIO_MAX, ''),
            ('hgt_eat_rate',   pop['hgt_eat_rate'],       HGT_EAT_MIN,       HGT_EAT_MAX, ''),
            ('hgt_contact',    pop['hgt_contact_rate'],   HGT_CONTACT_MIN,   HGT_CONTACT_MAX, ''),
        ]

        w(f"\n{'─'*60}")
        w(f"  {'trait':<18}  {'mean':>7}  {'min':>7}  {'max':>7}  {'range%':>7}")
        w(f"{'─'*60}")
        for label, vals, lo, hi, _ in TRAITS:
            m = float(vals.mean())
            w(f"  {label:<18}  {m:>7.2f}  {float(vals.min()):>7.2f}  {float(vals.max()):>7.2f}"
              f"  {_pct(m, lo, hi):>6.0f}%")
        an = pop.get('active_neurons', np.zeros(N))
        nr = pop.get('n_rays',         np.zeros(N))
        an_m = float(an.mean())
        nr_m = float(nr.mean())
        w(f"  {'active_neurons':<18}  {an_m:>7.1f}  {int(an.min()):>7}  {int(an.max()):>7}"
          f"  {_pct(an_m, 0, N_HIDDEN):>6.0f}%")
        w(f"  {'n_rays':<18}  {nr_m:>7.2f}  {int(nr.min()):>7}  {int(nr.max()):>7}"
          f"  {_pct(nr_m, 0, N_RAYS):>6.0f}%")
        w(f"{'─'*60}")
        r_m = float(pop['r'].mean()) if 'r' in pop else 0
        g_m = float(pop['g'].mean()) if 'g' in pop else 0
        b_m = float(pop['b'].mean()) if 'b' in pop else 0
        w(f"\n  mean color  r={r_m:.0f}  g={g_m:.0f}  b={b_m:.0f}")

    # hall of fame
    hof_entries = [
        ('longest survivor', hof.get('longest'),  'age',    lambda w: f"age {w['age']:,}"),
        ('most hunts',       hof.get('hunter'),   'hunts',  lambda w: f"hunts {w['hunts']}"),
        ('most grazed',      hof.get('grazer'),   'grazed', lambda w: f"grazed {w['grazed']}"),
        ('eldest lineage',   hof.get('eldest'),   'gen',    lambda w: f"gen {w['generation']}"),
    ]
    if any(wight for _, wight, _, _ in hof_entries):
        w(f"\n{'─'*60}")
        w("  hall of fame")
        w(f"{'─'*60}")
        for title, wight, _, metric_fn in hof_entries:
            if wight is None:
                continue
            w(f"  {title}")
            w(f"    {metric_fn(wight)}  ·  age {wight['age']:,}  ·  gen {wight['generation']}")
            w(f"    size {wight['size']:.2f}  speed {wight['speed']:.2f}  "
              f"fov {wight['fov_deg']:.1f}°  pred {wight['pred_ratio']:.2f}")

    # trajectory
    if samples and len(samples) > 1:
        sticks = [s['tick'] for s in samples]
        smid   = len(samples) // 2
        t0, tm, t1 = sticks[0], sticks[smid], sticks[-1]

        w(f"\n{'─'*60}")
        w(f"  trajectory  (tick {t0:,} → {tm:,} → {t1:,})")
        w(f"{'─'*60}")

        def col(key):
            return [s[key] for s in samples]

        rows = [
            ("pop",          col('pop')),
            ("max_gen",      col('max_gen')),
            ("size",         col('size_mean')),
            ("speed",        col('speed_mean')),
            ("fov°",         col('fov_mean_deg')),
            ("pred_ratio",   col('pred_ratio_mean')),
            ("mut_rate",     col('mutation_mean')),
            ("hgt_eat",      col('hgt_eat_mean')),
            ("hgt_contact",  col('hgt_contact_mean')),
            ("n_rays",       col('n_rays_mean')),
            ("active_neur",  col('active_neurons_mean')),
        ]
        for label, series in rows:
            v0, vm, v1 = series[0], series[smid], series[-1]
            spark = _sparkline(series)
            w(f"  {label:<12}  {v0:>6.1f} → {vm:>6.1f} → {v1:>6.1f}  {spark}")

        # drain
        w(f"\n  drain / tick (mean at end)")
        for key, label in [
            ('drain_kleiber', 'kleiber'),
            ('drain_speed',   'speed'),
            ('drain_size',    'size'),
            ('drain_sensing', 'sensing'),
            ('drain_brain',   'brain'),
        ]:
            if key in samples[-1]:
                series = col(key)
                w(f"    {label:<10}  {series[-1]:>7.4f}  {_sparkline(series)}")

    # genome at exit
    if samples and 'genes_norm' in samples[-1]:
        from sim.stats import GENE_NAMES
        gene_vals = samples[-1]['genes_norm']
        w(f"\n{'─'*60}")
        w(f"  genome at exit  (position within each gene's range)")
        w(f"{'─'*60}")
        BAR = "░▏▎▍▌▋▊▉█"
        for name, val in zip(GENE_NAMES, gene_vals):
            filled = int(val * 10)
            frac   = val * 10 - filled
            bar    = "█" * filled + BAR[round(frac * 8)] + "░" * (9 - filled)
            w(f"  {name:<16}  {bar}  {val*100:>3.0f}%")

    # lineage river
    lineage_series = stats._lineage_series
    lineage_first  = stats._lineage_first_tick
    lineage_hues   = stats._lineage_hues
    if lineage_series:
        totals = {uid: sum(c for _, c in pts) for uid, pts in lineage_series.items()}
        grand  = sum(totals.values())
        top    = sorted(totals, key=totals.__getitem__, reverse=True)[:8]

        lin_traits = _lineage_trait_means(stats, samples)

        w(f"\n{'─'*60}")
        w(f"  lineage river  ({len(totals)} total)")
        w(f"{'─'*60}")
        for uid in top:
            first  = lineage_first.get(uid, 0)
            share  = 100 * totals[uid] / grand if grand else 0
            pts    = lineage_series[uid]
            final  = pts[-1][1] if pts else 0
            spark  = _sparkline([c for _, c in pts])
            w(f"  {uid:<10}  born tick {first:>6,}  share {share:>4.0f}%  final {final:>5}  {spark}")
            t = lin_traits.get(uid)
            if t:
                w(f"             size {t['size']:.2f}  speed {t['speed']:.2f}  "
                  f"pred {t['pred']:.2f}  n_rays {t['n_rays']:.1f}  neurons {t['neurons']:.0f}")

    # strategy spread — size vs pred_ratio at final snapshot
    if samples:
        last   = samples[-1]
        sizes  = last.get('size_all', [])
        preds  = last.get('pred_ratio_all', [])
        speeds = last.get('speed_all', [])
        if sizes and preds:
            import math
            n_pop   = len(sizes)
            sz_mean = sum(sizes) / n_pop
            pr_mean = sum(preds) / n_pop
            sz_std  = math.sqrt(sum((x - sz_mean)**2 for x in sizes) / n_pop)
            pr_std  = math.sqrt(sum((x - pr_mean)**2 for x in preds) / n_pop)
            sp_mean = sum(speeds) / n_pop
            # rough cluster check: bimodal if std is large relative to range
            sz_range = max(sizes) - min(sizes)
            pr_range = max(preds) - min(preds)
            sz_cv = sz_std / sz_mean if sz_mean else 0
            pr_cv = pr_std / pr_mean if pr_mean else 0
            cluster_hint = ""
            if sz_cv > 0.08 or pr_cv > 0.08:
                cluster_hint = "  — spread suggests multiple strategies"
            else:
                cluster_hint = "  — tight cluster, monoculture"
            w(f"\n{'─'*60}")
            w(f"  strategy spread at final snapshot{cluster_hint}")
            w(f"{'─'*60}")
            w(f"  size        mean {sz_mean:>5.2f}  std {sz_std:>4.2f}  range {min(sizes):.2f}–{max(sizes):.2f}")
            w(f"  pred_ratio  mean {pr_mean:>5.2f}  std {pr_std:>4.2f}  range {min(preds):.2f}–{max(preds):.2f}")
            w(f"  speed       mean {sp_mean:>5.2f}")

    w("")

    if path is None:
        path = _report_stem(stats) + ".txt"
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(f"[report] {path}")


def generate_summary(stats, world=None, tick=None):
    """Print a short run summary to stdout — key stats only, no file written."""
    samples = stats.samples
    meta    = stats.run_meta
    hof     = stats.hall_fame

    lines = []
    w = lines.append

    ticks   = meta.get('ticks', 0)
    elapsed = meta.get('elapsed', 0)
    tps     = meta.get('tps', 0)
    seed    = meta.get('seed', '?')
    status  = "EXTINCT" if meta.get('extinct') else f"pop {meta.get('final_pop', '?')}"

    w("")
    w("=" * 60)
    w("  beetle-brain · run report")
    w(f"  {ticks:,} ticks  ·  {elapsed:.1f}s  ·  {tps:,.0f} t/s  ·  seed {seed}  ·  {status}")
    w("=" * 60)

    if samples:
        max_age   = max(s['max_age']   for s in samples)
        max_hunts  = max(s['max_hunts']  for s in samples)
        max_grazed = max(s['max_grazed'] for s in samples)
        max_gen   = max(s['max_gen']   for s in samples)
        w(f"\n  gen {max_gen}  ·  age {max_age:,}  ·  hunts {max_hunts}  ·  grazed {max_grazed}")

    hof_entries = [
        ('longest survivor', hof.get('longest'),  lambda w: f"age {w['age']:,}"),
        ('most hunts',       hof.get('hunter'),   lambda w: f"hunts {w['hunts']}"),
        ('most grazed',      hof.get('grazer'),   lambda w: f"grazed {w['grazed']}"),
        ('eldest lineage',   hof.get('eldest'),   lambda w: f"gen {w['generation']}"),
    ]
    if any(wight for _, wight, _ in hof_entries):
        w("")
        for title, wight, metric_fn in hof_entries:
            if wight is None:
                continue
            w(f"  {title:<18}  {metric_fn(wight)}  "
              f"size {wight['size']:.1f}  spd {wight['speed']:.2f}  pred {wight['pred_ratio']:.2f}")

    if samples and len(samples) > 1:
        smid = len(samples) // 2
        sticks = [s['tick'] for s in samples]
        t0, tm, t1 = sticks[0], sticks[smid], sticks[-1]

        w(f"\n{'─'*60}")
        w(f"  trajectory  (tick {t0:,} → {tm:,} → {t1:,})")
        w(f"{'─'*60}")

        def col(key): return [s[key] for s in samples]

        for label, key in [
            ("pop",         'pop'),
            ("speed",       'speed_mean'),
            ("size",        'size_mean'),
            ("n_rays",      'n_rays_mean'),
            ("active_neur", 'active_neurons_mean'),
        ]:
            series = col(key)
            v0, vm, v1 = series[0], series[smid], series[-1]
            w(f"  {label:<12}  {v0:>6.1f} → {vm:>6.1f} → {v1:>6.1f}  {_sparkline(series)}")

    if samples:
        last  = samples[-1]
        sizes = last.get('size_all', [])
        preds = last.get('pred_ratio_all', [])
        if sizes and preds:
            import math
            n       = len(sizes)
            sz_mean = sum(sizes) / n
            pr_mean = sum(preds) / n
            sz_cv   = math.sqrt(sum((x - sz_mean)**2 for x in sizes) / n) / sz_mean if sz_mean else 0
            pr_cv   = math.sqrt(sum((x - pr_mean)**2 for x in preds) / n) / pr_mean if pr_mean else 0
            hint    = "multiple strategies" if sz_cv > 0.08 or pr_cv > 0.08 else "monoculture"
            w(f"\n  size {min(sizes):.2f}–{max(sizes):.2f}  "
              f"pred {min(preds):.2f}–{max(preds):.2f}  —  {hint}")

    w("")
    print("\n".join(lines))


def generate(stats, path=None, world=None, tick=None, write_txt=True):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        return

    samples = stats.samples
    meta    = stats.run_meta
    hof     = stats.hall_fame

    if not samples:
        print("[report] no samples — run was too short")
        return

    ticks = [s['tick'] for s in samples]

    # ── build figures ────────────────────────────────────────────────────────

    figs = []

    lin_traits = _lineage_trait_means(stats, samples)
    figs.append(_fig_lineage_tree(stats, lin_traits))
    figs.append(_fig_lineage_traits(stats, lin_traits))
    figs.append(_fig_population(samples, ticks))
    figs.append(_fig_traits(samples, ticks))
    figs.append(_fig_brain_vision(samples, ticks))
    figs.append(_fig_sensing(samples, ticks))
    figs.append(_fig_rates(samples, ticks))
    figs.append(_fig_drain(samples, ticks))
    figs.append(_fig_genome_heatmap(samples, ticks))
    figs.append(_fig_genome_exit(samples))
    figs.append(_fig_phase_scatter(samples, lin_traits, stats))

    # ── apply shared dark theme ───────────────────────────────────────────────
    for fig in figs:
        if fig is None:
            continue
        fig.update_layout(
            paper_bgcolor='#0d1117',
            plot_bgcolor='#161b22',
            font=dict(color='#c9d1d9', family='monospace', size=11),
            legend=dict(bgcolor='rgba(22,27,34,0.9)', bordercolor='#30363d',
                        borderwidth=1, font=dict(size=10)),
            margin=dict(t=50, b=40, l=60, r=40),
        )
        fig.update_xaxes(gridcolor='#21262d', zerolinecolor='#30363d', tickfont=dict(size=10))
        fig.update_yaxes(gridcolor='#21262d', zerolinecolor='#30363d', tickfont=dict(size=10))

    # ── render to HTML blocks ────────────────────────────────────────────────
    # include plotly JS only once (first non-None figure)
    chart_blocks = []
    plotlyjs_included = False
    for fig in figs:
        if fig is None:
            continue
        include = not plotlyjs_included
        block = pio.to_html(fig, full_html=False, include_plotlyjs=include)
        plotlyjs_included = True
        chart_blocks.append(f'<div class="chart">{block}</div>')

    # ── hall of fame ─────────────────────────────────────────────────────────
    hof_html = _hof_html(hof)

    # ── run summary bar ───────────────────────────────────────────────────────
    extinct_badge = (
        '<span class="badge extinct">EXTINCTION</span>'
        if meta.get('extinct') else
        f'<span class="badge alive">survived · pop {meta.get("final_pop","?")}</span>'
    )
    seed_span = (f'<span class="metric">seed <b>{meta["seed"]}</b></span>'
                 if meta.get('seed') is not None else '')
    max_age    = max((s['max_age']    for s in samples), default=0)
    max_hunts  = max((s['max_hunts']  for s in samples), default=0)
    max_grazed = max((s['max_grazed'] for s in samples), default=0)
    summary_html = f"""
    <div class="summary">
      <span class="metric"><b>{meta.get('ticks',0):,}</b> ticks</span>
      <span class="metric"><b>{meta.get('elapsed',0):.1f}s</b> elapsed</span>
      <span class="metric"><b>{meta.get('tps',0):,.0f}</b> ticks/sec</span>
      <span class="metric"><b>{meta.get('final_max_gen',0)}</b> max gen</span>
      <span class="metric"><b>{max_age:,}</b> max age</span>
      <span class="metric"><b>{max_hunts}</b> max hunts</span>
      <span class="metric"><b>{max_grazed}</b> max grazed</span>
      {seed_span}
      {extinct_badge}
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>beetle-brain · run report</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: #0d1117; color: #c9d1d9;
    font-family: 'SF Mono', 'Fira Mono', monospace;
  }}
  header {{
    padding: 20px 32px 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 4px;
  }}
  header h1 {{
    margin: 0 0 10px; font-size: 1.05em; color: #7ecfff;
    letter-spacing: 0.08em; font-weight: 600;
  }}
  .summary {{
    display: flex; align-items: center; gap: 24px;
    padding: 10px 0 16px; flex-wrap: wrap;
  }}
  .metric {{ font-size: 0.82em; color: #8b949e; }}
  .metric b {{ color: #e6edf3; }}
  .badge {{
    font-size: 0.75em; padding: 3px 10px; border-radius: 12px;
    font-weight: 600; letter-spacing: 0.05em;
  }}
  .badge.extinct {{ background: #3d1a1a; color: #ff6b6b; }}
  .badge.alive   {{ background: #1a3d2a; color: #56d364; }}

  .section-title {{
    padding: 20px 32px 4px; font-size: 0.7em; color: #484f58;
    text-transform: uppercase; letter-spacing: 0.12em;
  }}

  .hof {{
    display: flex; gap: 12px; padding: 8px 32px 20px; flex-wrap: wrap;
  }}
  .card {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 14px 18px; flex: 1; min-width: 200px;
  }}
  .card h3 {{
    margin: 0 0 10px; font-size: 0.72em; text-transform: uppercase;
    letter-spacing: 0.1em; color: #ffd580;
  }}
  .card table {{ border-collapse: collapse; width: 100%; }}
  .card td {{ padding: 2px 0; font-size: 0.78em; }}
  .card td:first-child {{ color: #8b949e; width: 90px; }}
  .card td:last-child  {{ color: #e6edf3; }}
  .swatch {{
    display: inline-block; width: 12px; height: 12px;
    border-radius: 3px; vertical-align: middle; margin-right: 5px;
  }}

  .charts {{ padding: 0 16px 40px; }}
  .chart  {{ margin-bottom: 4px; }}
</style>
</head>
<body>
<header>
  <h1>beetle-brain · run report</h1>
  {summary_html}
</header>

<div class="charts">
{"".join(chart_blocks)}
</div>

<div class="section-title">hall of fame</div>
{hof_html}

</body>
</html>"""

    if path is None:
        path = _report_stem(stats) + ".html"
    with open(path, 'w') as f:
        f.write(html)
    print(f"[report] written → {path}")

    if write_txt:
        txt_path = path[:-5] + ".txt" if path.endswith(".html") else path + ".txt"
        generate_text(stats, txt_path, world=world, tick=tick)


# ── individual figures ────────────────────────────────────────────────────────

def _fig_lineage_tree(stats, lin_traits=None):
    import plotly.graph_objects as go

    hues       = stats._lineage_hues
    first_tick = stats._lineage_first_tick
    parent_map = stats._lineage_parent_map
    series     = stats._lineage_series

    if not hues:
        return None

    nodes = set(hues.keys())

    # ── tree layout ───────────────────────────────────────────────────────────
    children = {n: [] for n in nodes}
    roots    = []
    for node in nodes:
        par = parent_map.get(node)
        if par in nodes:
            children[par].append(node)
        else:
            roots.append(node)

    # assign y via DFS — leaves get sequential slots, internals get midpoint
    y_pos   = {}
    counter = [0]

    def _assign_y(node):
        kids = sorted(children[node], key=lambda k: first_tick.get(k, 0))
        if not kids:
            y_pos[node] = counter[0]
            counter[0] += 1
        else:
            for kid in kids:
                _assign_y(kid)
            y_pos[node] = sum(y_pos[k] for k in kids) / len(kids)

    for root in sorted(roots, key=lambda r: first_tick.get(r, 0)):
        _assign_y(root)

    # total count per lineage for node sizing
    totals = {uid: sum(c for _, c in series.get(uid, [])) for uid in nodes}
    max_total = max(totals.values()) if totals else 1

    # ── build traces ─────────────────────────────────────────────────────────
    edge_x, edge_y = [], []
    for child, par in parent_map.items():
        if child not in y_pos or par not in y_pos:
            continue
        cx = first_tick.get(child, 0)
        cy = y_pos[child]
        px = first_tick.get(par, 0)
        py = y_pos[par]
        # L-shaped connector: horizontal from parent, then drop to child
        edge_x += [px, cx, cx, None]
        edge_y += [py, py, cy, None]

    node_x     = [first_tick.get(n, 0) for n in nodes]
    node_y     = [y_pos.get(n, 0)      for n in nodes]
    node_color = [_hue_to_rgb_css(hues.get(n, 0)) for n in nodes]
    node_size  = [6 + 18 * (totals.get(n, 0) / max_total) for n in nodes]
    node_text = []
    for n in nodes:
        t = lin_traits.get(n) if lin_traits else None
        trait_line = (
            f'<br>size {t["size"]:.2f}  speed {t["speed"]:.2f}  pred {t["pred"]:.2f}'
            f'<br>n_rays {t["n_rays"]:.1f}  neurons {t["neurons"]:.0f}'
            if t else ''
        )
        node_text.append(
            f'lineage {n}<br>first tick {first_tick.get(n,0):,}<br>'
            f'total wight-ticks {totals.get(n,0)}{trait_line}'
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='#30363d', width=1),
        hoverinfo='skip',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(color=node_color, size=node_size,
                    line=dict(color='#21262d', width=1)),
        text=node_text,
        hovertemplate='%{text}<extra></extra>',
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text='lineage tree — forks over time, node size = dominance, color = phylo hue',
                   font=dict(size=12, color='#8b949e')),
        height=max(300, len(nodes) * 14),
        xaxis_title='tick of first appearance',
        yaxis=dict(visible=False),
    )
    return fig


def _fig_population(samples, ticks):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['pop'] for s in samples],
        mode='lines', name='population',
        line=dict(color='#7ecfff', width=2),
        fill='tozeroy', fillcolor='rgba(126,207,255,0.08)',
        hovertemplate='tick %{x:,}<br>pop %{y}<extra></extra>',
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['max_gen'] for s in samples],
        mode='lines', name='max generation',
        line=dict(color='#ffd580', width=1.5, dash='dot'),
        hovertemplate='tick %{x:,}<br>gen %{y}<extra></extra>',
    ), secondary_y=True)
    fig.update_layout(
        title=dict(text='population & generation', font=dict(size=12, color='#8b949e')),
        height=260,
    )
    fig.update_yaxes(title_text='population', secondary_y=False)
    fig.update_yaxes(title_text='max gen', secondary_y=True, gridcolor='rgba(0,0,0,0)')
    return fig


def _fig_traits(samples, ticks):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['size_mean'] for s in samples],
        mode='lines', name='size mean',
        line=dict(color='#ff7eb3', width=2),
        hovertemplate='tick %{x:,}<br>size %{y:.2f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['size_min'] for s in samples],
        mode='lines', name='size min',
        line=dict(color='#ff7eb3', width=0, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['size_max'] for s in samples],
        mode='lines', name='size range',
        fill='tonexty', fillcolor='rgba(255,126,179,0.12)',
        line=dict(color='#ff7eb3', width=0),
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['speed_mean'] for s in samples],
        mode='lines', name='speed mean',
        line=dict(color='#80ffb4', width=2),
        hovertemplate='tick %{x:,}<br>speed %{y:.2f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='size & speed evolution', font=dict(size=12, color='#8b949e')),
        height=260,
    )
    return fig


def _fig_brain_vision(samples, ticks):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=ticks, y=[s.get('active_neurons_mean', 0) for s in samples],
        mode='lines', name='active neurons (mean)',
        line=dict(color='#7ecfff', width=2),
        fill='tozeroy', fillcolor='rgba(126,207,255,0.08)',
        hovertemplate='tick %{x:,}<br>active neurons %{y:.1f}<extra></extra>',
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=ticks, y=[s.get('n_rays_mean', 0) for s in samples],
        mode='lines', name='n_rays (mean)',
        line=dict(color='#ffd580', width=2),
        hovertemplate='tick %{x:,}<br>n_rays %{y:.2f}<extra></extra>',
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=ticks, y=[s.get('n_rays_min', 0) for s in samples],
        mode='lines', name='n_rays (min)',
        line=dict(color='#ffd580', width=1, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ), secondary_y=True)
    fig.update_layout(
        title=dict(text='brain & vision — active neurons vs ray count over time',
                   font=dict(size=12, color='#8b949e')),
        height=260,
    )
    fig.update_yaxes(title_text='active neurons (0–32)', secondary_y=False)
    fig.update_yaxes(title_text='n_rays (0–7)', secondary_y=True,
                     range=[0, 7], gridcolor='rgba(0,0,0,0)')
    return fig


def _fig_sensing(samples, ticks):
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, key, color in [
        ('FOV° mean',    'fov_mean_deg', '#c9b1ff'),
        ('FOV° min',     'fov_min_deg',  '#c9b1ff'),
        ('ray len mean', 'ray_mean',     '#ffa07a'),
        ('ray len min',  'ray_min',      '#ffa07a'),
    ]:
        dash = 'dot' if 'min' in name else 'solid'
        show = 'min' not in name
        fig.add_trace(go.Scatter(
            x=ticks, y=[s[key] for s in samples],
            mode='lines', name=name,
            line=dict(color=color, width=2 if show else 1, dash=dash),
            showlegend=show,
            hovertemplate=f'tick %{{x:,}}<br>{name} %{{y:.1f}}<extra></extra>',
        ))

    fig.update_layout(
        title=dict(text='sensing evolution — FOV & ray length (no vision → zero)', font=dict(size=12, color='#8b949e')),
        height=260,
    )
    return fig


def _fig_rates(samples, ticks):
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, key, color in [
        ('pred_ratio',   'pred_ratio_mean',  '#ff6b6b'),
        ('mutation',     'mutation_mean',     '#aaffaa'),
        ('hgt eat',      'hgt_eat_mean',      '#ffd580'),
        ('hgt contact',  'hgt_contact_mean',  '#80ddff'),
    ]:
        fig.add_trace(go.Scatter(
            x=ticks, y=[s[key] for s in samples],
            mode='lines', name=name,
            line=dict(width=2),
            hovertemplate=f'tick %{{x:,}}<br>{name} %{{y:.4f}}<extra></extra>',
        ))
    fig.update_layout(
        title=dict(text='evolved rates — predation aggression, mutation, HGT', font=dict(size=12, color='#8b949e')),
        height=260,
    )
    return fig


def _fig_drain(samples, ticks):
    import plotly.graph_objects as go

    components = [
        ('Kleiber (size^0.75)', 'drain_kleiber', '#ff9966'),
        ('speed²',              'drain_speed',   '#ffcc44'),
        ('size²',               'drain_size',    '#ff6699'),
        ('sensing (ray×fov)',   'drain_sensing', '#9966ff'),
        ('brain (neur^1.5)',    'drain_brain',   '#7ecfff'),
    ]
    fig = go.Figure()
    for name, key, color in components:
        fig.add_trace(go.Scatter(
            x=ticks, y=[s[key] for s in samples],
            mode='lines', name=name,
            stackgroup='drain',
            line=dict(width=0.5, color=color),
            fillcolor=color.replace(')', ',0.7)').replace('rgb', 'rgba') if color.startswith('rgb') else color,
            hovertemplate=f'tick %{{x:,}}<br>{name} %{{y:.5f}}<extra></extra>',
        ))
    fig.update_layout(
        title=dict(text='metabolic drain breakdown (stacked, per wight per tick)', font=dict(size=12, color='#8b949e')),
        height=260,
        yaxis_title='energy/tick',
    )
    return fig


def _fig_genome_heatmap(samples, ticks):
    import plotly.graph_objects as go
    from sim.stats import GENE_NAMES

    if not samples or 'genes_norm' not in samples[0]:
        return None

    # z: shape (n_genes, n_ticks)
    z = [[s['genes_norm'][i] for s in samples] for i in range(len(GENE_NAMES))]

    fig = go.Figure(go.Heatmap(
        x=ticks,
        y=GENE_NAMES,
        z=z,
        colorscale='Viridis',
        zmin=0, zmax=1,
        colorbar=dict(
            title='← min  max →',
            tickvals=[0, 0.5, 1],
            ticktext=['min', 'mid', 'max'],
            thickness=12,
            len=0.8,
        ),
        hovertemplate='tick %{x:,}<br>gene %{y}<br>norm value %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='genome heatmap — all 20 genes, normalized 0→1 within range', font=dict(size=12, color='#8b949e')),
        height=380,
        yaxis=dict(autorange='reversed', tickfont=dict(size=10)),
        xaxis_title='tick',
    )
    return fig


def _fig_phase_scatter(samples, lin_traits=None, stats=None):
    import plotly.graph_objects as go

    if not samples:
        return None
    last = samples[-1]
    sizes  = last.get('size_all', [])
    speeds = last.get('speed_all', [])
    preds  = last.get('pred_ratio_all', [])
    hues   = last.get('lineage_hues_all', [])
    nr_all = last.get('n_rays_all', [])

    if not sizes:
        return None

    # build per-wight lineage uid lookup for hover
    hue_to_uid = {v: k for k, v in stats._lineage_hues.items()} if stats else {}
    hover = []
    for i, h in enumerate(hues):
        uid = hue_to_uid.get(h, '?')
        t   = (lin_traits or {}).get(uid)
        nr  = f'  n_rays {nr_all[i]:.0f}' if nr_all else ''
        lin = f'  lineage {uid}' if t else ''
        hover.append(f'size {sizes[i]:.2f}  speed {speeds[i]:.2f}  pred {preds[i]:.2f}{nr}{lin}')

    colors = [_hue_to_rgb_css(h) for h in hues]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=preds,
        mode='markers',
        name='wights',
        marker=dict(
            color=colors,
            size=[max(4, s * 2.5) for s in speeds],
            opacity=0.75,
            line=dict(width=0.3, color='rgba(255,255,255,0.15)'),
        ),
        text=hover,
        hovertemplate='%{text}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(
            text='phase space — size vs pred_ratio (color=lineage, marker size=speed, hover=full traits)',
            font=dict(size=12, color='#8b949e'),
        ),
        height=360,
        xaxis_title='size',
        yaxis_title='pred_ratio',
    )
    return fig


def _fig_lineage_traits(stats, lin_traits):
    """Parallel coordinates — each top lineage is a colored line across trait axes."""
    import plotly.graph_objects as go
    from sim.config import SPEED_MIN, SPEED_MAX, SIZE_MIN, SIZE_MAX, PRED_RATIO_MIN, PRED_RATIO_MAX, N_RAYS, N_HIDDEN

    if not lin_traits:
        return None

    hues   = stats._lineage_hues
    series = stats._lineage_series
    totals = {uid: sum(c for _, c in pts) for uid, pts in series.items()}
    top    = sorted(lin_traits, key=lambda u: totals.get(u, 0), reverse=True)[:12]

    if not top:
        return None

    # normalize each trait 0-1 within its range for parcoords
    def _norm(v, lo, hi): return (v - lo) / (hi - lo) if hi > lo else 0.5

    dims = [
        dict(label='size',    range=[SIZE_MIN, SIZE_MAX],        values=[lin_traits[u]['size']    for u in top]),
        dict(label='speed',   range=[SPEED_MIN, SPEED_MAX],      values=[lin_traits[u]['speed']   for u in top]),
        dict(label='pred',    range=[PRED_RATIO_MIN, PRED_RATIO_MAX], values=[lin_traits[u]['pred'] for u in top]),
        dict(label='n_rays',  range=[0, N_RAYS],                 values=[lin_traits[u]['n_rays']  for u in top]),
        dict(label='neurons', range=[0, N_HIDDEN],               values=[lin_traits[u]['neurons'] for u in top]),
    ]

    colors      = [hues.get(u, 0) for u in top]
    color_strs  = [_hue_to_rgb_css(h) for h in colors]
    pop_shares  = [totals.get(u, 0) for u in top]
    grand       = sum(pop_shares)
    line_widths = [max(1, 6 * s / grand) for s in pop_shares]

    fig = go.Figure()
    for i, uid in enumerate(top):
        vals = [d['values'][i] for d in dims]
        fig.add_trace(go.Scatter(
            x=[d['label'] for d in dims],
            y=vals,
            mode='lines+markers',
            name=f'lineage {uid}',
            line=dict(color=color_strs[i], width=max(1.5, line_widths[i])),
            marker=dict(size=6, color=color_strs[i]),
            hovertemplate=(
                f'lineage {uid}<br>'
                f'size {lin_traits[uid]["size"]:.2f}  '
                f'speed {lin_traits[uid]["speed"]:.2f}  '
                f'pred {lin_traits[uid]["pred"]:.2f}<br>'
                f'n_rays {lin_traits[uid]["n_rays"]:.1f}  '
                f'neurons {lin_traits[uid]["neurons"]:.0f}  '
                f'pop share {100*pop_shares[i]/grand:.0f}%'
                '<extra></extra>'
            ),
        ))

    fig.update_layout(
        title=dict(text='lineage strategy profiles — top clades across key traits (line weight = pop share)',
                   font=dict(size=12, color='#8b949e')),
        height=320,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        showlegend=True,
    )
    return fig


def _fig_genome_exit(samples):
    """Horizontal bar chart — each gene's final normalized position within its range."""
    import plotly.graph_objects as go
    from sim.stats import GENE_NAMES

    if not samples or 'genes_norm' not in samples[-1]:
        return None

    vals  = samples[-1]['genes_norm']
    names = GENE_NAMES
    colors = [
        f'hsl({int(v * 240)},70%,55%)' for v in vals
    ]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{v*100:.0f}%' for v in vals],
        textposition='outside',
        hovertemplate='%{y}  %{x:.3f}  (%{text} of range)<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='genome at exit — each gene normalized 0→1 within its range',
                   font=dict(size=12, color='#8b949e')),
        height=520,
        xaxis=dict(range=[0, 1.15], tickformat='.0%', title='position in range'),
        yaxis=dict(autorange='reversed'),
        bargap=0.25,
    )
    return fig


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _hof_html(hof):
    cards = []
    labels = {
        'longest': 'longest-lived',
        'hunter':  'most hunts',
        'grazer':  'most grazed',
        'eldest':  'highest gen',
    }
    for key, title in labels.items():
        w = hof.get(key)
        if w is None:
            continue
        r, g, b = w['r'], w['g'], w['b']
        lhue = w.get('lineage_hue', 0.0)
        lr, lg, lb = colorsys.hsv_to_rgb(lhue, 0.85, 0.9)
        body_swatch    = f'<span class="swatch" style="background:rgb({r},{g},{b})"></span>'
        lineage_swatch = f'<span class="swatch" style="background:rgb({int(lr*255)},{int(lg*255)},{int(lb*255)})"></span>'
        rows = [
            ('body color',   f'{body_swatch} rgb({r},{g},{b})'),
            ('lineage',      f'{lineage_swatch} hue {lhue:.3f}'),
            ('age',          f"{w['age']:,} ticks"),
            ('hunts',        str(w['hunts'])),
            ('grazed',       str(w['grazed'])),
            ('generation',   str(w['generation'])),
            ('size',         f"{w['size']:.2f}"),
            ('speed',        f"{w['speed']:.2f}"),
            ('FOV',          f"{w['fov_deg']:.1f}°"),
            ('ray_len',      f"{w['ray_len']:.1f}"),
            ('pred_ratio',   f"{w['pred_ratio']:.2f}"),
            ('mut_rate',     f"{w['mutation_rate']:.3f}"),
            ('hgt_eat',      f"{w['hgt_eat_rate']:.4f}"),
        ]
        table = ''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in rows)
        cards.append(f'<div class="card"><h3>{title}</h3><table>{table}</table></div>')

    return f'<div class="hof">{"".join(cards)}</div>' if cards else ''


if __name__ == '__main__':
    import sys
    import argparse
    import numpy as np
    from pathlib import Path
    from game.snapshot import load_snapshot, SNAPSHOT_PATH
    from sim.stats import StatsCollector

    parser = argparse.ArgumentParser(
        prog='python report.py',
        description='Generate run reports from a beetle-brain snapshot.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python report.py                      html + txt from snapshot.npz
  python report.py old_run.npz          html + txt from a specific snapshot
  python report.py --txt                txt only
  python report.py --html               html only
  python report.py --stdout             print to terminal, no file written
  python report.py old_run.npz --html   html only from specific snapshot
        """.rstrip(),
    )
    parser.add_argument('snapshot', nargs='?', default=SNAPSHOT_PATH,
                        help=f'path to snapshot.npz (default: {SNAPSHOT_PATH})')
    parser.add_argument('--html',   action='store_true', help='write html report only')
    parser.add_argument('--txt',    action='store_true', help='write txt report only')
    parser.add_argument('--stdout', action='store_true',
                        help='print txt to terminal instead of writing a file (implies --txt)')
    args = parser.parse_args()

    # --stdout implies txt only
    if args.stdout:
        args.txt = True

    # default: both
    if not args.html and not args.txt:
        args.html = args.txt = True

    rng   = np.random.default_rng()
    world, tick, history, hall_fame, stats = load_snapshot(rng, path=args.snapshot)

    if world is None:
        print(f"[report] no snapshot found at {args.snapshot}")
        sys.exit(1)

    pop = world['pop']
    if stats is None:
        stats = StatsCollector()
        stats.record(tick, pop, world.get('phylo'))
        stats.finalize(tick, 0.0, pop=pop, phylo_state=world.get('phylo'),
                       seed=world.get('seed'))
    elif not stats.run_meta:
        stats.finalize(tick, 0.0, pop=pop, phylo_state=world.get('phylo'),
                       seed=world.get('seed'))

    stem = _report_stem(stats)

    if args.html:
        generate(stats, path=stem + '.html', world=world, tick=tick, write_txt=False)

    if args.txt:
        if args.stdout:
            generate_text(stats, path='/dev/stdout', world=world, tick=tick)
        else:
            generate_text(stats, path=stem + '.txt', world=world, tick=tick)
