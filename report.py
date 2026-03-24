"""
report.py — generate a self-contained HTML report from a StatsCollector.

    uv run --with plotly --with numpy python report.py   # loads snapshot + re-generates

Or call generate(stats, path) directly from run_headless.py / game/main.py.
"""
import colorsys
import json
import numpy as np


def generate(stats, path="report.html"):
    """Write a self-contained Plotly report to path."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print("[report] plotly not installed — skipping report generation")
        print("         add --with plotly to your uv run command")
        return

    samples = stats.samples
    meta    = stats.run_meta
    hof     = stats.hall_fame

    if not samples:
        print("[report] no samples collected — run was too short")
        return

    ticks = [s['tick']      for s in samples]
    pop   = [s['pop']       for s in samples]

    # ── figure layout ────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Population over time",
            "Trait evolution — size & speed",
            "Sensing evolution — FOV & ray length",
            "Rates — predation, mutation, HGT",
            "Drain breakdown",
            "Final trait snapshot",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # ── 1. population ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ticks, y=pop, mode='lines', name='population',
        line=dict(color='#7ecfff', width=2),
        hovertemplate='tick %{x:,}<br>pop %{y}<extra></extra>',
    ), row=1, col=1)

    max_gen = [s['max_gen'] for s in samples]
    fig.add_trace(go.Scatter(
        x=ticks, y=max_gen, mode='lines', name='max gen',
        line=dict(color='#ffd580', width=1.5, dash='dot'),
        yaxis='y2',
        hovertemplate='tick %{x:,}<br>gen %{y}<extra></extra>',
    ), row=1, col=1)

    # ── 2. size & speed ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['size_mean'] for s in samples],
        mode='lines', name='size (mean)',
        line=dict(color='#ff7eb3', width=2),
        hovertemplate='tick %{x:,}<br>size %{y:.2f}<extra></extra>',
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=ticks,
        y=[s['size_min'] for s in samples],
        fill=None, mode='lines', name='size min',
        line=dict(color='#ff7eb3', width=0.5, dash='dot'),
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=ticks,
        y=[s['size_max'] for s in samples],
        fill='tonexty', mode='lines', name='size range',
        line=dict(color='#ff7eb3', width=0.5, dash='dot'),
        fillcolor='rgba(255,126,179,0.15)',
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['speed_mean'] for s in samples],
        mode='lines', name='speed (mean)',
        line=dict(color='#80ffb4', width=2),
        hovertemplate='tick %{x:,}<br>speed %{y:.2f}<extra></extra>',
    ), row=1, col=2)

    # ── 3. sensing ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['fov_mean'] for s in samples],
        mode='lines', name='FOV° (mean)',
        line=dict(color='#c9b1ff', width=2),
        hovertemplate='tick %{x:,}<br>FOV %{y:.1f}°<extra></extra>',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['fov_min'] for s in samples],
        mode='lines', name='FOV° (min)',
        line=dict(color='#c9b1ff', width=1, dash='dot'),
        hovertemplate='tick %{x:,}<br>FOV min %{y:.1f}°<extra></extra>',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['ray_mean'] for s in samples],
        mode='lines', name='ray len (mean)',
        line=dict(color='#ffa07a', width=2),
        hovertemplate='tick %{x:,}<br>ray %{y:.1f}<extra></extra>',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['ray_min'] for s in samples],
        mode='lines', name='ray len (min)',
        line=dict(color='#ffa07a', width=1, dash='dot'),
        hovertemplate='tick %{x:,}<br>ray min %{y:.1f}<extra></extra>',
    ), row=2, col=1)

    # ── 4. rates ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['pred_ratio_mean'] for s in samples],
        mode='lines', name='pred_ratio',
        line=dict(color='#ff6b6b', width=2),
        hovertemplate='tick %{x:,}<br>pred_ratio %{y:.2f}<extra></extra>',
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['mutation_mean'] for s in samples],
        mode='lines', name='mutation_rate',
        line=dict(color='#aaffaa', width=2),
        hovertemplate='tick %{x:,}<br>mutation %{y:.3f}<extra></extra>',
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['hgt_eat_mean'] for s in samples],
        mode='lines', name='hgt_eat',
        line=dict(color='#ffdd80', width=1.5),
        hovertemplate='tick %{x:,}<br>hgt_eat %{y:.4f}<extra></extra>',
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['hgt_contact_mean'] for s in samples],
        mode='lines', name='hgt_contact',
        line=dict(color='#80ddff', width=1.5),
        hovertemplate='tick %{x:,}<br>hgt_contact %{y:.4f}<extra></extra>',
    ), row=2, col=2)

    # ── 5. drain breakdown ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['drain_mean']   for s in samples],
        mode='lines', name='Kleiber drain',
        line=dict(color='#ff9966', width=2),
        stackgroup='drain',
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=ticks, y=[s['sensing_mean'] * 0.00004 for s in samples],
        mode='lines', name='sensing cost',
        line=dict(color='#9966ff', width=2),
        stackgroup='drain',
    ), row=3, col=1)

    # ── 6. final trait bars ──────────────────────────────────────────────────
    last = samples[-1]
    trait_names = ['size', 'speed', 'FOV°', 'ray', 'pred_ratio', 'mut_rate', 'hgt_eat', 'hgt_contact']
    trait_vals  = [
        last['size_mean'], last['speed_mean'], last['fov_mean'] / 100,
        last['ray_mean'] / 100, last['pred_ratio_mean'],
        last['mutation_mean'], last['hgt_eat_mean'] * 10, last['hgt_contact_mean'] * 100,
    ]
    fig.add_trace(go.Bar(
        x=trait_names, y=trait_vals,
        name='final traits (normalized)',
        marker_color='#7ecfff',
        hovertemplate='%{x}<br>%{y:.4f}<extra></extra>',
    ), row=3, col=2)

    # ── layout ───────────────────────────────────────────────────────────────
    extinct_str = "EXTINCTION" if meta.get('extinct') else f"survived — pop {meta.get('final_pop', '?')}"
    title_str   = (
        f"beetle-brain run report  |  "
        f"{meta.get('ticks', 0):,} ticks  ·  "
        f"{meta.get('elapsed', 0):.1f}s  ·  "
        f"{meta.get('tps', 0):,.0f} ticks/sec  ·  "
        f"{extinct_str}"
    )

    fig.update_layout(
        title=dict(text=title_str, font=dict(size=14, color='#ccc')),
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', family='monospace'),
        legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1, font=dict(size=10)),
        height=1000,
        margin=dict(t=80, b=40, l=60, r=40),
    )
    fig.update_xaxes(gridcolor='#21262d', zerolinecolor='#30363d')
    fig.update_yaxes(gridcolor='#21262d', zerolinecolor='#30363d')

    # ── hall of fame HTML block ───────────────────────────────────────────────
    hof_html = _hof_html(hof)

    # ── write ─────────────────────────────────────────────────────────────────
    chart_html = pio.to_html(fig, full_html=False, include_plotlyjs=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>beetle-brain report</title>
<style>
  body {{ margin: 0; background: #0d1117; color: #c9d1d9; font-family: monospace; }}
  h1   {{ padding: 24px 32px 0; font-size: 1.1em; color: #7ecfff; letter-spacing: 0.05em; }}
  .hof {{ display: flex; gap: 16px; padding: 20px 32px; flex-wrap: wrap; }}
  .card {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 16px 20px; min-width: 200px; flex: 1;
  }}
  .card h3 {{ margin: 0 0 10px; font-size: 0.85em; color: #ffd580; text-transform: uppercase; letter-spacing: 0.08em; }}
  .card table {{ border-collapse: collapse; width: 100%; font-size: 0.8em; }}
  .card td {{ padding: 2px 8px 2px 0; color: #8b949e; }}
  .card td:first-child {{ color: #c9d1d9; }}
  .swatch {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px; vertical-align: middle; margin-right: 6px; }}
  .charts {{ padding: 0 16px 32px; }}
</style>
</head>
<body>
<h1>beetle-brain · run report</h1>
{hof_html}
<div class="charts">{chart_html}</div>
</body>
</html>"""

    with open(path, 'w') as f:
        f.write(html)

    print(f"[report] written → {path}")


def _hof_html(hof):
    cards = []
    labels = {
        'longest': ('longest-lived',  'age'),
        'killer':  ('most kills',     'eaten'),
        'eldest':  ('highest gen',    'generation'),
    }
    for key, (title, highlight) in labels.items():
        w = hof.get(key)
        if w is None:
            continue
        r, g, b = w['r'], w['g'], w['b']
        swatch = f'<span class="swatch" style="background:rgb({r},{g},{b})"></span>'
        rows = [
            ('color',       f'{swatch} rgb({r},{g},{b})'),
            ('age',         f"{w['age']:,} ticks"),
            ('eaten',       str(w['eaten'])),
            ('generation',  str(w['generation'])),
            ('size',        f"{w['size']:.2f}"),
            ('speed',       f"{w['speed']:.2f}"),
            ('FOV',         f"{w['fov_deg']:.1f}°"),
            ('ray_len',     f"{w['ray_len']:.1f}"),
            ('pred_ratio',  f"{w['pred_ratio']:.2f}"),
            ('mut_rate',    f"{w['mutation_rate']:.3f}"),
            ('hgt_eat',     f"{w['hgt_eat_rate']:.4f}"),
        ]
        table = ''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in rows)
        cards.append(f'<div class="card"><h3>{title}</h3><table>{table}</table></div>')

    return f'<div class="hof">{"".join(cards)}</div>'


if __name__ == '__main__':
    # regenerate report from last snapshot
    import numpy as np
    from game.snapshot import load_snapshot
    from sim.stats import StatsCollector

    rng   = np.random.default_rng()
    world, tick, history, hall_fame = load_snapshot(rng)
    if world is None:
        print("no snapshot found")
    else:
        stats = StatsCollector()
        pop   = world['pop']
        if len(pop['x']) > 0:
            stats.record(int(tick), pop)
        stats.finalize(int(tick), 0.0, pop)
        generate(stats)
