"""
report.py — generate a self-contained HTML report from a StatsCollector.

    uv run --with plotly --with numpy python report.py   # regenerate from snapshot

Or call generate(stats, path) directly from run_headless.py / game/main.py.
"""
import colorsys


def _hue_to_rgb_css(hue, s=0.85, v=0.9):
    r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def _hue_to_rgba_css(hue, alpha, s=0.85, v=0.9):
    r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


def generate(stats, path="report.html"):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print("[report] plotly not installed — skipping")
        print("         add --with plotly to your uv run command")
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

    figs.append(_fig_lineage_river(stats, ticks))
    figs.append(_fig_population(samples, ticks))
    figs.append(_fig_traits(samples, ticks))
    figs.append(_fig_sensing(samples, ticks))
    figs.append(_fig_rates(samples, ticks))
    figs.append(_fig_drain(samples, ticks))
    figs.append(_fig_genome_heatmap(samples, ticks))
    figs.append(_fig_phase_scatter(samples))

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
    summary_html = f"""
    <div class="summary">
      <span class="metric"><b>{meta.get('ticks',0):,}</b> ticks</span>
      <span class="metric"><b>{meta.get('elapsed',0):.1f}s</b> elapsed</span>
      <span class="metric"><b>{meta.get('tps',0):,.0f}</b> ticks/sec</span>
      <span class="metric"><b>{meta.get('final_max_gen',0)}</b> max gen</span>
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

<div class="section-title">lineage river</div>
<div class="hof" style="padding-top:0">{_lineage_legend_html(stats)}</div>

<div class="charts">
{"".join(chart_blocks)}
</div>

<div class="section-title">hall of fame</div>
{hof_html}

</body>
</html>"""

    with open(path, 'w') as f:
        f.write(html)
    print(f"[report] written → {path}")


# ── individual figures ────────────────────────────────────────────────────────

def _fig_lineage_river(stats, ticks):
    import plotly.graph_objects as go

    series = stats._lineage_series
    hues   = stats._lineage_hues
    if not series:
        return None

    # sort by total count descending so dominant lineages are at the bottom
    sorted_ids = sorted(series.keys(), key=lambda uid: -sum(c for _, c in series[uid]))

    # build a dense tick→count map for each lineage
    tick_set = sorted(set(ticks))
    tick_idx = {t: i for i, t in enumerate(tick_set)}

    fig = go.Figure()
    for uid in sorted_ids:
        hue   = hues.get(uid, 0.0)
        color = _hue_to_rgb_css(hue)
        fill_color = _hue_to_rgba_css(hue, 0.75)

        y = [0] * len(tick_set)
        for t, cnt in series[uid]:
            if t in tick_idx:
                y[tick_idx[t]] = cnt

        fig.add_trace(go.Scatter(
            x=tick_set, y=y,
            mode='lines',
            name=f'lineage {uid}',
            stackgroup='river',
            line=dict(width=0.5, color=color),
            fillcolor=fill_color,
            hovertemplate=f'lineage {uid}<br>tick %{{x:,}}<br>count %{{y}}<extra></extra>',
        ))

    fig.update_layout(
        title=dict(text='lineage river — population share over time', font=dict(size=12, color='#8b949e')),
        height=280,
        showlegend=False,
        xaxis_title='tick',
        yaxis_title='wights',
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
        title=dict(text='sensing evolution — FOV & ray length (blind wights → zero)', font=dict(size=12, color='#8b949e')),
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
        title=dict(text='genome heatmap — all 18 genes, normalized 0→1 within range', font=dict(size=12, color='#8b949e')),
        height=380,
        yaxis=dict(autorange='reversed', tickfont=dict(size=10)),
        xaxis_title='tick',
    )
    return fig


def _fig_phase_scatter(samples):
    import plotly.graph_objects as go

    if not samples:
        return None
    last = samples[-1]
    sizes  = last.get('size_all', [])
    speeds = last.get('speed_all', [])
    preds  = last.get('pred_ratio_all', [])
    hues   = last.get('lineage_hues_all', [])

    if not sizes:
        return None

    colors = [_hue_to_rgb_css(h) for h in hues]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=preds,
        mode='markers',
        name='wights',
        marker=dict(
            color=colors,
            size=[max(6, s * 1.5) for s in speeds],
            opacity=0.85,
            line=dict(width=0.5, color='rgba(255,255,255,0.2)'),
        ),
        hovertemplate='size %{x:.2f}<br>pred_ratio %{y:.2f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(
            text='phase space — size vs pred_ratio at last sample (color=lineage, marker size=speed)',
            font=dict(size=12, color='#8b949e'),
        ),
        height=320,
        xaxis_title='size',
        yaxis_title='pred_ratio',
    )
    return fig


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _hof_html(hof):
    cards = []
    labels = {
        'longest': ('longest-lived',  'age'),
        'killer':  ('most kills',     'eaten'),
        'eldest':  ('highest gen',    'generation'),
    }
    for key, (title, _) in labels.items():
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
            ('kills',        str(w['eaten'])),
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


def _lineage_legend_html(stats):
    if not stats._lineage_hues:
        return ''
    # show top 12 lineages by total count
    totals = {uid: sum(c for _, c in series) for uid, series in stats._lineage_series.items()}
    top    = sorted(totals, key=lambda u: -totals[u])[:12]
    items  = []
    for uid in top:
        hue = stats._lineage_hues.get(uid, 0.0)
        css = _hue_to_rgb_css(hue)
        items.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:12px;font-size:0.75em;color:#8b949e">'
            f'<span style="width:12px;height:12px;border-radius:3px;background:{css};display:inline-block"></span>'
            f'lineage {uid} ({totals[uid]})</span>'
        )
    return ''.join(items)


if __name__ == '__main__':
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
            stats.record(int(tick), pop, world['phylo'])
        stats.finalize(int(tick), 0.0, pop, world['phylo'])
        generate(stats)
