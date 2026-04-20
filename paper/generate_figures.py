"""Generate all main-paper figures for the NeurIPS submission.

Accessibility: all figures use distinct marker shapes, line styles, and
light hatching patterns so they remain readable in grayscale and for
readers with color vision deficiency. See Appendix (Accessibility Note).

Figures:
  F1 -- Per-seed Final50 dot plot (4 ablation configs)
  F2 -- Variance ladder + fix paths (7 configs, bar chart of std)
  F3 -- 6-method benchmark on AA (bar chart with error bars)
  F4 -- Cross-layout comparison (AA vs Cramped Room, grouped bars)
  F5 -- Vision cross-domain 2x2 (grouped bars, Best + Final5)

Usage:
    cd ICML/
    python paper/generate_figures.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ── Style: match LaTeX body font ──
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.linewidth': 0.6,
    'patch.linewidth': 0.6,
    'lines.linewidth': 1.5,
})

OUT = Path("paper/figures/neurips")
OUT.mkdir(parents=True, exist_ok=True)

# Load data
p2 = json.load(open("results/canonical_phase2.json"))
cr = json.load(open("results/canonical_phase2_cramped.json"))
bl = json.load(open("results/canonical_phase2_baselines.json"))

# ── Color palette: accessible (colorblind-safe IBM palette) ──
COLORS = {
    'A_full':              '#DC267F',  # magenta (pathology -- stands out)
    'A_no_attn':           '#648FFF',  # blue
    'A_no_aux':            '#785EF0',  # purple
    'A_neither':           '#FE6100',  # orange
    'B_anneal':            '#FFB000',  # gold
    'B_stopgrad':          '#009E73',  # teal
    'B_anneal_stopgrad':   '#56B4E9',  # sky blue
}

# Grayscale-safe markers (distinct shapes per config)
MARKERS = {
    'A_full':    'X',   # x-marker
    'A_no_attn': 's',   # square
    'A_no_aux':  'D',   # diamond
    'A_neither': 'o',   # circle
}

# Light hatching (sparse, not visually heavy)
HATCHES = {
    'A_full':              '//',
    'A_no_attn':           '..',
    'A_no_aux':            '\\\\',
    'A_neither':           'xx',
    'B_anneal':            '--',
    'B_stopgrad':          '++',
    'B_anneal_stopgrad':   'oo',
}


# ============================================================
# F1: Per-seed Final50 dot plot
# ============================================================
def fig1_dotplot():
    # Sized for 0.48\textwidth subfigure placement; fonts enlarged so
    # they remain readable after LaTeX scales the PDF down.
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    configs = ['A_full', 'A_neither', 'A_no_attn', 'A_no_aux']
    labels = ['Full\n(attn+aux)', 'Neither\n(mean only)',
              'No Attn\n(mean+aux)', 'No Aux\n(attn only)']

    for i, cfg in enumerate(configs):
        seeds = p2['configs'][cfg]['final50_per_seed']
        mean = p2['configs'][cfg]['final50_mean']
        color = COLORS[cfg]
        marker = MARKERS[cfg]
        x = np.full(len(seeds), i) + np.random.uniform(-0.08, 0.08, len(seeds))
        ax.scatter(x, seeds, c=color, s=80, zorder=3, alpha=0.85,
                   edgecolors='k', linewidths=0.5, marker=marker)
        ax.hlines(mean, i - 0.22, i + 0.22, colors=color, linewidths=2.2, zorder=4)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel('Final50', fontsize=14)
    ax.grid(axis='y', alpha=0.25, linewidth=0.4)
    ax.set_ylim(445, 480)

    # Annotate pathology (subtle, not screaming red)
    ax.annotate('bimodal,\nhigh variance',
                xy=(0, 451), xytext=(1.5, 448),
                arrowprops=dict(arrowstyle='->', color='#DC267F', lw=1.4),
                fontsize=11, color='#DC267F', ha='center', style='italic')

    fig.savefig(OUT / 'f1_dotplot.png')
    fig.savefig(OUT / 'f1_dotplot.pdf')
    plt.close(fig)
    print(f"  F1 saved")


# ============================================================
# F2: Variance ladder (7 configs)
# ============================================================
def fig2_variance():
    # Sized for 0.48\textwidth subfigure placement; fonts enlarged so
    # they remain readable after LaTeX scales the PDF down.
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    configs = ['A_full', 'A_neither', 'B_stopgrad', 'A_no_attn',
               'B_anneal_stopgrad', 'A_no_aux', 'B_anneal']
    labels = ['Full\n(pathology)', 'Neither', 'Stop-\ngrad', 'No Attn',
              'Anneal+\nSG', 'No Aux', 'Anneal']

    stds = [p2['configs'][c]['final50_std'] for c in configs]
    colors = [COLORS[c] for c in configs]
    hatch_list = [HATCHES[c] for c in configs]

    bars = ax.bar(range(len(configs)), stds, color=colors, edgecolor='k', linewidth=0.5)
    for bar, h in zip(bars, hatch_list):
        bar.set_hatch(h)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel('Cross-seed std of Final50', fontsize=14)
    ax.grid(axis='y', alpha=0.25, linewidth=0.4)

    # Value annotations
    for i, v in enumerate(stds):
        weight = 'bold' if i == 0 else 'normal'
        color = '#DC267F' if i == 0 else '#333333'
        ax.text(i, v + 0.25, f'{v:.1f}', ha='center', fontsize=11,
                fontweight=weight, color=color)

    fig.savefig(OUT / 'f2_variance.png')
    fig.savefig(OUT / 'f2_variance.pdf')
    plt.close(fig)
    print(f"  F2 saved")


# ============================================================
# F3: 6-method benchmark on AA
# ============================================================
def fig3_benchmark():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    methods = [
        ('TarMAC',           'tarmac_asymmetric_advantages',   True),
        ('CommNet',          'commnet_asymmetric_advantages',  True),
        ('MAPPO',            'mappo_asymmetric_advantages',    False),
        ('VABL (best)',      None,                             False),
        ('AERIAL',           'aerial_asymmetric_advantages',   False),
        ('VABL (pathology)', None,                             False),
    ]

    means, stds, colors_list, hatch_list = [], [], [], []
    xlabels = []
    for name, key, comm in methods:
        if key and key in bl['configs']:
            c = bl['configs'][key]
            means.append(c['final50_mean']); stds.append(c['final50_std'])
        elif name == 'VABL (best)':
            c = p2['configs']['A_no_aux']
            means.append(c['final50_mean']); stds.append(c['final50_std'])
        elif name == 'VABL (pathology)':
            c = p2['configs']['A_full']
            means.append(c['final50_mean']); stds.append(c['final50_std'])
        else:
            means.append(0); stds.append(0)

        if comm:
            colors_list.append('#009E73');  hatch_list.append('//')
        elif 'best' in name:
            colors_list.append('#785EF0');  hatch_list.append('\\\\')
        elif 'pathology' in name:
            colors_list.append('#DC267F');  hatch_list.append('//')
        else:
            colors_list.append('#648FFF');  hatch_list.append('')
        xlabels.append(name)

    x = np.arange(len(means))
    bars = ax.bar(x, means, yerr=stds, color=colors_list, edgecolor='k',
                  linewidth=0.5, capsize=3, error_kw={'linewidth': 0.8})
    for bar, h in zip(bars, hatch_list):
        if h:
            bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Final50 (5 seeds)')
    ax.grid(axis='y', alpha=0.25, linewidth=0.4)

    legend_elements = [
        Patch(facecolor='#009E73', hatch='//', edgecolor='k', label='With communication'),
        Patch(facecolor='#648FFF', edgecolor='k', label='No communication'),
        Patch(facecolor='#785EF0', hatch='\\\\', edgecolor='k', label='VABL (our method)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
              framealpha=0.9, edgecolor='#cccccc')

    fig.savefig(OUT / 'f3_benchmark.png')
    fig.savefig(OUT / 'f3_benchmark.pdf')
    plt.close(fig)
    print(f"  F3 saved")


# ============================================================
# F4: Cross-layout comparison
# ============================================================
def fig4_crosslayout():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))

    configs_aa = ['A_full', 'A_no_attn', 'A_no_aux', 'A_neither']
    configs_cr = ['full', 'no_attn', 'no_aux', 'neither']
    labels = ['Full', 'No Attn', 'No Aux', 'Neither']
    colors = [COLORS[c] for c in configs_aa]
    hatch_aa = [HATCHES[c] for c in configs_aa]

    # AA
    means_aa = [p2['configs'][c]['final50_mean'] for c in configs_aa]
    stds_aa = [p2['configs'][c]['final50_std'] for c in configs_aa]
    bars_aa = ax1.bar(range(4), means_aa, yerr=stds_aa, color=colors,
                      edgecolor='k', linewidth=0.5, capsize=3)
    for bar, h in zip(bars_aa, hatch_aa):
        bar.set_hatch(h)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('Final50')
    ax1.set_title('(a) Asymmetric Advantages', fontsize=9)
    ax1.set_ylim(455, 480)
    ax1.grid(axis='y', alpha=0.25, linewidth=0.4)

    # Cramped Room
    means_cr = [cr['configs'][c]['final50_mean'] for c in configs_cr]
    stds_cr = [cr['configs'][c]['final50_std'] for c in configs_cr]
    bars_cr = ax2.bar(range(4), means_cr, yerr=stds_cr, color=colors,
                      edgecolor='k', linewidth=0.5, capsize=3)
    for bar, h in zip(bars_cr, hatch_aa):
        bar.set_hatch(h)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_title('(b) Cramped Room', fontsize=9)
    ax2.set_ylim(405, 420)
    ax2.grid(axis='y', alpha=0.25, linewidth=0.4)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / 'f4_crosslayout.png')
    fig.savefig(OUT / 'f4_crosslayout.pdf')
    plt.close(fig)
    print(f"  F4 saved")


# ============================================================
# F5: Vision cross-domain 2x2
# ============================================================
def fig5_vision():
    configs = {
        'A_full':    {'best': 43.90, 'final5': 42.89},
        'A_no_aux':  {'best': 43.47, 'final5': 42.96},
        'A_no_attn': {'best': 48.02, 'final5': 47.34},
        'A_neither': {'best': 48.06, 'final5': 47.08},
    }

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    labels = ['Full\n(ViT+aux)', 'No Aux\n(ViT)', 'No Attn\n(pool+aux)', 'Neither\n(pool)']
    order = ['A_full', 'A_no_aux', 'A_no_attn', 'A_neither']
    x = np.arange(len(order))
    width = 0.35
    colors = [COLORS[c] for c in order]
    hatch_list = [HATCHES[c] for c in order]

    bests = [configs[c]['best'] for c in order]
    finals = [configs[c]['final5'] for c in order]

    bars_best = ax.bar(x - width/2, bests, width, label='Best',
                       color=colors, edgecolor='k', linewidth=0.5, alpha=0.5)
    bars_final = ax.bar(x + width/2, finals, width, label='Final5',
                        color=colors, edgecolor='k', linewidth=0.5)
    for bar, h in zip(bars_best, hatch_list):
        bar.set_hatch(h)
    for bar, h in zip(bars_final, hatch_list):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Test accuracy (%)')
    ax.legend(fontsize=8, framealpha=0.9, edgecolor='#cccccc')
    ax.grid(axis='y', alpha=0.25, linewidth=0.4)
    ax.set_ylim(40, 50)

    fig.savefig(OUT / 'f5_vision.png')
    fig.savefig(OUT / 'f5_vision.pdf')
    plt.close(fig)
    print(f"  F5 saved")


# ============================================================
# F6: Gradient decomposition (3 panels, labels below)
# ============================================================
def fig6_gradients():
    # Seed 0 only (cleaner than overlaying all 5); caption already notes
    # the pattern is consistent across all 5 seeds with cosine range
    # [-0.47, +0.33].
    d = json.load(open("results/celestia_pull/gradient_diagnostics.json"))['diagnostic']

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax_mag, ax_cos, ax_rat = axes

    # Panel (a): PPO and aux gradient magnitudes
    ax_mag.plot(d['iterations'], d['norm_ppo'], color='#648FFF',
                marker='o', markersize=7, linewidth=2.0,
                label=r'$\|g_{\mathrm{PPO}}\|$')
    ax_mag.plot(d['iterations'], d['norm_aux'], color='#DC267F',
                marker='s', markersize=7, linewidth=2.0,
                label=r'$\|g_{\mathrm{aux}}\|$')
    ax_mag.set_xlabel('Training iteration', fontsize=12)
    ax_mag.set_ylabel('Gradient norm', fontsize=12)
    ax_mag.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='#cccccc')
    ax_mag.grid(alpha=0.25, linewidth=0.4)
    ax_mag.tick_params(labelsize=11)

    # Panel (b): cosine similarity
    ax_cos.plot(d['iterations'], d['cosine_sim'], color='#785EF0',
                marker='D', markersize=7, linewidth=2.0)
    ax_cos.axhline(0, color='k', linewidth=0.6, linestyle='--', alpha=0.5)
    ax_cos.set_xlabel('Training iteration', fontsize=12)
    ax_cos.set_ylabel(r'$\cos(g_{\mathrm{PPO}}, g_{\mathrm{aux}})$', fontsize=12)
    ax_cos.grid(alpha=0.25, linewidth=0.4)
    ax_cos.tick_params(labelsize=11)
    ax_cos.set_ylim(-0.55, 0.45)

    # Panel (c): magnitude ratio
    ax_rat.plot(d['iterations'], d['grad_ratio'], color='#009E73',
                marker='^', markersize=7, linewidth=2.0)
    ax_rat.axhline(0.25, color='k', linewidth=0.6, linestyle='--', alpha=0.5)
    ax_rat.set_xlabel('Training iteration', fontsize=12)
    ax_rat.set_ylabel(r'$\|g_{\mathrm{aux}}\| / \|g_{\mathrm{PPO}}\|$', fontsize=12)
    ax_rat.grid(alpha=0.25, linewidth=0.4)
    ax_rat.tick_params(labelsize=11)
    ax_rat.set_ylim(0, 0.3)

    # Panel labels BELOW each subplot (after tight_layout so they don't get clipped)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    panel_labels = [
        '(a) Gradient magnitudes',
        r'(b) Cosine similarity',
        '(c) Magnitude ratio',
    ]
    for ax, lbl in zip(axes, panel_labels):
        ax.text(0.5, -0.33, lbl, transform=ax.transAxes,
                ha='center', va='top', fontsize=13, fontweight='bold')

    fig.savefig(OUT / 'f6_gradients.png')
    fig.savefig(OUT / 'f6_gradients.pdf')
    plt.close(fig)
    print(f"  F6 saved")


# ============================================================
if __name__ == '__main__':
    print("Generating NeurIPS figures (accessible, no matplotlib titles)...")
    np.random.seed(42)
    fig1_dotplot()
    fig2_variance()
    fig3_benchmark()
    fig4_crosslayout()
    fig5_vision()
    fig6_gradients()
    print(f"All figures saved to {OUT}/")
