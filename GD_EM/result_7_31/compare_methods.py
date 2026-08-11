"""
Compare the three observation regimes run in THIS directory, all under the
current (updated) transition dynamics and all four progressive-relaxation
stages (G / G,a / G,a,q / G,a,q,c):

    results_stage*_Gtrue*.csv        -- states X only (MajorAgentEstimator)
    results_u_stage*_Gtrue*.csv      -- per-agent control u also observed (UObservedEstimator)
    results_u_bar_stage*_Gtrue*.csv  -- only the AGGREGATE minor control u_bar observed (UBarObservedEstimator)

Unlike compare_dynamics.py (which compares two DIFFERENT directories with
different stage sets and has to intersect them), all three methods here were
run from the same experiment.py STAGES definition in the same directory, so
every (stage, G_true) config is present for all three methods -- no
intersection/skipping logic needed. This script still checks that and warns
if a file is unexpectedly missing rather than silently dropping a config.

Writes (into this directory):
    comparison_summary.csv
    comparison_accuracy.png
    comparison_mean_steps.png
    comparison_parameter_estimates.png

Usage (run from inside result_7_31/, or pass --results-dir):
    python compare_methods.py
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# validated categorical palette (dataviz skill, references/palette.md) --
# same family compare_dynamics.py / summarize_results.py draw from.
CONFIG_COLORS = {
    'G':         "#2a78d6",  # blue
    'G|a':       "#008300",  # green
    'G|a|q':     "#e87ba4",  # magenta
    'G|a|q|c':   "#eda100",  # orange
}
CONFIG_LABELS = {
    'G':       'G',
    'G|a':     'G, a',
    'G|q':     'G, q',
    'G|a|q':   'G, a, q',
    'G|a|q|c': 'G, a, q, c',
}
METHOD_FILE_PREFIX = {
    'x':     'results_stage',       # states only
    'u_bar': 'results_u_bar_stage', # aggregate minor control observed
    'u':     'results_u_stage',     # per-agent control observed
}
METHOD_COLORS = {
    'x':     "#2a78d6",  # blue
    'u_bar': "#eda100",  # orange
    'u':     "#008300",  # green
}
METHOD_STYLE = {
    'x':     dict(linestyle='-',  marker='o'),
    'u_bar': dict(linestyle='--', marker='s'),
    'u':     dict(linestyle=':',  marker='^'),
}
METHOD_LABELS = {
    'x':     'X only',
    'u_bar': 'u_bar observed',
    'u':     'u observed',
}
METHOD_ORDER = ['x', 'u_bar', 'u']
PARAM_ORDER = ['G', 'a', 'q', 'c']


def _fit_title_fontsize(title, fig_width_in, base=22, min_size=13):
    """Bold suptitle fontsize that fits the ACTUAL figure width -- a fixed
    fontsize (tuned for the wide 4-column default grid) silently overflows
    off the left/right edges once a narrower subset (e.g. --rows/--params
    down to a 2-column grid) shrinks the figure. ~0.0083 in/char/pt is a
    rough calibration for bold sans-serif; good enough to stay a bit
    conservative rather than exactly tight."""
    return max(min_size, min(base, fig_width_in / (len(title) * 0.0083)))


def load_method(results_dir, method):
    pattern = os.path.join(results_dir, f"{METHOD_FILE_PREFIX[method]}*_Gtrue*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files matching {pattern} -- has this method been run yet?")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df['method'] = method
    print(f"[{METHOD_LABELS[method]}] loaded {len(paths)} files, {len(df)} seed-rows")
    return df


def summarize(df):
    rows = []
    for (unknown_params, G_true, method), g in df.groupby(['unknown_params', 'G_true', 'method']):
        unknown = unknown_params.split('|')
        row = {
            'unknown_params': unknown_params,
            'G_true': G_true,
            'method': method,
            'n_seeds': len(g),
            'accuracy': g['correct'].mean(),
            'mean_steps': g['n_steps'].mean(),
            'mean_wall_time': g['wall_time_sec'].mean(),
        }
        for p in unknown:
            fit_col, true_col = f"{p}_fit", f"{p}_true"
            if fit_col in g.columns:
                true_val = g[true_col].iloc[0] if true_col in g.columns else np.nan
                if pd.isna(true_val) and p == 'G':
                    true_val = g['G_true'].iloc[0]
                err = g[fit_col] - true_val
                row[f"{p}_rmse"] = float(np.sqrt((err ** 2).mean()))
                row[f"{p}_bias"] = float(err.mean())
                row[f"{p}_mean_fit"] = float(g[fit_col].mean())
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ['unknown_params', 'G_true', 'method'],
        key=lambda col: col.map(lambda u: len(u.split('|'))) if col.name == 'unknown_params' else col,
    ).reset_index(drop=True)
    return summary


def _plot_summary_metric(summary, column, ylabel, title, out_path, ylim=None):
    """Small multiples: one subplot per config (2x2 grid), each showing all
    three methods as separate lines (color = method). Replaces the earlier
    single-axes version (one line per config x method, 12 lines total,
    needing two legends to stay legible) -- with method now encoded purely
    by color/linestyle within each subplot, only one small legend is needed
    for the whole figure, and each config's own accuracy/steps curve is much
    easier to read in isolation."""
    configs = sorted(summary['unknown_params'].unique(), key=lambda u: len(u.split('|')))
    ncols = 2
    nrows = -(-len(configs) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.8 * nrows), squeeze=False, sharey=True)

    for i, cfg in enumerate(configs):
        ax = axes[i // ncols][i % ncols]
        for method in METHOD_ORDER:
            s = summary[(summary['unknown_params'] == cfg) & (summary['method'] == method)].sort_values('G_true')
            if s.empty:
                continue
            style = METHOD_STYLE[method]
            ax.plot(s['G_true'], s[column], color=METHOD_COLORS[method], linewidth=2.5,
                     markersize=9, label=METHOD_LABELS[method], **style)
        ax.set_title(f"unknown = {CONFIG_LABELS[cfg]}", fontsize=15, fontweight='bold')
        ax.set_xlabel('G (true)', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis='both', labelsize=12)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)

    for j in range(len(configs), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig_width_in = 6.5 * ncols
    fig.suptitle(title, y=0.995, fontsize=_fit_title_fontsize(title, fig_width_in), fontweight='bold')
    fig.legend(handles, labels, loc='upper center', ncol=len(METHOD_ORDER),
               bbox_to_anchor=(0.5, 0.93), frameon=False, fontsize=14, markerscale=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_accuracy(summary, out_path):
    _plot_summary_metric(
        summary, 'accuracy', 'Accuracy',
        'Detection accuracy: X only vs. u_bar observed vs. u observed',
        out_path, ylim=(-0.05, 1.05),
    )


def plot_mean_steps(summary, out_path):
    _plot_summary_metric(
        summary, 'mean_steps', 'Mean EM iterations',
        'Convergence speed: X only vs. u_bar observed vs. u observed',
        out_path,
    )


def plot_parameter_estimates(df, out_path, params=None, configs=None):
    """
    One row per selected config, one column per selected parameter -- a cell
    is populated only if that config actually estimates that parameter.
    Each populated cell overlays all three methods' per-seed fitted values
    (colored by method) plus the true-value reference curve.

    params  : which parameters to show as columns -- subset of PARAM_ORDER.
              Default (None): every parameter in PARAM_ORDER.
    configs : which configs to show as rows -- subset of the unknown_params
              strings present in df. Default (None): every config present.
    """
    available_configs = sorted(df['unknown_params'].unique(), key=lambda u: len(u.split('|')))
    if configs is None:
        configs = available_configs
    else:
        bad = [c for c in configs if c not in available_configs]
        if bad:
            raise ValueError(f"config(s) not present in data: {bad}  (available: {available_configs})")
        configs = sorted(configs, key=lambda u: len(u.split('|')))

    if params is None:
        params = PARAM_ORDER
    else:
        bad = [p for p in params if p not in PARAM_ORDER]
        if bad:
            raise ValueError(f"unrecognized parameter(s): {bad}  (available: {PARAM_ORDER})")

    # drop any (row, col) pair that would be entirely blank BEFORE sizing the
    # figure -- e.g. selecting rows=[G, G|a] with the default params=all 4
    # left 'q' and 'c' as two fully-empty columns, wasting half the figure
    # and making the actual data shrink to a corner once embedded in a slide.
    params = [p for p in params if any(p in cfg.split('|') for cfg in configs)]
    configs = [c for c in configs if any(p in c.split('|') for p in params)]

    nrows, ncols = len(configs), len(params)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)

    jitter_for = {'x': -0.012, 'u_bar': 0.0, 'u': 0.012}
    legend_handles, legend_labels = [], []
    for r, cfg in enumerate(configs):
        unknown = cfg.split('|')
        sub_cfg = df[df['unknown_params'] == cfg]
        for c, p in enumerate(params):
            ax = axes[r][c]
            if p not in unknown:
                ax.axis('off')
                continue
            fit_col = f'{p}_fit'
            true_col = 'G_true' if p == 'G' else f'{p}_true'

            for method in METHOD_ORDER:
                s = sub_cfg[sub_cfg['method'] == method]
                if s.empty or fit_col not in s.columns:
                    continue
                sc = ax.scatter(s['G_true'] + jitter_for[method], s[fit_col], color=METHOD_COLORS[method],
                                 s=26, alpha=0.55, linewidths=0, label=METHOD_LABELS[method])
                if METHOD_LABELS[method] not in legend_labels:
                    legend_handles.append(sc)
                    legend_labels.append(METHOD_LABELS[method])

            if p == 'G':
                g_vals = sorted(sub_cfg['G_true'].unique())
                ax.plot(g_vals, g_vals, color='#52514e', linewidth=1.8, linestyle=':', zorder=0)
            elif true_col in sub_cfg.columns:
                ref = sub_cfg[['G_true', true_col]].drop_duplicates().sort_values('G_true')
                ax.plot(ref['G_true'], ref[true_col], color='#52514e', linewidth=1.8, linestyle=':', zorder=0)

            ax.set_title(f"{p}  (unknown = {CONFIG_LABELS[cfg]})", fontsize=15, fontweight='bold')
            ax.set_xlabel('G (true)', fontsize=13)
            ax.set_ylabel(f'{p} estimate', fontsize=13)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True, alpha=0.25)

    # reserve a roughly constant ABSOLUTE inch margin (title + legend row) at
    # the top, rather than a fixed fraction -- a fraction tuned for a tall
    # grid leaves too little absolute room once nrows/ncols shrink.
    fig_height = 4.6 * nrows
    top_margin_in = 1.15
    top_frac = max(0.6, 1 - top_margin_in / fig_height)
    fig.tight_layout(rect=(0, 0, 1, top_frac))
    title = 'Parameter recovery: X only vs. u_bar observed vs. u observed'
    fig_width_in = 5.2 * ncols
    fig.suptitle(title, y=1 - 0.12 * (1 - top_frac),
                 fontsize=_fit_title_fontsize(title, fig_width_in), fontweight='bold')
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper center', ncol=3,
                   bbox_to_anchor=(0.5, 1 - 0.55 * (1 - top_frac)), frameon=False, fontsize=14,
                   markerscale=1.8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', default='.',
                         help="directory containing results_stage*/_u_stage*/_u_bar_stage* csvs "
                              "(default: current directory, i.e. run this from inside result_7_31/)")
    parser.add_argument('--params', default=None,
                         help=f"comma-separated columns for the parameter-estimate plot, "
                              f"subset of {PARAM_ORDER} (default: all)")
    parser.add_argument('--rows', default=None,
                         help="comma-separated rows (configs) for the parameter-estimate plot, "
                              "each a '|'-joined unknown-params string, e.g. 'G,G|a|q' "
                              "(default: all configs)")
    args = parser.parse_args()
    params = args.params.split(',') if args.params else None
    row_configs = args.rows.split(',') if args.rows else None

    dfs = {m: load_method(args.results_dir, m) for m in METHOD_ORDER}

    config_sets = {m: set(d['unknown_params'].unique()) for m, d in dfs.items()}
    all_common = set.intersection(*config_sets.values())
    for m, cfgs in config_sets.items():
        extra = cfgs - all_common
        if extra:
            print(f"  NOTE: {METHOD_LABELS[m]} has config(s) not shared by all methods: {sorted(extra)}")

    df = pd.concat(dfs.values(), ignore_index=True)

    summary = summarize(df)

    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', None)
    print()
    print(summary.to_string(index=False))

    out_summary = os.path.join(args.results_dir, 'comparison_summary.csv')
    summary.to_csv(out_summary, index=False)
    print(f"\nsaved {out_summary}")

    plot_accuracy(summary, os.path.join(args.results_dir, 'comparison_accuracy.png'))
    plot_mean_steps(summary, os.path.join(args.results_dir, 'comparison_mean_steps.png'))
    plot_parameter_estimates(
        df, os.path.join(args.results_dir, 'comparison_parameter_estimates.png'),
        params=params, configs=row_configs,
    )


if __name__ == "__main__":
    main()
