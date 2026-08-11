"""
Compare the UPDATED transition-dynamics experiment results (result_7_31/,
mfg.py's current state_transtition) against the PREVIOUS-dynamics results
(result_new/) -- same progressive-relaxation experiment (experiment.py /
run_experiment.sbatch), run under two different versions of the mean-field
state-transition logic.

The two runs used DIFFERENT stage numbering (result_7_31's stage 2 is
"G, a" while result_new's "G, a" is stage 5; result_new has extra a_0/q_0
stages result_7_31 never ran; result_7_31 has a "G, a, q, c" stage result_new
never ran) -- so configs are matched by the SET of unknown parameters itself
(the unknown_params column, e.g. "G", "G|a", "G|a|q"), not by stage number.
Only configs present in BOTH directories are compared; everything else is
reported as skipped, not silently dropped.

Writes:
    comparison_summary.csv
    comparison_accuracy.png
    comparison_mean_steps.png
    comparison_parameter_estimates.png

Usage:
    python compare_dynamics.py
    python compare_dynamics.py --updated-dir result_7_31 --previous-dir result_new
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# validated categorical palette (dataviz skill, references/palette.md) --
# same palette summarize_results.py draws STAGE_COLORS from.
CONFIG_COLORS = {
    'G':       "#2a78d6",  # blue
    'G|a':     "#008300",  # green
    'G|a|q':   "#e87ba4",  # magenta
}
CONFIG_LABELS = {
    'G':     'G',
    'G|a':   'G, a',
    'G|a|q': 'G, a, q',
}
DYNAMICS_COLORS = {
    'updated':  "#2a78d6",
    'previous': "#eda100",
}
DYNAMICS_STYLE = {
    'updated':  dict(linestyle='-',  marker='o'),
    'previous': dict(linestyle='--', marker='s'),
}
DYNAMICS_LABELS = {
    'updated':  'updated dynamics',
    'previous': 'previous dynamics',
}
PARAM_ORDER = ['G', 'a', 'q']


def load_dir(results_dir, dynamics_label):
    pattern = os.path.join(results_dir, "results_stage*_Gtrue*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no results_stage*_Gtrue*.csv files found in {results_dir}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df['dynamics'] = dynamics_label
    print(f"[{dynamics_label}] loaded {len(paths)} files, {len(df)} seed-rows from {results_dir}/")
    return df


def summarize(df):
    rows = []
    for (unknown_params, G_true, dynamics), g in df.groupby(['unknown_params', 'G_true', 'dynamics']):
        unknown = unknown_params.split('|')
        row = {
            'unknown_params': unknown_params,
            'G_true': G_true,
            'dynamics': dynamics,
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
        ['unknown_params', 'G_true', 'dynamics'],
        key=lambda col: col.map(lambda u: len(u.split('|'))) if col.name == 'unknown_params' else col,
    ).reset_index(drop=True)
    return summary


def _plot_summary_metric(summary, column, ylabel, title, out_path, ylim=None):
    """Small multiples: one subplot per config (2-column grid), each showing
    both dynamics as separate lines (color = dynamics). Replaces the earlier
    single-axes version (one line per config x dynamics, needing two
    legends to stay legible) -- with dynamics now encoded purely by
    color/linestyle within each subplot, only one small legend is needed
    for the whole figure, and each config's own curve is easier to read in
    isolation."""
    configs = sorted(summary['unknown_params'].unique(), key=lambda u: len(u.split('|')))
    ncols = 2
    nrows = -(-len(configs) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.8 * nrows), squeeze=False, sharey=True)

    for i, cfg in enumerate(configs):
        ax = axes[i // ncols][i % ncols]
        for dyn in ['updated', 'previous']:
            s = summary[(summary['unknown_params'] == cfg) & (summary['dynamics'] == dyn)].sort_values('G_true')
            if s.empty:
                continue
            style = DYNAMICS_STYLE[dyn]
            ax.plot(s['G_true'], s[column], color=DYNAMICS_COLORS[dyn], linewidth=2,
                     markersize=7, label=DYNAMICS_LABELS[dyn], **style)
        ax.set_title(f"unknown = {CONFIG_LABELS[cfg]}")
        ax.set_xlabel('G (true)')
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)

    for j in range(len(configs), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(title, y=0.995)
    fig.legend(handles, labels, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 0.965), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_accuracy(summary, out_path):
    _plot_summary_metric(
        summary, 'accuracy', 'Accuracy (major agent correctly identified)',
        'Detection accuracy: updated vs. previous transition dynamics',
        out_path, ylim=(-0.05, 1.05),
    )


def plot_mean_steps(summary, out_path):
    _plot_summary_metric(
        summary, 'mean_steps', 'Mean EM iterations to convergence / early-stop',
        'Convergence speed: updated vs. previous transition dynamics',
        out_path,
    )


def plot_parameter_estimates(df, out_path, params=None, configs=None):
    """
    One row per selected config (e.g. G / G,a / G,a,q), one column per
    selected parameter (e.g. G, a, q) -- a cell is populated only if that
    config actually estimates that parameter (e.g. config "G" only has a
    populated G column). Each populated cell overlays both dynamics'
    per-seed fitted values (colored by dynamics) plus the true-value
    reference curve.

    params  : which parameters to show as columns -- subset of PARAM_ORDER
              (e.g. ['a', 'q']). Default (None): every parameter in
              PARAM_ORDER.
    configs : which configs to show as rows -- subset of the unknown_params
              strings present in df (e.g. ['G', 'G|a|q']). Default (None):
              every config present in df.
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

    nrows, ncols = len(configs), len(params)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.5 * nrows), squeeze=False)

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

            for dyn in ['updated', 'previous']:
                s = sub_cfg[sub_cfg['dynamics'] == dyn]
                if s.empty or fit_col not in s.columns:
                    continue
                jitter = 0.006 if dyn == 'updated' else -0.006
                sc = ax.scatter(s['G_true'] + jitter, s[fit_col], color=DYNAMICS_COLORS[dyn],
                                 s=16, alpha=0.5, linewidths=0, label=DYNAMICS_LABELS[dyn])
                if DYNAMICS_LABELS[dyn] not in legend_labels:
                    legend_handles.append(sc)
                    legend_labels.append(DYNAMICS_LABELS[dyn])

            if p == 'G':
                g_vals = sorted(sub_cfg['G_true'].unique())
                ax.plot(g_vals, g_vals, color='#52514e', linewidth=1.3, linestyle=':', zorder=0)
            elif true_col in sub_cfg.columns:
                ref = sub_cfg[['G_true', true_col]].drop_duplicates().sort_values('G_true')
                ax.plot(ref['G_true'], ref[true_col], color='#52514e', linewidth=1.3, linestyle=':', zorder=0)

            ax.set_title(f"{p}  (unknown = {CONFIG_LABELS[cfg]})", fontsize=9)
            ax.set_xlabel('G (true)')
            ax.set_ylabel(f'{p} estimate')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid(True, alpha=0.2)

    # reserve a roughly constant ABSOLUTE inch margin (title + legend row) at
    # the top, rather than a fixed fraction -- a fixed fraction (tuned for a
    # tall multi-row grid) leaves far too little absolute room once nrows
    # shrinks (e.g. selecting just one config via `configs=`).
    fig_height = 15 * nrows
    top_margin_in = 1.0
    top_frac = max(0.6, 1 - top_margin_in / fig_height)
    fig.tight_layout(rect=(0, 0, 1, top_frac))
    fig.suptitle('Parameter recovery: updated vs. previous transition dynamics',
                 y=1 - 0.15 * (1 - top_frac))
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper center', ncol=2,
                   bbox_to_anchor=(0.5, 1 - 0.55 * (1 - top_frac)), frameon=False, fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--updated-dir', default='result_7_31',
                         help="results from the UPDATED transition dynamics")
    parser.add_argument('--previous-dir', default='result_new',
                         help="results from the PREVIOUS transition dynamics")
    parser.add_argument('--params', default=None,
                         help="comma-separated columns for the parameter-estimate plot, "
                              f"subset of {PARAM_ORDER} (default: all)")
    parser.add_argument('--rows', default=None,
                         help="comma-separated rows (configs) for the parameter-estimate plot, "
                              "each a '|'-joined unknown-params string, e.g. 'G,G|a|q' "
                              "(default: all matched configs)")
    args = parser.parse_args()
    params = args.params.split(',') if args.params else None
    row_configs = args.rows.split(',') if args.rows else None

    df_updated = load_dir(args.updated_dir, 'updated')
    df_previous = load_dir(args.previous_dir, 'previous')

    updated_configs = set(df_updated['unknown_params'].unique())
    previous_configs = set(df_previous['unknown_params'].unique())
    common = sorted(updated_configs & previous_configs, key=lambda u: len(u.split('|')))
    only_updated = sorted(updated_configs - previous_configs)
    only_previous = sorted(previous_configs - updated_configs)

    print(f"\ncomparable configs (present in both): {common}")
    if only_updated:
        print(f"  skipping (updated-only, no previous-dynamics counterpart): {only_updated}")
    if only_previous:
        print(f"  skipping (previous-only, no updated-dynamics counterpart): {only_previous}")
    if not common:
        raise SystemExit("no configs in common between the two result directories -- nothing to compare")

    df = pd.concat([
        df_updated[df_updated['unknown_params'].isin(common)],
        df_previous[df_previous['unknown_params'].isin(common)],
    ], ignore_index=True)

    summary = summarize(df)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    print()
    print(summary.to_string(index=False))

    summary.to_csv('comparison_summary.csv', index=False)
    print("\nsaved comparison_summary.csv")

    plot_accuracy(summary, 'comparison_accuracy.png')
    plot_mean_steps(summary, 'comparison_mean_steps.png')
    plot_parameter_estimates(df, 'comparison_parameter_estimates.png', params=params, configs=row_configs)


if __name__ == "__main__":
    main()
