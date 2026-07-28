"""
Summarize the progressive-relaxation experiment's per-seed result files
(written by experiment.py / run_experiment.sbatch) into one summary table
and a headline chart.

Reads every results_stage{K}_{name}_Gtrue{G}.csv in --results-dir (default:
current directory), and for each (stage, G_true) config computes:
    n_seeds        -- how many seeds' results were found
    accuracy       -- fraction with the correct major agent detected
    mean_steps     -- mean EM iterations to convergence/early-stop
    mean_wall_time -- mean wall-clock seconds per seed
    {param}_rmse, {param}_bias, {param}_mean_fit
                   -- for every parameter unknown in that stage (NaN for
                      parameters not estimated in that stage, same
                      convention as the earlier comparison_*.txt tables)

Writes:
    summary.csv                        -- the table above
    summary_accuracy.png               -- accuracy vs G_true, one line per stage
    summary_parameter_estimates.png    -- one panel per estimated parameter,
                                           every seed's fit vs G_true, colored
                                           by stage, true value overlaid

Usage:
    python summarize_results.py
    python summarize_results.py --results-dir /path/to/pulled/results
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# validated categorical palette (dataviz skill, references/palette.md),
# first 4 slots -- passes the adjacent-pair CVD/contrast checks for a
# small multi-line chart.
STAGE_COLORS = {
    1: "#2a78d6",  # blue
    2: "#008300",  # green
    3: "#e87ba4",  # magenta
    4: "#eda100",  # yellow
    5: "#a0a0a0",  # gray
    6: "#d6a200",  # orange
}
STAGE_LABELS = {
    1: "G",
    2: "G, a, a₀",
    3: "G, a, a₀, q, q₀",
    4: "G, a, a₀, q, q₀, c, c₀",
    5: "G, a",
    6: "G, a, q",
}


def load_all(results_dir):
    pattern = "results_stage*_Gtrue*.csv"
    paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not paths:
        # fall back to a results/ subdirectory of results_dir (the layout
        # experiment.py's sbatch runs have been using on the cluster)
        fallback = os.path.join(results_dir, "results")
        paths = sorted(glob.glob(os.path.join(fallback, pattern)))
        if paths:
            print(f"(found files under {fallback}/, not {results_dir}/ directly)")
    if not paths:
        raise SystemExit(f"no {pattern} files found in {results_dir} or {results_dir}/results")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(paths)} files, {len(df)} total seed-rows")
    return df


def summarize(df):
    rows = []
    for (stage, unknown_params, G_true), g in df.groupby(['stage', 'unknown_params', 'G_true']):
        # if stage not in [1,5,6]:
        #     continue
        unknown = unknown_params.split('|')
        row = {
            'stage': stage,
            'unknown_params': unknown_params,
            'G_true': G_true,
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

    summary = pd.DataFrame(rows).sort_values(['stage', 'G_true']).reset_index(drop=True)
    return summary


def plot_accuracy(summary, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for stage in sorted(summary['stage'].unique()):
        s = summary[summary['stage'] == stage].sort_values('G_true')
        ax.plot(s['G_true'], s['accuracy'], color=STAGE_COLORS[stage], linewidth=2,
                 marker='o', markersize=8, label=f"Stage {stage}: {STAGE_LABELS[stage]}")

    ax.set_xlabel('G (true)')
    ax.set_ylabel('Accuracy (major agent correctly identified)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Detection accuracy vs. signal strength, by unknown-parameter stage')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_parameter_estimates(df, out_path):
    """
    One panel per parameter that was ever unknown, showing every seed's
    fitted value against G_true (x-axis), colored by stage, with the TRUE
    value overlaid as a dashed reference curve -- a diagonal for G and a_0
    (both vary with G_true; a_0 = a*G_true by the market-clearing baseline),
    a flat line for parameters whose true value is fixed regardless of
    G_true (a, q, q_0, c, c_0, epslon, epslon_0).
    """
    param_order = ['G', 'a', 'a_0', 'q', 'q_0', 'epslon', 'epslon_0', 'c', 'c_0']
    params = [p for p in param_order
              if f'{p}_fit' in df.columns and df[f'{p}_fit'].notna().any()]
    if not params:
        print("no fitted-parameter columns found -- skipping parameter-estimate plot")
        return

    stages = sorted(df['stage'].unique())
    ncols = min(3, len(params))
    nrows = -(-len(params) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    for i, p in enumerate(params):
        ax = axes[i // ncols][i % ncols]
        fit_col = f'{p}_fit'
        true_col = 'G_true' if p == 'G' else f'{p}_true'
        sub = df[df[fit_col].notna()]

        for stage in stages:
            if stage not in [1,2,3,6]:
                        continue
            s = sub[sub['stage'] == stage]
            if s.empty:
                continue
            # small per-stage horizontal jitter so overlapping G_true values separate visually
            jitter = (stage - (len(stages) + 1) / 2) * 0.006
            ax.scatter(s['G_true'] + jitter, s[fit_col], color=STAGE_COLORS[stage],
                       s=18, alpha=0.55, linewidths=0, label=f"Stage {stage}")

        if p == 'G':
            # G's own true value IS G_true -- the reference line is just y=x
            g_vals = sorted(sub['G_true'].unique())
            ax.plot(g_vals, g_vals, color='#52514e', linewidth=1.5,
                     linestyle='--', zorder=0, label='true value')
        elif true_col in sub.columns:
            ref = sub[['G_true', true_col]].drop_duplicates().sort_values('G_true')
            ax.plot(ref['G_true'], ref[true_col], color='#52514e', linewidth=1.5,
                     linestyle='--', zorder=0, label='true value')

        ax.set_title(p)
        ax.set_xlabel('G (true)')
        ax.set_ylabel(f'{p} estimate')
        ax.grid(True, alpha=0.2)

    for j in range(len(params), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.suptitle('Parameter recovery: fitted vs. true, by stage', y=0.98)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles),
               bbox_to_anchor=(0.5, 0.94), frameon=False)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', default='.')
    args = parser.parse_args()

    df = load_all(args.results_dir)
    summary = summarize(df)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    print(summary.to_string(index=False))

    summary.to_csv('summary.csv', index=False)
    print("saved summary.csv")

    plot_accuracy(summary, 'summary_accuracy.png')
    plot_parameter_estimates(df, 'summary_parameter_estimates.png')


if __name__ == "__main__":
    main()
