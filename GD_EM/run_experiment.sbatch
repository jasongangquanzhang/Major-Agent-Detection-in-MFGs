#!/bin/bash
#SBATCH --account=def-sebjai
#SBATCH --time=10:00:00
#SBATCH --job-name=MFG
#SBATCH --array=1-20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4096
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-type=END
#SBATCH --mail-user=gangquan.zhang@mail.utoronto.ca

# One array task per (stage, G_true) config: SLURM_ARRAY_TASK_ID (1-20) IS
# the --config-id value. G_true is the INNER loop over [0.1,0.3,0.5,0.7,0.9]:
#    1- 5 = stage 1 (G only)              across G_true = 0.1,0.3,0.5,0.7,0.9
#    6-10 = stage 2 (G, a, a_0)           across G_true = 0.1,0.3,0.5,0.7,0.9
#   11-15 = stage 3 (+ q, q_0)            across G_true = 0.1,0.3,0.5,0.7,0.9
#   16-20 = stage 4 (+ c, c_0)            across G_true = 0.1,0.3,0.5,0.7,0.9
# The authoritative mapping lives in CONFIGS in experiment.py -- if you ever
# change the stage list or G grid there, this comment block is the only
# thing that could go stale, the actual --array range/--config-id decoding
# does not need to change here.
#
# Each task writes to its OWN results_stage{K}_*_Gtrue{G}.csv -- no shared-
# file contention between array tasks, so all 20 run fully in parallel.
#
# NOTE: SLURM does not create --output's directory for you --
# run `mkdir -p logs` in GD_EM/ before the first `sbatch` submission.
#
# If a task dies (walltime, node failure) before its config finishes,
# experiment.py resumes from the last completed seed on re-submission --
# resubmit just that array index:
#   sbatch --array=<config_id> run_experiment.sbatch

module load python/3.12.4
source /home/gqzhang/RL_option_hedging/olympus/bin/activate
echo "Array task $SLURM_ARRAY_TASK_ID -> config-id $SLURM_ARRAY_TASK_ID"

srun /home/gqzhang/RL_option_hedging/olympus/bin/python3.12 \
    /home/gqzhang/Major-Agent-Detection-in-MFGs/GD_EM/experiment.py \
    --config-id $SLURM_ARRAY_TASK_ID
