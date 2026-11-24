"""Question 3: compare kept features using the fixed five CCC cutoffs."""

from redesign2.threshold_experiment import run_threshold_experiment


def run(workspace, run_id):
    return run_threshold_experiment(workspace, run_id=run_id)
