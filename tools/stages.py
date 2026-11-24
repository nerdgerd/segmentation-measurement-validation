"""Run the main computational stages with explicit inputs and fresh run names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "study"
sys.path.insert(0, str(W / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in (
        "prepare",
        "generate",
        "construct",
        "evaluate",
        "population",
        "finite",
        "coverage",
        "thresholds",
    ):
        stage = commands.add_parser(name)
        stage.add_argument("--run-id", required=True)
        if name in {"prepare", "generate", "finite", "coverage"}:
            stage.add_argument("--workers", type=int, default=1)
        if name in {"generate", "construct", "evaluate", "finite", "coverage"}:
            stage.add_argument("--prepared-run", type=Path, required=True)
        if name in {"generate", "construct", "evaluate"}:
            stage.add_argument("--source", required=True)
            stage.add_argument("--target", required=True)
        if name in {"construct", "evaluate", "finite"}:
            stage.add_argument("--generation-run", type=Path, required=True)
        if name in {"construct", "evaluate"}:
            stage.add_argument(
                "--scope",
                choices=["full", "discovery"]
                if name == "construct"
                else ["full", "audit"],
                required=True,
            )
        if name == "generate":
            stage.add_argument("--refine-from", type=Path)
            stage.add_argument("--convergence-run", type=Path)
        if name == "evaluate":
            stage.add_argument("--construction-run", type=Path, required=True)
            stage.add_argument(
                "--commitments",
                type=Path,
                required=True,
                help="JSON array of all eight commitment paths",
            )
            stage.add_argument("--coverage-run", type=Path, required=True)
        if name == "finite":
            stage.add_argument("--cohort", required=True)
    args = parser.parse_args()
    from redesign2.execution import activate_inputs

    activate_inputs(W)
    from redesign2.prepare_inputs import require_protocol_lock
    from redesign2.provenance import resolve_input_path

    cfg = require_protocol_lock(W)
    if args.stage == "prepare":
        from redesign2.prepare_inputs import prepare_inputs

        result = prepare_inputs(W, args.run_id, workers=args.workers)
    elif args.stage == "generate":
        from redesign2.generate import generate_primary

        result = generate_primary(
            W,
            args.prepared_run,
            args.source,
            args.target,
            args.run_id,
            workers=args.workers,
            refine_from=args.refine_from,
            convergence_run=args.convergence_run,
        )
    elif args.stage == "construct":
        from redesign2.construction import construct_direction

        result = construct_direction(
            W,
            cfg,
            args.prepared_run,
            args.generation_run,
            source=args.source,
            target=args.target,
            scope=args.scope,
            run_id=args.run_id,
        )
    elif args.stage == "evaluate":
        from redesign2.evaluate import evaluate_direction

        commitments = json.loads(args.commitments.read_text(encoding="utf-8"))
        if isinstance(commitments, dict):
            commitments = commitments.get(
                "all_commitment_paths", commitments.get("commitment_paths")
            )
        if not isinstance(commitments, list):
            raise ValueError("Commitment index must provide the eight explicit paths")
        commitments = [resolve_input_path(path) for path in commitments]
        result = evaluate_direction(
            W,
            cfg,
            args.prepared_run,
            args.generation_run,
            args.construction_run,
            commitments,
            args.source,
            args.target,
            args.scope,
            args.run_id,
            coverage_run=args.coverage_run,
        )
    elif args.stage == "population":
        from redesign2.simulate import simulate_population

        result = simulate_population(W, args.run_id)
    elif args.stage == "finite":
        from redesign2.simulation_pipeline import run_finite

        result = run_finite(
            W,
            cfg,
            args.prepared_run,
            args.generation_run,
            args.cohort,
            args.run_id,
            workers=args.workers,
        )
    elif args.stage == "coverage":
        from redesign2.simulation_pipeline import run_coverage

        result = run_coverage(
            W, cfg, args.prepared_run, args.run_id, workers=args.workers
        )
    else:
        from redesign2.threshold_experiment import run_threshold_experiment

        result = run_threshold_experiment(W, run_id=args.run_id)
    print(result)


if __name__ == "__main__":
    main()
