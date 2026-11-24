"""One command line for the current study, its figures and computational reruns."""

from pathlib import Path
import argparse
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "study"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STUDY / "src"))

from study.experiments.catalogue import EXPERIMENTS, experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "list", help="List every retained experiment in manuscript order"
    )
    commands.add_parser(
        "verify",
        aliases=["check"],
        help="Verify recorded bytes and all feature-selection cells",
    )
    commands.add_parser(
        "summary", help="Print the principal results from their saved tables"
    )
    figures = commands.add_parser(
        "figures", help="Rebuild all main and supplementary figures"
    )
    figures.add_argument("--output", type=Path, default=STUDY / "outputs/paper_assets")
    run = commands.add_parser(
        "run", help="Run one named experiment with imported patient inputs"
    )
    run.add_argument("experiment", choices=[item.key for item in EXPERIMENTS])
    run.add_argument("--run-id", required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        for item in EXPERIMENTS:
            print(f"{item.key:25s} {item.question:7s} {item.purpose}")
        print(
            "Q2 shares contour evaluation with Q1; see tools/stages.py evaluate and docs/EXPERIMENTS.md."
        )
        return
    if args.command in {"verify", "check"}:
        from study.experiments.integrity import verify_repository
        from study.experiments.reference_results import verify_threshold_points

        files = verify_repository(ROOT)
        cells = verify_threshold_points(STUDY / "reference")
        print(f"Verified {files} recorded files and {cells} feature-selection cells.")
        return
    if args.command == "summary":
        from study.experiments.reference_results import manuscript_summary

        print(json.dumps(manuscript_summary(STUDY / "reference"), indent=2))
        return
    if args.command == "figures":
        from study.figures.build import build_all

        build_all(STUDY / "reference", args.output)
        return
    if args.workers < 1:
        parser.error("--workers must be positive")
    if (args.pilot or args.workers != 1) and args.experiment != "ranking-finite":
        parser.error("--pilot/--workers apply to ranking-finite")
    if args.experiment != "measurement-differences":
        from redesign2.execution import activate_inputs

        activate_inputs(STUDY)
    item = experiment(args.experiment)
    module = importlib.import_module("study.experiments." + item.module)
    if args.experiment == "ranking-finite":
        return module.run(
            STUDY,
            args.run_id,
            workers=args.workers,
            pilot=args.pilot,
        )
    return module.run(STUDY, args.run_id)


if __name__ == "__main__":
    main()
