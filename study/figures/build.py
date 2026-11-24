"""Rebuild the complete current figure set through one ordinary import pipeline."""

from pathlib import Path
import json

from .common import sha, style
from .selection import SelectionFigures
from . import main, overview, supplement


def build_all(reference, output):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    context = SelectionFigures(reference, output)
    style("main")
    original = reference / "figure_sources/figure1_nature_original.pdf"
    record = overview.relabel(original, context.figs / "figure1_nature.pdf")
    context.inputs["figure_sources/" + original.name] = sha(original)
    context.outputs["figs/figure1_nature.pdf"] = record["updated_sha256"]
    main.build_figure2(context, context.outputs)
    main.build_figure3(context, context.outputs)
    main.build_figure4(context, context.outputs)
    main.build_supplement_threshold_figures(context, context.outputs)
    style("selection")
    context.averaging()
    context.screening()
    style("complete")
    supplement.mechanism(
        context, supplement.MRI, ["full", "audit"], "figureS_complete_mechanism"
    )
    support = supplement.complete_fidelity(context)
    supplement.coverage(context)
    supplement.transport(context)
    source = Path(__file__).parent
    manifest = {
        "inputs": context.inputs,
        "outputs": context.outputs,
        "source_files": {path.name: sha(path) for path in sorted(source.glob("*.py"))},
        "new_contour_generation": False,
        "resampling": False,
        "model_fitting": False,
        "figure1_text_edit": record,
        "complete_fidelity_support": support,
    }
    (output / "FIGURE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"Built {sum(name.endswith('.pdf') for name in context.outputs)} figures at {output}"
    )
    return manifest
