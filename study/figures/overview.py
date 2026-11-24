"""Update the supplied overview PDF without reconstructing its scientific graphics.

The original vector PDF is an immutable input. Redactions remove only the named
text spans; contours, bars, axes and numerical annotations must remain identical.
Full DejaVu fonts come from matplotlib, so new wording does not depend on fonts
installed on the author's computer.
"""

from collections import Counter
from pathlib import Path
import hashlib
import json
import re

import fitz
from matplotlib import font_manager

ORIGINAL_SHA256 = "30f95715556d28f903e4af0fbf79f6649c3377c1fb002336532b0d91b3540f9c"
# Wording follows the manuscript: source/target cohorts, observed/generated/source CCC
# values, and the Introduction's three questions (correlation, scale, decision agreement).
LABELS = {
    "Source cohort": "Source cohort (calibration)",
    "Target cohort": "Target cohort (evaluation)",
    "reference and real alternate": "corrected and original contours",
    "target reference": "corrected contour",
    "synthetic alternates, many draws": "generated contours, many draws",
    "reference": "corrected",
    "real alternate": "original",
    "synthetic alternate": "generated",
    "CCC profiles": "Feature CCC values",
    "source": "source CCC",
    "synthetic target": "generated CCC",
    "real target (criterion)": "observed CCC (reference)",
    "descriptors, sorted by real target CCC": "features, sorted by observed CCC",
    "Three tests": "Three questions",
    "Ordering": "Correlation agreement",
    "Magnitude": "Scale agreement",
    "Thresholds": "Decision agreement",
    "Does shared ranking imply shared error?": "Does ρ exceed the simulated baseline?",
    "Is the disagreement size reproduced?": "Is the magnitude of differences reproduced?",
    "Are cutoff decisions the same?": "Are the same features retained?",
    "TCGA → UPENN": "TCGA-to-UPENN",
    "Same ranking ≠ same disagreement ≠ same threshold decisions: each needs its own validation.": "Correlation ≠ scale agreement ≠ decision agreement: each needs its own validation.",
    "profile association (ρ)": "CCC correlation (ρ)",
    "no coupling": "independent scales",
    "coupled": "associated scales",
    "  RMSE ratio, synthetic / real": "  RMSE ratio, generated / observed",
    "≈ 43% of real RMSE": "≈ 43% of observed RMSE",
    "descriptors retained at CCC ≥ 0.95": "features retained at CCC ≥ 0.95",
    "real": "observed",
    "synthetic": "generated",
}
# The Q3 bar label "source" already reads as the text does; only the panel-b heading changes.
UNCHANGED_IN_PANEL_C = {"source"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_graphics(page):
    # Text deletion changes content ordering, but must not alter any vector path.
    drawings = [
        {key: value for key, value in item.items() if key != "seqno"}
        for item in page.get_drawings()
    ]
    return hashlib.sha256(
        json.dumps(drawings, default=list, sort_keys=True).encode()
    ).hexdigest()


def numerical_labels(page):
    return Counter(re.findall(r"(?<!\w)[−+-]?\d+(?:\.\d+)?%?", page.get_text()))


def relabel(source, destination):
    source, destination = Path(source), Path(destination)
    if sha(source) != ORIGINAL_SHA256:
        raise ValueError("The supplied overview differs from the recorded original")
    doc = fitz.open(source)
    if len(doc) != 1:
        raise ValueError("Expected the original single-page overview")
    page = doc[0]
    original_graphics = scientific_graphics(page)
    original_numbers = numerical_labels(page)
    edits, seen = [], Counter()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line["spans"]
            joined = "".join(span["text"] for span in spans)
            groups = (
                [spans]
                if joined in LABELS
                else [[span] for span in spans if span["text"] in LABELS]
            )
            for group in groups:
                old = "".join(span["text"] for span in group)
                bbox = fitz.Rect(group[0]["bbox"])
                for span in group[1:]:
                    bbox |= fitz.Rect(span["bbox"])
                span = group[0]
                seen[old] += 1
                if old in UNCHANGED_IN_PANEL_C and span["origin"][0] > 350:
                    continue
                edits.append(
                    {
                        "old": old,
                        "new": LABELS[old],
                        "bbox": list(bbox),
                        "origin": list(span["origin"]),
                        "font": span["font"],
                        "size": span["size"],
                        "color": span["color"],
                    }
                )
                page.add_redact_annot(bbox, fill=False)
    expected = Counter({key: 1 for key in LABELS})
    expected["reference and real alternate"] = 2
    expected["source"] = 2
    expected["TCGA → UPENN"] = 2
    if seen != expected:
        raise ValueError(
            f"Overview labels changed: {seen - expected}; missing: {expected - seen}"
        )
    page.apply_redactions(images=0, graphics=0, text=0)
    fonts = {}
    for name in sorted({edit["font"] for edit in edits}):
        properties = font_manager.FontProperties(
            family="DejaVu Sans",
            weight="bold" if "Bold" in name else "normal",
            style="oblique" if "Oblique" in name else "normal",
        )
        path = font_manager.findfont(properties, fallback_to_default=False)
        resource = "Readable" + (
            "Bold" if "Bold" in name else "Oblique" if "Oblique" in name else "Regular"
        )
        page.insert_font(fontname=resource, fontfile=path)
        fonts[name] = (resource, fitz.Font(fontfile=path))
    for edit in edits:
        resource, font = fonts[edit["font"]]
        bbox, origin = fitz.Rect(edit["bbox"]), fitz.Point(edit["origin"])
        size, text = edit["size"], edit["new"]
        align, width = "left", 506 - origin.x
        if edit["old"] in {"real", "synthetic", "no coupling", "coupled"}:
            align, width = "right", bbox.x1 - (360 if origin.x > 350 else 0)
        elif edit["old"] == "TCGA → UPENN":  # right-aligned tag beside Q2/Q3 headings
            align, width = "right", bbox.x1 - 461
        elif edit["old"] in {"target reference", "synthetic alternates, many draws"}:
            width = (60 if origin.x < 60 else 214) - origin.x
        elif edit["old"] in {"Magnitude", "Thresholds"}:
            width = 459 - origin.x
        elif edit["old"] == "Ordering":
            width = 474 - origin.x
        elif edit["old"] in {
            "descriptors, sorted by real target CCC",
            "real target (criterion)",
        }:
            width = 347 - origin.x
        elif edit["old"] in {
            "profile association (ρ)",
            "descriptors retained at CCC ≥ 0.95",
        }:
            align, width = "center", 144
        elif edit["old"].startswith("Same ranking"):
            align, width = "center", 490
        size = min(size, width / font.text_length(text, fontsize=1))
        length = font.text_length(text, fontsize=size)
        if align == "right":
            origin.x = bbox.x1 - length
        elif align == "center":
            origin.x = (bbox.x0 + bbox.x1 - length) / 2
        color = tuple(((edit["color"] >> shift) & 255) / 255 for shift in [16, 8, 0])
        page.insert_text(origin, text, fontsize=size, fontname=resource, color=color)
        edit["rendered_size"], edit["rendered_origin"] = size, list(origin)
    if (
        scientific_graphics(page) != original_graphics
        or numerical_labels(page) != original_numbers
    ):
        raise AssertionError(
            "The wording edit changed scientific graphics or numerical annotations"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc.set_metadata({})
    doc.save(destination, garbage=4, deflate=True, no_new_id=True)
    doc.close()
    updated = fitz.open(destination)
    if (
        scientific_graphics(updated[0]) != original_graphics
        or numerical_labels(updated[0]) != original_numbers
    ):
        raise AssertionError(
            "The exported wording edit changed scientific graphics or numerical annotations"
        )
    updated.close()
    return {
        "original_sha256": ORIGINAL_SHA256,
        "updated_sha256": sha(destination),
        "vector_graphics_sha256": original_graphics,
        "numerical_labels": dict(original_numbers),
        "labels": edits,
        "operation": "text-only PDF edit; vector paths and numerical annotations preserved",
    }
