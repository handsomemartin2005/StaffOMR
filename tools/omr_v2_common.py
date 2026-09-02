from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import traditional_omr_demo as v1


V2_BASE_CLASS_NAMES = [
    "filled_notehead",
    "open_notehead",
    "stem",
    "beam",
    "barline",
    "ledger_line",
    "sharp",
    "flat",
    "natural",
    "rest",
    "treble_clef",
    "bass_clef",
    "text_region",
    "slur_or_tie",
]

V2_EXPANDED_CLASS_NAMES = [
    "notehead_black",
    "notehead_half",
    "notehead_whole",
    "notehead_double_whole",
    "stem",
    "beam",
    "ledger_line",
    "augmentation_dot",
    "repeat_dot",
    "flag_8th_up",
    "flag_8th_down",
    "flag_16th_up",
    "flag_16th_down",
    "flag_32nd_up",
    "flag_32nd_down",
    "flag_64th_up",
    "flag_64th_down",
    "flag_128th_up",
    "flag_128th_down",
    "barline",
    "repeat_barline",
    "treble_clef",
    "bass_clef",
    "c_clef_alto",
    "c_clef_tenor",
    "clef_8",
    "clef_15",
    "accidental_sharp",
    "accidental_flat",
    "accidental_natural",
    "accidental_double_sharp",
    "accidental_double_flat",
    "key_sharp",
    "key_flat",
    "key_natural",
    "rest_double_whole",
    "rest_whole",
    "rest_half",
    "rest_quarter",
    "rest_8th",
    "rest_16th",
    "rest_32nd",
    "rest_64th",
    "rest_128th",
    "slur_or_tie",
    "time_sig_0",
    "time_sig_1",
    "time_sig_2",
    "time_sig_3",
    "time_sig_4",
    "time_sig_5",
    "time_sig_6",
    "time_sig_7",
    "time_sig_8",
    "time_sig_9",
    "time_sig_common",
    "time_sig_cut_common",
    "dynamic_p",
    "dynamic_f",
    "dynamic_m",
    "dynamic_s",
    "dynamic_z",
    "dynamic_crescendo_hairpin",
    "dynamic_diminuendo_hairpin",
    "artic_staccato",
    "artic_staccatissimo",
    "artic_accent",
    "artic_tenuto",
    "artic_marcato",
    "fermata",
    "tuplet_3",
    "tuplet_6",
    "fingering",
    "pedal_mark",
    "pedal_up",
    "ornament_mordent",
    "ornament_turn",
    "arpeggiato",
    "brace",
    "bracket",
    "text_region",
]

V2_EXPANDED_CLEAN_CLASS_NAMES = [
    "notehead_black",
    "notehead_half",
    "notehead_whole",
    "stem",
    "beam",
    "ledger_line",
    "augmentation_dot",
    "repeat_dot",
    "flag_8th",
    "flag_16th",
    "flag_32nd",
    "flag_64th",
    "treble_clef",
    "bass_clef",
    "c_clef_alto",
    "c_clef_tenor",
    "accidental_sharp",
    "accidental_flat",
    "accidental_natural",
    "rest_whole",
    "rest_half",
    "rest_quarter",
    "rest_8th",
    "rest_16th",
    "slur_or_tie",
    "time_sig_0",
    "time_sig_1",
    "time_sig_2",
    "time_sig_3",
    "time_sig_4",
    "time_sig_5",
    "time_sig_6",
    "time_sig_7",
    "time_sig_8",
    "time_sig_9",
    "time_sig_common",
    "time_sig_cut_common",
    "dynamic_p",
    "dynamic_f",
    "dynamic_m",
    "dynamic_s",
    "dynamic_z",
    "dynamic_crescendo_hairpin",
    "dynamic_diminuendo_hairpin",
    "artic_staccato",
    "artic_accent",
    "artic_tenuto",
    "artic_marcato",
    "fermata",
    "tuplet_3",
    "tuplet_6",
    "fingering",
    "pedal_mark",
    "pedal_up",
    "ornament_mordent",
    "ornament_turn",
    "arpeggiato",
    "brace",
]

V2_EXPANDED_CLEAN_ALIASES: dict[str, str | None] = {
    "notehead_double_whole": "notehead_whole",
    "flag_8th_up": "flag_8th",
    "flag_8th_down": "flag_8th",
    "flag_16th_up": "flag_16th",
    "flag_16th_down": "flag_16th",
    "flag_32nd_up": "flag_32nd",
    "flag_32nd_down": "flag_32nd",
    "flag_64th_up": "flag_64th",
    "flag_64th_down": "flag_64th",
    "flag_128th_up": "flag_64th",
    "flag_128th_down": "flag_64th",
    "barline": None,
    "repeat_barline": None,
    "clef_8": "treble_clef",
    "clef_15": "treble_clef",
    "accidental_double_sharp": "accidental_sharp",
    "accidental_double_flat": "accidental_flat",
    "key_sharp": "accidental_sharp",
    "key_flat": "accidental_flat",
    "key_natural": "accidental_natural",
    "rest_double_whole": "rest_whole",
    "rest_32nd": "rest_16th",
    "rest_64th": "rest_16th",
    "rest_128th": "rest_16th",
    "artic_staccatissimo": "artic_staccato",
    "bracket": None,
    "text_region": None,
}

V2_CLASS_NAMES = V2_BASE_CLASS_NAMES

V2_TAXONOMIES = {
    "base": V2_BASE_CLASS_NAMES,
    "expanded": V2_EXPANDED_CLASS_NAMES,
    "expanded_clean": V2_EXPANDED_CLEAN_CLASS_NAMES,
}

V2_EXPANDED_TO_BASE_CLASS = {
    "notehead_black": "filled_notehead",
    "notehead_half": "open_notehead",
    "notehead_whole": "open_notehead",
    "notehead_double_whole": "open_notehead",
    "repeat_barline": "barline",
    "accidental_sharp": "sharp",
    "accidental_flat": "flat",
    "accidental_natural": "natural",
    "accidental_double_sharp": "sharp",
    "accidental_double_flat": "flat",
    "accidental_sharp": "sharp",
    "accidental_flat": "flat",
    "accidental_natural": "natural",
    "key_sharp": "sharp",
    "key_flat": "flat",
    "key_natural": "natural",
    "rest_double_whole": "rest",
    "rest_whole": "rest",
    "rest_half": "rest",
    "rest_quarter": "rest",
    "rest_8th": "rest",
    "rest_16th": "rest",
    "rest_32nd": "rest",
    "rest_64th": "rest",
    "rest_128th": "rest",
    "clef_8": "treble_clef",
    "clef_15": "treble_clef",
    "c_clef_alto": "treble_clef",
    "c_clef_tenor": "treble_clef",
}

V2_CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(V2_CLASS_NAMES)}
V2_ID_TO_CLASS = {idx: name for name, idx in V2_CLASS_TO_ID.items()}
V2_ZERO_BASED_ID_TO_CLASS = {idx: name for idx, name in enumerate(V2_CLASS_NAMES)}
V2_CLASS_TO_ZERO_BASED_ID = {name: idx for idx, name in V2_ZERO_BASED_ID_TO_CLASS.items()}


def class_names_for_taxonomy(taxonomy: str) -> list[str]:
    if taxonomy not in V2_TAXONOMIES:
        raise ValueError(f"Unknown V2 taxonomy: {taxonomy}")
    return list(V2_TAXONOMIES[taxonomy])


def clean_expanded_class_for_symbol_class(symbol_class: str | None) -> str | None:
    if symbol_class is None:
        return None
    mapped = V2_EXPANDED_CLEAN_ALIASES.get(symbol_class, symbol_class)
    if mapped is None:
        return None
    if mapped in V2_EXPANDED_CLEAN_CLASS_NAMES:
        return mapped
    return None


def zero_based_class_to_id(class_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}


def base_class_for_symbol_class(symbol_class: str) -> str | None:
    if symbol_class in V2_BASE_CLASS_NAMES:
        return symbol_class
    return V2_EXPANDED_TO_BASE_CLASS.get(symbol_class)


def expanded_class_from_deepscores_name(name: str) -> str | None:
    lower = name.lower()
    if lower == "staff":
        return "staff"
    if lower in {"ledgerline", "legerline"}:
        return "ledger_line"
    if "noteheaddoublewhole" in lower:
        return "notehead_double_whole"
    if "noteheadblack" in lower or "noteheadfull" in lower:
        return "notehead_black"
    if "noteheadhalf" in lower:
        return "notehead_half"
    if "noteheadwhole" in lower:
        return "notehead_whole"
    if lower == "stem":
        return "stem"
    if lower == "beam":
        return "beam"
    if lower == "barline":
        return "barline"
    if "repeat" in lower and "barline" in lower:
        return "repeat_barline"
    if lower == "clefg":
        return "treble_clef"
    if lower == "cleff":
        return "bass_clef"
    if lower == "clefcalto":
        return "c_clef_alto"
    if lower == "clefctenor":
        return "c_clef_tenor"
    if lower == "clef8":
        return "clef_8"
    if lower == "clef15":
        return "clef_15"
    if lower == "keysharp":
        return "key_sharp"
    if lower == "keyflat":
        return "key_flat"
    if lower == "keynatural":
        return "key_natural"
    if "accidentaldoublesharp" in lower:
        return "accidental_double_sharp"
    if "accidentaldoubleflat" in lower:
        return "accidental_double_flat"
    if "accidentalsharp" in lower:
        return "accidental_sharp"
    if "accidentalflat" in lower:
        return "accidental_flat"
    if "accidentalnatural" in lower:
        return "accidental_natural"
    if lower == "augmentationdot":
        return "augmentation_dot"
    if lower == "repeatdot":
        return "repeat_dot"
    if lower.startswith("flag"):
        for value in ("128th", "64th", "32nd", "16th", "8th"):
            if value in lower:
                direction = "up" if "up" in lower else "down"
                return f"flag_{value}_{direction}"
    if lower.startswith("rest") and lower != "resthbar":
        rest_map = {
            "restdoublewhole": "rest_double_whole",
            "restwhole": "rest_whole",
            "resthalf": "rest_half",
            "restquarter": "rest_quarter",
            "rest8th": "rest_8th",
            "rest16th": "rest_16th",
            "rest32nd": "rest_32nd",
            "rest64th": "rest_64th",
            "rest128th": "rest_128th",
        }
        return rest_map.get(lower, "rest_quarter")
    if lower in {"slur", "tie"}:
        return "slur_or_tie"
    if lower.startswith("timesig"):
        suffix = lower.replace("timesig", "")
        if suffix.isdigit():
            return f"time_sig_{suffix}"
        if suffix == "common":
            return "time_sig_common"
        if suffix in {"cutcommon", "cut"}:
            return "time_sig_cut_common"
    if lower.startswith("dynamic"):
        if "crescendohairpin" in lower:
            return "dynamic_crescendo_hairpin"
        if "diminuendohairpin" in lower:
            return "dynamic_diminuendo_hairpin"
        suffix = lower.replace("dynamic", "").lower()
        if suffix in {"p", "f", "m", "s", "z"}:
            return f"dynamic_{suffix}"
    if lower.startswith("artic"):
        if "staccatissimo" in lower:
            return "artic_staccatissimo"
        if "staccato" in lower:
            return "artic_staccato"
        if "accent" in lower:
            return "artic_accent"
        if "tenuto" in lower:
            return "artic_tenuto"
        if "marcato" in lower:
            return "artic_marcato"
    if lower.startswith("fermata"):
        return "fermata"
    if lower == "tuplet3":
        return "tuplet_3"
    if lower == "tuplet6":
        return "tuplet_6"
    if lower.startswith("fingering"):
        return "fingering"
    if lower == "keyboardpedalped":
        return "pedal_mark"
    if lower == "keyboardpedalup":
        return "pedal_up"
    if lower == "ornamentmordent":
        return "ornament_mordent"
    if lower == "ornamentturn":
        return "ornament_turn"
    if lower == "arpeggiato":
        return "arpeggiato"
    if lower == "brace":
        return "brace"
    if lower == "bracket":
        return "bracket"
    return None


def class_from_deepscores_name(name: str, taxonomy: str = "base") -> str | None:
    if taxonomy == "expanded_clean":
        return clean_expanded_class_for_symbol_class(expanded_class_from_deepscores_name(name))
    if taxonomy == "expanded":
        return expanded_class_from_deepscores_name(name)
    lower = name.lower()
    if lower == "staff":
        return "staff"
    if lower in {"ledgerline", "legerline"}:
        return "ledger_line"
    if "noteheadblack" in lower or "noteheadfull" in lower:
        return "filled_notehead"
    if (
        "noteheadhalf" in lower
        or "noteheadwhole" in lower
        or "noteheaddoublewhole" in lower
    ):
        return "open_notehead"
    if lower == "stem":
        return "stem"
    if lower == "beam":
        return "beam"
    if lower == "clefg":
        return "treble_clef"
    if lower == "cleff":
        return "bass_clef"
    if "accidentalsharp" in lower or lower == "keysharp":
        return "sharp"
    if "accidentalflat" in lower or lower == "keyflat":
        return "flat"
    if "accidentalnatural" in lower or lower == "keynatural":
        return "natural"
    if lower.startswith("rest") and lower != "resthbar":
        return "rest"
    if lower in {"slur", "tie"}:
        return "slur_or_tie"
    return None


def canonical_deepscores_class(
    ann: dict[str, Any],
    categories: dict[str, Any],
    taxonomy: str = "base",
) -> str | None:
    for cid in ann.get("cat_id", []):
        cat = categories.get(str(cid))
        if not cat:
            continue
        if cat.get("annotation_set") != "deepscores":
            continue
        mapped = class_from_deepscores_name(cat["name"], taxonomy=taxonomy)
        if mapped is not None:
            return mapped
    for cid in ann.get("cat_id", []):
        cat = categories.get(str(cid))
        if not cat:
            continue
        mapped = class_from_deepscores_name(cat["name"], taxonomy=taxonomy)
        if mapped is not None:
            return mapped
    return None


def default_input_path() -> Path:
    return v1.DEFAULT_INPUT if v1.DEFAULT_INPUT.exists() else v1.LOCAL_FALLBACK_INPUT


def resolve_input_path(path: Path | None) -> Path:
    if path is None:
        return default_input_path()
    return path.expanduser().resolve()


def run_v1_symbol_pipeline(input_path: Path | None = None, pdf_dpi: int = 220) -> dict[str, Any]:
    resolved = resolve_input_path(input_path)
    image = v1.load_input_image(resolved, dpi=pdf_dpi)
    gray = np.array(image.convert("L"))
    mask, threshold = v1.binarize(gray)

    staff_lines = v1.find_staff_line_candidates(mask)
    staff_recovery = {
        "mode": "not_needed",
        "original_lines": len(staff_lines),
        "original_staves": 0,
        "recovered_staves": 0,
    }
    try:
        if staff_lines:
            staff_space = v1.estimate_staff_space(staff_lines)
        else:
            staff_lines = v1.find_staff_line_candidates_relaxed(mask)
            staff_space = v1.estimate_staff_space(staff_lines)
    except RuntimeError:
        relaxed_lines = v1.find_staff_line_candidates_relaxed(mask)
        morph_lines = v1.find_staff_line_candidates_morphological(gray)
        merged_lines = v1.merge_staff_line_candidates([*staff_lines, *relaxed_lines, *morph_lines])
        staff_space = v1.estimate_staff_space(merged_lines)
        template_staves, template_report = v1.recover_complete_staves_by_template(gray, staff_space, merged_lines)
        staff_lines = merged_lines
        staff_recovery = {
            "mode": "spacing_fallback_template_recovery",
            "original_lines": len(staff_lines),
            "relaxed_lines": len(relaxed_lines),
            "morphological_lines": len(morph_lines),
            "merged_lines": len(merged_lines),
            "original_staves": 0,
            "recovered_staves": len(template_staves),
            "recovered_space": float(staff_space),
        }
        staff_recovery.update(template_report)
        staves = template_staves
    else:
        staves = v1.group_staves(staff_lines, staff_space)
        staff_recovery.update(
            {
                "original_lines": len(staff_lines),
                "original_staves": len(staves),
                "recovered_staves": len(staves),
            }
        )
    if len(staves) < 2 and staff_lines:
        staff_lines, staff_space, staves, staff_recovery = v1.recover_staff_geometry_for_scans(
            mask,
            staff_lines,
            staff_space,
            staves,
            gray=gray,
        )
    cleaned = v1.remove_staff_lines(mask, staves)
    v1.detect_clefs(cleaned, staves)
    long_run_mask = v1.find_long_run_mask(cleaned, staves)
    preliminary_text_regions = v1.detect_text_regions(cleaned, staves, [])
    accidentals = v1.detect_accidentals(cleaned, staves, preliminary_text_regions)
    text_regions = v1.detect_text_regions(cleaned, staves, accidentals)
    accidental_mask = v1.mask_regions(cleaned.shape, accidentals, pad_x=2, pad_y=2)
    text_mask = v1.mask_regions(cleaned.shape, text_regions, pad_x=2, pad_y=2)
    note_mask = cleaned & ~long_run_mask & ~accidental_mask & ~text_mask
    notes = v1.detect_noteheads(note_mask, staves, raw_mask=mask)
    stems = v1.detect_stems(cleaned, notes, staves)
    ledger_lines = v1.detect_ledger_lines(mask, notes, staves)
    barlines = v1.detect_barlines(mask, staves)
    beam_links = v1.detect_beam_links_from_stems(cleaned, stems, staves)
    beam_groups = v1.group_beam_links(beam_links, stems, staves)
    pruned = v1.erase_detected_notes_and_stems(cleaned, notes, stems, staves)
    rests, beams, slurs = v1.detect_secondary_symbols(pruned, staves, notes, stems, beam_groups, barlines)

    return {
        "input": resolved,
        "image": image,
        "threshold": threshold,
        "staff_lines": staff_lines,
        "staff_space": staff_space,
        "staves": staves,
        "staff_recovery": staff_recovery,
        "mask": mask,
        "cleaned": cleaned,
        "accidentals": accidentals,
        "text_regions": text_regions,
        "noteheads": notes,
        "stems": stems,
        "ledger_lines": ledger_lines,
        "barlines": barlines,
        "rests": rests,
        "beams": beams,
        "beam_links": beam_links,
        "beam_groups": beam_groups,
        "slurs_or_ties": slurs,
    }


def clamp_bbox(bbox: list[float] | tuple[float, float, float, float], width: int, height: int) -> list[int] | None:
    x0, y0, x1, y1 = bbox
    ix0 = max(0, min(width - 1, int(math.floor(x0))))
    iy0 = max(0, min(height - 1, int(math.floor(y0))))
    ix1 = max(0, min(width, int(math.ceil(x1))))
    iy1 = max(0, min(height, int(math.ceil(y1))))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return [ix0, iy0, ix1, iy1]


def bbox_xyxy_to_xywh(bbox: list[int]) -> list[float]:
    x0, y0, x1, y1 = bbox
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def bbox_area(bbox: list[int]) -> float:
    return float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))


def notehead_bbox(note: Any, staff: Any, width: int, height: int) -> list[int] | None:
    rx = max(5, int(round(staff.space * 0.78)))
    ry = max(4, int(round(staff.space * 0.50)))
    return clamp_bbox([note.x - rx, note.y - ry, note.x + rx, note.y + ry], width, height)


def line_bbox(x: float, y0: float, y1: float, half_width: int, width: int, height: int) -> list[int] | None:
    return clamp_bbox([x - half_width, y0, x + half_width + 1, y1], width, height)


def add_symbol(
    symbols: list[dict[str, Any]],
    image_size: tuple[int, int],
    symbol_class: str,
    bbox: list[int] | None,
    confidence: float,
    source: str,
    attributes: dict[str, Any] | None = None,
) -> None:
    if bbox is None or symbol_class not in V2_CLASS_TO_ID:
        return
    width, height = image_size
    bbox = clamp_bbox(bbox, width, height)
    if bbox is None or bbox_area(bbox) <= 0:
        return
    symbols.append(
        {
            "id": f"weak_{len(symbols):06d}",
            "class": symbol_class,
            "bbox": bbox,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "source": source,
            "attributes": attributes or {},
        }
    )


def v1_result_to_weak_symbols(result: dict[str, Any]) -> list[dict[str, Any]]:
    image: Image.Image = result["image"]
    image_size = image.size
    staves = result["staves"]
    symbols: list[dict[str, Any]] = []

    for staff in staves:
        if staff.clef_bbox is None:
            continue
        clef_class = "treble_clef" if staff.clef_type == "treble" else "bass_clef"
        add_symbol(
            symbols,
            image_size,
            clef_class,
            [int(v) for v in staff.clef_bbox],
            0.80,
            "v1_clef_region",
            {"staff": staff.index, "clef_type": staff.clef_type},
        )

    for idx, note in enumerate(result["noteheads"]):
        staff = staves[note.staff]
        symbol_class = "open_notehead" if note.kind == "open" else "filled_notehead"
        add_symbol(
            symbols,
            image_size,
            symbol_class,
            notehead_bbox(note, staff, *image_size),
            note.score,
            "v1_notehead",
            {
                "staff": note.staff,
                "note_index": idx,
                "pitch_step": note.pitch_step,
                "kind": note.kind,
                "black_density": note.black_density,
                "center_density": note.center_density,
            },
        )

    for idx, stem in enumerate(result["stems"]):
        add_symbol(
            symbols,
            image_size,
            "stem",
            line_bbox(stem.x, stem.y0, stem.y1, 2, *image_size),
            stem.score,
            "v1_stem",
            {
                "staff": stem.staff,
                "stem_index": idx,
                "note_index": stem.note_index,
                "direction": stem.direction,
                "length": stem.length,
            },
        )

    for idx, group in enumerate(result["beam_groups"]):
        add_symbol(
            symbols,
            image_size,
            "beam",
            [int(v) for v in group.bbox],
            group.score,
            "v1_beam_group",
            {
                "staff": group.staff,
                "beam_group_index": idx,
                "direction": group.direction,
                "stem_indices": group.stem_indices,
                "link_indices": group.link_indices,
                "slope": group.slope,
                "hit_ratio": group.hit_ratio,
                "density": group.density,
            },
        )

    for idx, barline in enumerate(result["barlines"]):
        add_symbol(
            symbols,
            image_size,
            "barline",
            line_bbox(barline.x, barline.y0, barline.y1, 2, *image_size),
            barline.score,
            "v1_barline",
            {"staff": barline.staff, "barline_index": idx, "kind": barline.kind},
        )

    for idx, ledger in enumerate(result["ledger_lines"]):
        add_symbol(
            symbols,
            image_size,
            "ledger_line",
            [int(v) for v in ledger.bbox],
            ledger.score,
            "v1_ledger_line",
            {
                "staff": ledger.staff,
                "ledger_index": idx,
                "note_index": ledger.note_index,
                "side": ledger.side,
                "line_number": ledger.line_number,
            },
        )

    for idx, accidental in enumerate(result["accidentals"]):
        add_symbol(
            symbols,
            image_size,
            accidental.kind,
            [int(v) for v in accidental.bbox],
            accidental.score,
            "v1_accidental",
            {
                "staff": accidental.staff,
                "accidental_index": idx,
                "kind": accidental.kind,
                "density": accidental.density,
            },
        )

    for idx, rest in enumerate(result["rests"]):
        add_symbol(
            symbols,
            image_size,
            "rest",
            [int(v) for v in rest.bbox],
            rest.score,
            "v1_rest",
            {"staff": rest.staff, "rest_index": idx, "kind": rest.kind, "density": rest.density},
        )

    for idx, text_region in enumerate(result["text_regions"]):
        add_symbol(
            symbols,
            image_size,
            "text_region",
            [int(v) for v in text_region.bbox],
            text_region.score,
            "v1_text_region",
            {
                "staff": text_region.staff,
                "text_region_index": idx,
                "text": text_region.text,
                "component_count": text_region.component_count,
            },
        )

    for idx, slur in enumerate(result["slurs_or_ties"]):
        add_symbol(
            symbols,
            image_size,
            "slur_or_tie",
            [int(v) for v in slur.bbox],
            slur.score,
            "v1_slur_or_tie",
            {"staff": slur.staff, "slur_index": idx, "kind": slur.kind, "density": slur.density},
        )

    return symbols


def enrich_symbols_with_staff_geometry(symbols: list[dict[str, Any]], staves: list[Any]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for symbol in symbols:
        item = dict(symbol)
        item["attributes"] = dict(symbol.get("attributes") or {})
        x0, y0, x1, y1 = item["bbox"]
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        if staves:
            staff = min(
                staves,
                key=lambda candidate: min(
                    abs(cy - candidate.y0),
                    abs(cy - candidate.y1),
                    abs(cy - 0.5 * (candidate.y0 + candidate.y1)),
                ),
            )
            item["attributes"].setdefault("staff", staff.index)
            item["attributes"]["staff_space"] = staff.space
            item["attributes"]["normalized_width"] = (x1 - x0) / max(1.0, staff.space)
            item["attributes"]["normalized_height"] = (y1 - y0) / max(1.0, staff.space)
            if item["class"] in {"filled_notehead", "open_notehead"}:
                pitch_step, pitch_y = min(v1.legal_pitch_centers(staff), key=lambda pair: abs(cy - pair[1]))
                item["attributes"].setdefault("pitch_step", pitch_step)
                item["attributes"]["pitch_y_error"] = cy - pitch_y
        item["attributes"]["center"] = [float(cx), float(cy)]
        enriched.append(item)
    return enriched


def symbols_to_coco(
    image_id: int,
    image_path: Path,
    image_size: tuple[int, int],
    symbols: list[dict[str, Any]],
) -> dict[str, Any]:
    width, height = image_size
    images = [
        {
            "id": image_id,
            "file_name": str(image_path),
            "width": width,
            "height": height,
        }
    ]
    annotations = []
    for idx, symbol in enumerate(symbols, start=1):
        bbox = symbol["bbox"]
        annotations.append(
            {
                "id": idx,
                "image_id": image_id,
                "category_id": V2_CLASS_TO_ID[symbol["class"]],
                "bbox": bbox_xyxy_to_xywh(bbox),
                "area": bbox_area(bbox),
                "iscrowd": 0,
                "score": symbol.get("confidence", 1.0),
                "source": symbol.get("source", "unknown"),
                "attributes": symbol.get("attributes", {}),
            }
        )
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": idx, "name": name} for name, idx in V2_CLASS_TO_ID.items()],
    }


def merge_coco_documents(docs: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {
        "images": [],
        "annotations": [],
        "categories": [{"id": idx, "name": name} for name, idx in V2_CLASS_TO_ID.items()],
    }
    next_ann_id = 1
    for doc in docs:
        merged["images"].extend(doc["images"])
        for ann in doc["annotations"]:
            item = dict(ann)
            item["id"] = next_ann_id
            next_ann_id += 1
            merged["annotations"].append(item)
    return merged


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def color_for_class(name: str) -> tuple[int, int, int, int]:
    palette = {
        "filled_notehead": (220, 30, 40, 230),
        "open_notehead": (20, 160, 220, 230),
        "stem": (160, 40, 210, 230),
        "beam": (0, 170, 60, 230),
        "barline": (255, 135, 0, 230),
        "ledger_line": (120, 70, 0, 230),
        "sharp": (255, 70, 180, 230),
        "flat": (255, 70, 180, 230),
        "natural": (255, 70, 180, 230),
        "rest": (210, 165, 0, 230),
        "treble_clef": (255, 140, 0, 230),
        "bass_clef": (255, 140, 0, 230),
        "text_region": (70, 90, 255, 230),
        "slur_or_tie": (150, 80, 255, 230),
    }
    return palette.get(name, (0, 0, 0, 230))


def draw_symbol_overlay(image: Image.Image, symbols: list[dict[str, Any]], out_path: Path) -> None:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    for symbol in symbols:
        x0, y0, x1, y1 = symbol["bbox"]
        color = color_for_class(symbol["class"])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{symbol['class']} {symbol.get('confidence', 0.0):.2f}"
        draw.rectangle([x0, max(0, y0 - 14), x0 + 7 * len(label) + 4, y0], fill=(255, 255, 255, 210))
        draw.text((x0 + 2, max(0, y0 - 13)), label, fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def count_by_class(symbols: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol["class"]] = counts.get(symbol["class"], 0) + 1
    return dict(sorted(counts.items()))

