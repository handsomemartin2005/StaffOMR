from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any


DURATION_TO_KERN = {
    "whole": "1",
    "half": "2",
    "quarter": "4",
    "eighth": "8",
    "16th": "16",
    "32nd": "32",
    "64th": "64",
}
STEP_ORDER = {step: index for index, step in enumerate("CDEFGAB")}


@dataclass(frozen=True)
class CanonicalEvent:
    event: dict[str, Any]
    staff_slot: int
    x: float
    pitch: tuple[int, int, int, str]
    token: str


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def event_x(event: dict[str, Any]) -> float:
    center = event.get("center") or []
    fallback = center[0] if center else 0.0
    return _number(event.get("x"), _number(fallback))


def event_y(event: dict[str, Any]) -> float:
    center = event.get("center") or []
    if len(center) >= 2:
        return _number(center[1])
    bbox = event.get("bbox") or []
    if len(bbox) >= 4:
        return (_number(bbox[1]) + _number(bbox[3])) / 2.0
    return 0.0


def pitch_key(event: dict[str, Any]) -> tuple[int, int, int, str]:
    pitch = event.get("pitch") or {}
    step = str(pitch.get("step") or "C").upper()
    try:
        octave = int(pitch.get("octave", 4))
    except (TypeError, ValueError):
        octave = 4
    try:
        alter = int(pitch.get("alter", 0) or 0)
    except (TypeError, ValueError):
        alter = 0
    return octave, STEP_ORDER.get(step, 0), alter, str(event.get("id") or "")


def kern_pitch(step: str, octave: Any, alter: Any) -> str:
    step = str(step or "C").upper()
    try:
        octave_number = int(octave)
    except (TypeError, ValueError):
        octave_number = 4
    if octave_number >= 4:
        token = step.lower() * max(1, octave_number - 3)
    else:
        token = step.upper() * max(1, 4 - octave_number)
    try:
        alteration = int(alter or 0)
    except (TypeError, ValueError):
        alteration = 0
    if alteration > 0:
        token += "#" * alteration
    elif alteration < 0:
        token += "-" * -alteration
    return token


def semantic_event_to_ekern(event: dict[str, Any]) -> str | None:
    duration = DURATION_TO_KERN.get(str(event.get("duration_hint") or ""), "4")
    try:
        dot_count = max(0, int(event.get("dot_count") or 0))
    except (TypeError, ValueError):
        dot_count = 0
    dots = "." * dot_count
    if event.get("type") == "rest":
        return f"{duration}{dots}@r"
    if event.get("type") != "note":
        return None
    pitch = event.get("pitch") or {}
    token = kern_pitch(pitch.get("step", "C"), pitch.get("octave"), pitch.get("alter"))
    return f"{duration}{dots}@{token}"


def part_events(part: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for measure in part.get("measures") or []:
        events.extend(measure.get("events") or [])
    return events


def _part_y(part: dict[str, Any]) -> float:
    ys = [event_y(event) for event in part_events(part) if event_y(event)]
    if ys:
        return float(median(ys))
    return _number(part.get("staff")) * 100.0


def canonical_parts(semantics: dict[str, Any]) -> list[dict[str, Any]]:
    """Return physical staves in page reading order (top to bottom)."""

    parts = list(semantics.get("parts") or [])

    def key(part: dict[str, Any]) -> tuple[float, float, str]:
        # The assembler numbers physical staves from top to bottom.  Prefer that
        # stable index and use observed y only when the index is absent.
        staff = part.get("staff")
        if staff is None:
            return _part_y(part), _part_y(part), str(part.get("id") or "")
        return _number(staff), _part_y(part), str(part.get("id") or "")

    return sorted(parts, key=key)


def group_piano_systems(parts: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Pair adjacent physical staves without learning or target annotations.

    Polish pages are piano-form pages.  A treble staff followed by a bass staff
    forms a system.  If clef recognition is missing or wrong, consecutive
    top-to-bottom pairs are the deterministic fallback.  An unmatched staff is
    retained as a one-staff system instead of being dropped.
    """

    systems: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(parts):
        upper = parts[index]
        if index + 1 >= len(parts):
            systems.append([upper])
            break
        lower = parts[index + 1]
        upper_clef = str(upper.get("clef_type") or "").lower()
        lower_clef = str(lower.get("clef_type") or "").lower()
        if upper_clef == "bass" and lower_clef == "treble":
            systems.append([upper])
            index += 1
            continue
        systems.append([upper, lower])
        index += 2
    return systems


def _cluster_positions(values: Iterable[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or abs(value - sum(groups[-1]) / len(groups[-1])) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(group) / len(group) for group in groups]


def system_measure_boundaries(system: Sequence[dict[str, Any]], tolerance: float = 6.0) -> list[float]:
    """Create shared measure boundaries from all staves in a system.

    For a one-staff system, all detected barlines are retained.  Requiring
    support from both piano staves removes staff-local vertical false positives;
    the resulting shared boundary is then applied to both staves.
    """

    observations: list[tuple[float, int]] = []
    for slot, part in enumerate(system):
        observations.extend((_number(value), slot) for value in part.get("barlines") or [])
    clusters: list[list[tuple[float, int]]] = []
    for observation in sorted(observations):
        if not clusters:
            clusters.append([observation])
            continue
        center = sum(item[0] for item in clusters[-1]) / len(clusters[-1])
        if abs(observation[0] - center) <= tolerance:
            clusters[-1].append(observation)
        else:
            clusters.append([observation])
    required_support = 1 if len(system) == 1 else 2
    return [
        sum(item[0] for item in cluster) / len(cluster)
        for cluster in clusters
        if len({item[1] for item in cluster}) >= required_support
    ]


def _measure_index(x: float, boundaries: Sequence[float]) -> int:
    return sum(x >= boundary for boundary in boundaries)


def canonical_system_events(system: Sequence[dict[str, Any]]) -> list[CanonicalEvent]:
    result: list[CanonicalEvent] = []
    for staff_slot, part in enumerate(system):
        for event in part_events(part):
            token = semantic_event_to_ekern(event)
            if token:
                result.append(
                    CanonicalEvent(
                        event=event,
                        staff_slot=staff_slot,
                        x=event_x(event),
                        pitch=pitch_key(event),
                        token=token,
                    )
                )
    return result


def onset_groups(events: Sequence[CanonicalEvent], tolerance: float = 4.0) -> list[list[CanonicalEvent]]:
    """Cluster nearby x positions into deterministic onset hypotheses."""

    groups: list[list[CanonicalEvent]] = []
    for event in sorted(events, key=lambda item: (item.x, item.staff_slot, item.pitch)):
        if not groups:
            groups.append([event])
            continue
        center = sum(item.x for item in groups[-1]) / len(groups[-1])
        same_stem = (
            abs(event.x - center) <= 2.0 * tolerance
            and bool(event.event.get("stem_id"))
            and any(
                item.staff_slot == event.staff_slot
                and item.event.get("stem_id") == event.event.get("stem_id")
                for item in groups[-1]
            )
        )
        if abs(event.x - center) <= tolerance or same_stem:
            groups[-1].append(event)
        else:
            groups.append([event])
    return groups


def _clef_token(part: dict[str, Any] | None) -> str:
    clef = str((part or {}).get("clef_type") or "").lower()
    return "*clefF4" if clef == "bass" else "*clefG2"


def semantic_to_canonical_ekern(
    semantics: dict[str, Any],
    *,
    onset_tolerance: float = 4.0,
    barline_tolerance: float = 6.0,
) -> str:
    """Serialize page semantics in piano system reading order.

    The legacy Polish proxy iterated part-major, placing an entire upper staff
    before its lower staff.  This serializer instead uses system-major,
    measure-major, onset-major order.  The output keeps two persistent piano
    spines (lower/bass first, upper/treble second), groups same-onset notes into
    chord cells, and sorts chord pitches low to high.
    """

    parts = canonical_parts(semantics)
    systems = group_piano_systems(parts)
    first_system = systems[0] if systems else []
    first_upper = first_system[0] if first_system else None
    first_lower = first_system[1] if len(first_system) > 1 else None
    lines = ["**ekern\t**ekern", f"{_clef_token(first_lower)}\t{_clef_token(first_upper)}"]
    measure_number = 1

    for system in systems:
        boundaries = system_measure_boundaries(system, tolerance=barline_tolerance)
        by_measure: dict[int, list[CanonicalEvent]] = {}
        for event in canonical_system_events(system):
            by_measure.setdefault(_measure_index(event.x, boundaries), []).append(event)
        if not by_measure:
            continue
        for local_measure in sorted(by_measure):
            lines.append(f"={measure_number}\t={measure_number}")
            measure_number += 1
            for group in onset_groups(by_measure[local_measure], tolerance=onset_tolerance):
                cells: list[str] = []
                # eKern/Polish uses the lower (bass) spine before the upper
                # (treble) spine.  A missing lower staff is represented by '.'.
                for staff_slot in (1, 0):
                    cell_events = sorted(
                        (item for item in group if item.staff_slot == staff_slot),
                        key=lambda item: (item.pitch, item.x, str(item.event.get("id") or "")),
                    )
                    cells.append(" ".join(item.token for item in cell_events) or ".")
                lines.append("\t".join(cells))
    lines.append("*-\t*-")
    return "\n".join(lines) + "\n"


__all__ = [
    "canonical_parts",
    "canonical_system_events",
    "event_x",
    "group_piano_systems",
    "onset_groups",
    "part_events",
    "pitch_key",
    "semantic_event_to_ekern",
    "semantic_to_canonical_ekern",
    "system_measure_boundaries",
]
