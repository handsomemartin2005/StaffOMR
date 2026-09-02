from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.polish_canonical_serialization import (
    CanonicalEvent,
    canonical_parts,
    canonical_system_events,
    group_piano_systems,
    onset_groups,
    system_measure_boundaries,
)


Normalize = Callable[[str], list[str]]
NORMALIZED_EVENT = re.compile(r"^(?P<duration>\d+\.*):(?P<pitch>.+)$")


@dataclass(frozen=True)
class Spine:
    base_staff: int
    voice_path: tuple[int, ...] = ()


@dataclass(frozen=True)
class GoldEvent:
    token: str
    measure: int
    onset: int
    base_staff: int
    voice_column: int
    voice_path: tuple[int, ...]
    chord_index: int


@dataclass(frozen=True)
class GoldOnset:
    measure: int
    onset: int
    events: tuple[GoldEvent, ...]

    @property
    def signature(self) -> tuple[tuple[int, str], ...]:
        return tuple(sorted((event.base_staff, event.token) for event in self.events))

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(event.token for event in self.events)


@dataclass(frozen=True)
class PredictedOnset:
    system: int
    measure: int
    onset: int
    events: tuple[CanonicalEvent, ...]
    normalized: tuple[str, ...]

    @property
    def signature(self) -> tuple[tuple[int, str], ...]:
        pairs: list[tuple[int, str]] = []
        for event, token in zip(self.events, self.normalized):
            base_staff = 0 if event.staff_slot == 1 else 1
            pairs.append((base_staff, token))
        return tuple(sorted(pairs))


def _apply_spine_operations(spines: list[Spine], fields: Sequence[str]) -> list[Spine]:
    result: list[Spine] = []
    index = 0
    while index < len(spines):
        field = fields[index].strip() if index < len(fields) else "*"
        spine = spines[index]
        if field == "*^":
            result.extend(
                [
                    Spine(spine.base_staff, spine.voice_path + (0,)),
                    Spine(spine.base_staff, spine.voice_path + (1,)),
                ]
            )
            index += 1
            continue
        if field == "*v":
            end = index + 1
            while end < len(spines) and end < len(fields) and fields[end].strip() == "*v":
                end += 1
            merged = spines[index:end]
            parent = merged[0].voice_path[:-1] if merged[0].voice_path else ()
            result.append(Spine(merged[0].base_staff, parent))
            index = end
            continue
        if field == "*-":
            index += 1
            continue
        result.append(spine)
        index += 1

    exchange = [index for index, field in enumerate(fields[: len(result)]) if field.strip() == "*x"]
    for left, right in zip(exchange[0::2], exchange[1::2]):
        result[left], result[right] = result[right], result[left]
    return result


def parse_ekern_hierarchy(text: str, normalize: Normalize) -> list[GoldOnset]:
    """Parse the hierarchy that survives in Polish eKern transcription."""

    spines: list[Spine] = []
    measure = 0
    onset_index = 0
    onsets: list[GoldOnset] = []
    for raw_line in text.splitlines():
        line = raw_line.strip("\r\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        fields = line.split("\t")
        if stripped.startswith("**"):
            spines = [Spine(base_staff=index) for index in range(len(fields))]
            continue
        if stripped.startswith("="):
            measure += 1
            continue
        if stripped.startswith("*"):
            if spines:
                spines = _apply_spine_operations(spines, fields)
            continue
        if not spines:
            continue
        events: list[GoldEvent] = []
        for column, (spine, cell) in enumerate(zip(spines, fields)):
            for chord_index, token in enumerate(normalize(cell)):
                events.append(
                    GoldEvent(
                        token=token,
                        measure=measure,
                        onset=onset_index,
                        base_staff=spine.base_staff,
                        voice_column=column,
                        voice_path=spine.voice_path,
                        chord_index=chord_index,
                    )
                )
        if events:
            onsets.append(GoldOnset(measure=measure, onset=onset_index, events=tuple(events)))
            onset_index += 1
    return onsets


def predicted_onsets(semantics: dict[str, Any], normalize: Normalize) -> list[PredictedOnset]:
    result: list[PredictedOnset] = []
    global_measure = 0
    global_onset = 0
    systems = group_piano_systems(canonical_parts(semantics))
    for system_index, system in enumerate(systems):
        boundaries = system_measure_boundaries(system)
        by_measure: dict[int, list[CanonicalEvent]] = defaultdict(list)
        for event in canonical_system_events(system):
            measure_index = sum(event.x >= boundary for boundary in boundaries)
            by_measure[measure_index].append(event)
        for local_measure in sorted(by_measure):
            for group in onset_groups(by_measure[local_measure]):
                ordered = tuple(
                    sorted(
                        group,
                        key=lambda item: (
                            0 if item.staff_slot == 1 else 1,
                            item.pitch,
                            item.x,
                            str(item.event.get("id") or ""),
                        ),
                    )
                )
                tokens: list[str] = []
                kept: list[CanonicalEvent] = []
                for event in ordered:
                    normalized = normalize(event.token)
                    if normalized:
                        kept.append(event)
                        tokens.append(normalized[0])
                if kept:
                    result.append(
                        PredictedOnset(
                            system=system_index,
                            measure=global_measure,
                            onset=global_onset,
                            events=tuple(kept),
                            normalized=tuple(tokens),
                        )
                    )
                    global_onset += 1
            global_measure += 1
    return result


def flatten_gold(onsets: Sequence[GoldOnset]) -> list[str]:
    return [token for onset in onsets for token in onset.tokens]


def flatten_predicted(onsets: Sequence[PredictedOnset]) -> list[str]:
    return [token for onset in onsets for token in onset.normalized]


def bag_oracle(ref: Sequence[str], hyp: Sequence[str]) -> dict[str, Any]:
    matches = sum((Counter(ref) & Counter(hyp)).values())
    minimum_errors = max(len(ref) - matches, len(hyp) - matches)
    return {
        "matches": matches,
        "ref_tokens": len(ref),
        "pred_tokens": len(hyp),
        "minimum_errors": minimum_errors,
        "SER_lower_bound": 100.0 * minimum_errors / len(ref) if ref else float(bool(hyp)) * 100.0,
    }


def staff_assignment_oracle(
    gold_onsets: Sequence[GoldOnset], predicted: Sequence[PredictedOnset]
) -> dict[str, Any]:
    gold_by_staff: dict[int, Counter[str]] = defaultdict(Counter)
    pred_by_staff: dict[int, Counter[str]] = defaultdict(Counter)
    for onset in gold_onsets:
        for event in onset.events:
            gold_by_staff[event.base_staff][event.token] += 1
    for onset in predicted:
        for event, token in zip(onset.events, onset.normalized):
            base_staff = 0 if event.staff_slot == 1 else 1
            pred_by_staff[base_staff][token] += 1
    constrained_matches = sum(
        sum((gold_by_staff[staff] & pred_by_staff[staff]).values())
        for staff in set(gold_by_staff) | set(pred_by_staff)
    )
    unconstrained = bag_oracle(flatten_gold(gold_onsets), flatten_predicted(predicted))
    constrained_errors = sum(
        max(
            sum(gold_by_staff[staff].values())
            - sum((gold_by_staff[staff] & pred_by_staff[staff]).values()),
            sum(pred_by_staff[staff].values())
            - sum((gold_by_staff[staff] & pred_by_staff[staff]).values()),
        )
        for staff in set(gold_by_staff) | set(pred_by_staff)
    )
    ref_len = len(flatten_gold(gold_onsets))
    return {
        "ref_tokens": ref_len,
        "pred_tokens": len(flatten_predicted(predicted)),
        "staff_constrained_matches": constrained_matches,
        "staff_constrained_errors": constrained_errors,
        "staff_constrained_SER_lower_bound": 100.0 * constrained_errors / ref_len if ref_len else 0.0,
        "oracle_reassigned_matches": unconstrained["matches"],
        "oracle_reassigned_errors": unconstrained["minimum_errors"],
        "oracle_reassigned_SER_lower_bound": unconstrained["SER_lower_bound"],
        "potential_SER_improvement": (
            100.0 * constrained_errors / ref_len - unconstrained["SER_lower_bound"] if ref_len else 0.0
        ),
    }


def exact_onset_order_oracle(
    gold_onsets: Sequence[GoldOnset], predicted: Sequence[PredictedOnset]
) -> tuple[list[str], dict[str, Any]]:
    """Reorder only predicted onset groups that exactly match a gold group."""

    gold_positions: dict[tuple[tuple[int, str], ...], deque[int]] = defaultdict(deque)
    for index, onset in enumerate(gold_onsets):
        gold_positions[onset.signature].append(index)
    matched: list[tuple[int, PredictedOnset]] = []
    unmatched: list[PredictedOnset] = []
    for onset in predicted:
        positions = gold_positions.get(onset.signature)
        if positions:
            matched.append((positions.popleft(), onset))
        else:
            unmatched.append(onset)
    ordered = [onset for _, onset in sorted(matched)] + unmatched
    exact_events = sum(len(onset.normalized) for _, onset in matched)
    return flatten_predicted(ordered), {
        "matched_onset_groups": len(matched),
        "gold_onset_groups": len(gold_onsets),
        "pred_onset_groups": len(predicted),
        "tokens_in_exact_groups": exact_events,
    }


def chord_diagnostics(
    gold_onsets: Sequence[GoldOnset], predicted: Sequence[PredictedOnset]
) -> dict[str, Any]:
    gold_chords: Counter[tuple[int, tuple[str, ...]]] = Counter()
    pred_chords: Counter[tuple[int, tuple[str, ...]]] = Counter()
    for onset in gold_onsets:
        cells: dict[tuple[int, int], list[str]] = defaultdict(list)
        for event in onset.events:
            cells[(event.base_staff, event.voice_column)].append(event.token)
        for (staff, _), tokens in cells.items():
            if len(tokens) > 1:
                gold_chords[(staff, tuple(sorted(tokens)))] += 1
    for onset in predicted:
        cells: dict[int, list[str]] = defaultdict(list)
        for event, token in zip(onset.events, onset.normalized):
            staff = 0 if event.staff_slot == 1 else 1
            cells[staff].append(token)
        for staff, tokens in cells.items():
            if len(tokens) > 1:
                pred_chords[(staff, tuple(sorted(tokens)))] += 1
    matches = sum((gold_chords & pred_chords).values())
    return {
        "gold_chords": sum(gold_chords.values()),
        "pred_chords": sum(pred_chords.values()),
        "exact_chord_matches": matches,
        "proxy_SER_effect": 0.0,
        "proxy_SER_effect_reason": "Current normalization splits spaces/tabs and discards chord-cell boundaries.",
    }


def voice_observability(
    gold_onsets: Sequence[GoldOnset], predicted: Sequence[PredictedOnset]
) -> dict[str, Any]:
    gold_voice_events = sum(
        bool(event.voice_path) for onset in gold_onsets for event in onset.events
    )
    gold_multivoice_onsets = 0
    for onset in gold_onsets:
        columns_by_staff: dict[int, set[int]] = defaultdict(set)
        for event in onset.events:
            columns_by_staff[event.base_staff].add(event.voice_column)
        gold_multivoice_onsets += any(len(columns) > 1 for columns in columns_by_staff.values())
    predicted_voice_fields = sum(
        "voice" in event.event for onset in predicted for event in onset.events
    )
    return {
        "gold_events_in_split_voice_spines": gold_voice_events,
        "gold_multivoice_onsets": gold_multivoice_onsets,
        "predicted_events_with_voice_field": predicted_voice_fields,
        "oracle_status": "unavailable" if predicted_voice_fields == 0 else "available",
        "reason": "Prediction semantics do not expose voice assignment."
        if predicted_voice_fields == 0
        else None,
    }


def split_normalized_event(token: str) -> tuple[str, str] | None:
    match = NORMALIZED_EVENT.fullmatch(token)
    if not match:
        return None
    return match.group("duration"), match.group("pitch")


def _maximum_binding_matches(
    predicted_pitches: Counter[str],
    predicted_durations: Counter[str],
    gold_joint: Counter[tuple[str, str]],
) -> int:
    source = ("source", "")
    sink = ("sink", "")
    capacity: dict[tuple[str, str], dict[tuple[str, str], int]] = defaultdict(dict)

    def add_edge(left: tuple[str, str], right: tuple[str, str], value: int) -> None:
        capacity[left][right] = capacity[left].get(right, 0) + value
        capacity[right].setdefault(left, 0)

    for pitch, count in predicted_pitches.items():
        add_edge(source, ("pitch", pitch), count)
    for (duration, pitch), count in gold_joint.items():
        add_edge(("pitch", pitch), ("duration", duration), count)
    for duration, count in predicted_durations.items():
        add_edge(("duration", duration), sink, count)

    flow = 0
    while True:
        parent: dict[tuple[str, str], tuple[str, str] | None] = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            for neighbour, residual in capacity[node].items():
                if residual > 0 and neighbour not in parent:
                    parent[neighbour] = node
                    queue.append(neighbour)
        if sink not in parent:
            break
        amount = 10**18
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            amount = min(amount, capacity[previous][node])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            capacity[previous][node] -= amount
            capacity[node][previous] = capacity[node].get(previous, 0) + amount
            node = previous
        flow += amount
    return flow


def event_binding_oracle(ref: Sequence[str], hyp: Sequence[str]) -> dict[str, Any]:
    gold_joint: Counter[tuple[str, str]] = Counter()
    predicted_joint: Counter[tuple[str, str]] = Counter()
    gold_rests: Counter[str] = Counter()
    pred_rests: Counter[str] = Counter()
    for tokens, target, rests in (
        (ref, gold_joint, gold_rests),
        (hyp, predicted_joint, pred_rests),
    ):
        for token in tokens:
            parsed = split_normalized_event(token)
            if not parsed:
                continue
            duration, pitch = parsed
            if pitch == "r":
                rests[token] += 1
            else:
                target[(duration, pitch)] += 1
    predicted_pitches = Counter()
    predicted_durations = Counter()
    for (duration, pitch), count in predicted_joint.items():
        predicted_pitches[pitch] += count
        predicted_durations[duration] += count
    binding_matches = _maximum_binding_matches(predicted_pitches, predicted_durations, gold_joint)
    rest_matches = sum((gold_rests & pred_rests).values())
    current_matches = sum((Counter(ref) & Counter(hyp)).values())
    matches = binding_matches + rest_matches
    minimum_errors = max(len(ref) - matches, len(hyp) - matches)
    return {
        "current_token_bag_matches": current_matches,
        "oracle_note_binding_matches": binding_matches,
        "fixed_rest_matches": rest_matches,
        "oracle_total_matches": matches,
        "ref_tokens": len(ref),
        "pred_tokens": len(hyp),
        "minimum_errors": minimum_errors,
        "oracle_SER_lower_bound": 100.0 * minimum_errors / len(ref) if ref else 0.0,
        "uses_target_annotations": True,
    }


def sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "GoldEvent",
    "GoldOnset",
    "PredictedOnset",
    "bag_oracle",
    "chord_diagnostics",
    "event_binding_oracle",
    "exact_onset_order_oracle",
    "flatten_gold",
    "flatten_predicted",
    "parse_ekern_hierarchy",
    "predicted_onsets",
    "sha256_files",
    "split_normalized_event",
    "staff_assignment_oracle",
    "voice_observability",
    "write_json",
]
