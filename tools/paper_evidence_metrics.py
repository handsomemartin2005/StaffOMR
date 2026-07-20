from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence


Projection = Literal["joint", "pitch", "duration"]


@dataclass(frozen=True, order=True)
class Event:
    pitch: str
    duration: Fraction


def musicxml_events(path: Path) -> list[Event]:
    root = ET.parse(path).getroot()
    events: list[Event] = []
    for part in root.findall(".//part"):
        divisions = 1
        for measure in part.findall("measure"):
            value = measure.findtext("attributes/divisions")
            if value:
                divisions = max(1, int(value))
            for note in measure.findall("note"):
                pitch = note.find("pitch")
                if pitch is None:
                    pitch_token = "R"
                else:
                    step = pitch.findtext("step", "C")
                    alter = pitch.findtext("alter", "0")
                    octave = pitch.findtext("octave", "4")
                    pitch_token = f"{step}{alter}@{octave}"
                duration = int(note.findtext("duration", "0") or 0)
                events.append(Event(pitch_token, Fraction(duration, divisions)))
    return events


def _project(event: Event, projection: Projection) -> object:
    if projection == "joint":
        return event
    if projection == "pitch":
        return event.pitch
    if projection == "duration":
        return event.duration
    raise ValueError(f"Unknown projection: {projection}")


def multiset_counts(
    gold: Iterable[Event], predicted: Iterable[Event], projection: Projection = "joint"
) -> dict[str, int]:
    gold_counter = Counter(_project(event, projection) for event in gold)
    pred_counter = Counter(_project(event, projection) for event in predicted)
    return {
        "matches": sum((gold_counter & pred_counter).values()),
        "predicted": sum(pred_counter.values()),
        "gold": sum(gold_counter.values()),
    }


def aggregate_counts(rows: Iterable[Mapping[str, int]]) -> dict[str, float | int]:
    rows = list(rows)
    matches = sum(int(row["matches"]) for row in rows)
    predicted = sum(int(row["predicted"]) for row in rows)
    gold = sum(int(row["gold"]) for row in rows)
    precision = 100.0 * matches / predicted if predicted else 0.0
    recall = 100.0 * matches / gold if gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matches": matches,
        "predicted_events": predicted,
        "gold_events": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "output_to_gold_ratio": 100.0 * predicted / gold if gold else 0.0,
    }


def decompose_event_errors(gold: Sequence[Event], predicted: Sequence[Event]) -> dict[str, object]:
    joint = multiset_counts(gold, predicted, "joint")
    pitch = multiset_counts(gold, predicted, "pitch")
    duration = multiset_counts(gold, predicted, "duration")
    return {
        "joint": joint,
        "pitch": pitch,
        "duration": duration,
        "pitch_correct_duration_wrong_opportunities": max(0, pitch["matches"] - joint["matches"]),
        "duration_correct_pitch_wrong_opportunities": max(
            0, duration["matches"] - joint["matches"]
        ),
    }


def paired_bootstrap_delta(
    full: Sequence[Mapping[str, int]],
    ablated: Sequence[Mapping[str, int]],
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    if len(full) != len(ablated):
        raise ValueError("Paired bootstrap requires equal page counts")
    if not full:
        raise ValueError("Paired bootstrap requires at least one page")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    observed = float(aggregate_counts(full)["f1"]) - float(aggregate_counts(ablated)["f1"])
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(full)) for _ in full]
        deltas.append(
            float(aggregate_counts([full[i] for i in indices])["f1"])
            - float(aggregate_counts([ablated[i] for i in indices])["f1"])
        )
    deltas.sort()
    low = deltas[int(0.025 * samples)]
    high = deltas[min(samples - 1, int(0.975 * samples))]
    return {
        "full_minus_ablated": observed,
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_samples": samples,
    }
