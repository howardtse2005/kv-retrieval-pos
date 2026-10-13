"""Generate deterministic synthetic KV retrieval datasets."""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

DEFAULT_SIZES = (75, 140, 300)
DEFAULT_CONDITIONS = ("high", "low")
POSITIONS = ("beginning", "early", "middle", "late", "end")
KEY_TYPES = ("date of birth", "favorite color", "pet name", "phone number", "employee ID")
COLORS = ("red", "blue", "green", "yellow", "purple", "orange", "black", "white")
PET_NAMES = ("Fluffy", "Mittens", "Buddy", "Coco", "Pepper", "Luna", "Milo", "Nala")


def position_index(label: str, size: int) -> int:
    """Return the deterministic index for a named target position."""
    if label == "beginning":
        return 0
    if label == "early":
        return size // 4
    if label == "middle":
        return size // 2
    if label == "late":
        return (3 * size) // 4
    if label == "end":
        return size - 1
    raise ValueError(f"Unknown position label: {label}")


def value_for(key_type: str, index: int) -> str:
    """Generate a unique value for a key type."""
    if key_type == "date of birth":
        return (date(1950, 1, 1) + timedelta(days=index)).strftime("%-d %B %Y")
    if key_type == "favorite color":
        # Include an ordinal suffix so values remain unique in large haystacks.
        return f"{COLORS[index % len(COLORS)]}-{index:03d}"
    if key_type == "pet name":
        return f"{PET_NAMES[index % len(PET_NAMES)]}-{index:03d}"
    if key_type == "phone number":
        return f"555-{index // 10000:03d}-{index % 10000:04d}"
    if key_type == "employee ID":
        return f"E{index:05X}"[-6:]
    raise ValueError(f"Unknown key type: {key_type}")


def shuffled_values(keys: list[str], example_index: int, rng: random.Random) -> list[str]:
    """Load generated values and shuffle them without changing their key type."""
    values_by_key: dict[str, list[str]] = {key: [] for key in KEY_TYPES}
    for index, key in enumerate(keys):
        values_by_key[key].append(value_for(key, example_index * len(keys) + index))
    for values in values_by_key.values():
        rng.shuffle(values)
    offsets = {key: 0 for key in KEY_TYPES}
    values = []
    for key in keys:
        values.append(values_by_key[key][offsets[key]])
        offsets[key] += 1
    return values


def make_example(
    size: int,
    condition: str,
    position: str,
    example_index: int,
    seed: int,
) -> dict:
    if condition not in DEFAULT_CONDITIONS:
        raise ValueError(f"Unknown similarity condition: {condition}")
    rng = random.Random(seed)
    subjects = [f"Subject_{i:04d}" for i in range(size)]
    rng.shuffle(subjects)
    if condition == "high":
        # High similarity uses every key type across examples, but one key type
        # consistently appears on every line within an individual haystack.
        key_type = KEY_TYPES[example_index % len(KEY_TYPES)]
        keys = [key_type] * size
    else:
        # Low similarity mixes key types within the same haystack.
        keys = [KEY_TYPES[i % len(KEY_TYPES)] for i in range(size)]
    values = shuffled_values(keys, example_index, rng)
    records = [[subject, key, value]
               for subject, key, value in zip(subjects, keys, values)]
    target_index = rng.randrange(size)
    target = records.pop(target_index)
    target_position = position_index(position, size)
    records.insert(target_position, target)
    target_key = [target[0], target[1]]
    values = [record[2] for record in records]
    if len(subjects) != len(set(subjects)):
        raise AssertionError("Generated subjects are not unique")
    if len(values) != len(set(values)):
        raise AssertionError("Generated values are not unique")
    if records[target_position] != target:
        raise AssertionError("Target was not placed at the requested position")
    return {
        "example_id": f"{size}_{condition}_{position}_{example_index:04d}",
        "ordered_kv_records": records,
        "target_key": target_key,
        "target_value": target[2],
        "position": target_position,
        "position_label": position,
        "similarity_condition": condition,
        "size": size,
        "seed": seed,
    }


def generate_dataset(
    output_dir: Path,
    sizes: Iterable[int] = DEFAULT_SIZES,
    conditions: Iterable[str] = DEFAULT_CONDITIONS,
    examples_per_position: int = 100,
    seed: int = 42,
    positions: Iterable[str] = ("beginning", "middle", "end"),
) -> list[Path]:
    """Write one JSONL file per size/condition/position."""
    if examples_per_position < 1:
        raise ValueError("examples_per_position must be positive")
    positions = tuple(positions)
    if not positions:
        raise ValueError("positions must not be empty")
    condition_offsets = {"high": 17, "low": 31}
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for size in sizes:
        for condition in conditions:
            for position in positions:
                path = output_dir / f"kv_dataset_{size}_{condition}_{position}.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for index in range(examples_per_position):
                        example_seed = (seed + size * 1_000_000 + index * 101
                                        + condition_offsets.get(condition, 0)
                                        + positions.index(position) * 10_000)
                        example = make_example(size, condition, position, index, example_seed)
                        handle.write(json.dumps(example, ensure_ascii=True) + "\n")
                paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--examples-per-position", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positions", nargs="+", choices=POSITIONS,
                        default=["beginning", "middle", "end"])
    args = parser.parse_args()
    paths = generate_dataset(args.output_dir, examples_per_position=args.examples_per_position,
                             seed=args.seed, positions=args.positions)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
