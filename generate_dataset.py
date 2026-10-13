"""Generate deterministic synthetic KV retrieval datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

DEFAULT_SIZES = (75, 140, 300)
DEFAULT_CONDITIONS = ("high", "low")
POSITIONS = ("beginning", "25th_percentile", "50th_percentile",
             "75th_percentile", "end")
KEY_TYPES = (
    "date of birth", "favorite color", "pet name", "phone number", "employee ID",
    "street address", "city of residence", "country of residence", "nationality",
    "language spoken", "occupation", "department", "job title", "company name",
    "school name", "university name", "degree", "major subject", "graduation year",
    "favorite food", "favorite drink", "favorite movie", "favorite book",
    "favorite song", "favorite band", "favorite sport", "favorite team",
    "favorite holiday", "favorite season", "favorite animal", "hobby",
    "musical instrument", "shirt size", "shoe size", "height", "weight",
    "eye color", "hair color", "blood type", "marital status", "number of siblings",
    "mother's name", "father's name", "birth city", "birth country",
    "email address", "website", "social handle", "passport number", "license number",
    "bank name", "account type", "currency", "tax region", "time zone",
    "preferred currency", "emergency contact", "emergency phone", "allergy",
    "dietary restriction", "medical condition", "insurance provider",
    "insurance number", "doctor name", "hospital name", "room number",
    "vehicle make", "vehicle model", "vehicle color", "vehicle year",
    "license plate", "flight number", "frequent flyer number", "hotel name",
    "room preference", "check-in date", "check-out date", "package carrier",
    "tracking number", "favorite website", "operating system", "device model",
    "screen size", "internet provider", "router model", "wifi network",
    "project name", "project code", "team name", "manager name", "office floor",
    "desk number", "building name", "meeting room", "access level",
    "security question", "preferred contact method", "customer tier",
    "subscription plan", "account status",
)
BASE_VALUES = {
    "date of birth": ("1 January 1950", "15 June 1975", "28 November 2000"),
    "favorite color": ("red", "purple", "green"),
    "pet name": ("Fluffy", "Mittens", "Buddy"),
    "phone number": ("555-0100", "555-0101", "555-0102"),
    "employee ID": ("A7X3B9", "B4K8P2", "C9M1Q7"),
    "street address": ("12 Oak Street", "48 Maple Avenue", "7 Cedar Road"),
    "city of residence": ("London", "Toronto", "Singapore"),
    "country of residence": ("United Kingdom", "Canada", "Republic of Singapore"),
    "nationality": ("British", "Canadian", "Singaporean"),
    "language spoken": ("English", "French", "Mandarin"),
    "occupation": ("teacher", "engineer", "nurse"),
    "department": ("finance", "marketing", "engineering"),
    "job title": ("analyst", "manager", "director"),
    "company name": ("Northstar Labs", "Cedar Systems", "Orion Works"),
    "school name": ("Oakridge School", "Riverside Academy", "Hillcrest School"),
    "university name": ("Oxford University", "McGill University", "Nanyang University"),
    "degree": ("Bachelor of Arts", "Master of Science", "Doctor of Philosophy"),
    "major subject": ("history", "computer science", "biology"),
    "graduation year": ("2018", "2020", "2022"),
    "favorite food": ("pizza", "sushi", "pasta"),
    "favorite drink": ("coffee", "tea", "lemonade"),
    "favorite movie": ("Inception", "Moonlight", "Arrival"),
    "favorite book": ("Dune", "Beloved", "1984"),
    "favorite song": ("Imagine", "Hallelujah", "Yesterday"),
    "favorite band": ("The Beatles", "Radiohead", "Coldplay"),
    "favorite sport": ("football", "tennis", "basketball"),
    "favorite team": ("Arsenal", "Raptors", "Lakers"),
    "favorite holiday": ("Christmas", "Diwali", "New Year"),
    "favorite season": ("spring", "summer", "autumn"),
    "favorite animal": ("dog", "cat", "elephant"),
    "hobby": ("painting", "gardening", "photography"),
    "musical instrument": ("piano", "guitar", "violin"),
    "shirt size": ("small", "medium", "large"),
    "shoe size": ("size 8", "size 9", "size 10"),
    "height": ("170 cm", "180 cm", "190 cm"),
    "weight": ("60 kg", "75 kg", "90 kg"),
    "eye color": ("hazel", "blue", "emerald"),
    "hair color": ("black", "auburn", "blonde"),
    "blood type": ("A positive", "B positive", "O positive"),
    "marital status": ("single", "married", "divorced"),
    "number of siblings": ("0 siblings", "1 sibling", "2 siblings"),
    "mother's name": ("Maria", "Susan", "Fatima"),
    "father's name": ("James", "Robert", "Ahmed"),
    "birth city": ("Manchester", "Vancouver", "Kuala Lumpur"),
    "birth country": ("England", "Canadian territory", "Malaysia"),
    "email address": ("alex@example.com", "sam@example.com", "lee@example.com"),
    "website": ("example.org", "example.net", "example.com"),
    "social handle": ("@alex", "@sam", "@lee"),
    "passport number": ("P1234567", "P2345678", "P3456789"),
    "license number": ("D123456", "D234567", "D345678"),
    "bank name": ("Barclays", "RBC", "DBS"),
    "account type": ("checking", "savings", "business"),
    "currency": ("GBP", "CAD", "SGD"),
    "tax region": ("England tax zone", "Ontario", "Singapore tax zone"),
    "time zone": ("Europe/London", "America/Toronto", "Asia/Singapore"),
    "preferred currency": ("pound sterling", "Canadian dollar", "Singapore dollar"),
    "emergency contact": ("Alice Smith", "Brian Lee", "Carol Wong"),
    "emergency phone": ("555-0110", "555-0111", "555-0112"),
    "allergy": ("none", "peanuts", "pollen"),
    "dietary restriction": ("no restriction", "vegetarian", "vegan"),
    "medical condition": ("no condition", "asthma", "migraine"),
    "insurance provider": ("Aviva", "Manulife", "AIA"),
    "insurance number": ("INS10001", "INS20002", "INS30003"),
    "doctor name": ("Dr Patel", "Dr Chen", "Dr Williams"),
    "hospital name": ("Central Hospital", "Riverside Hospital", "City Hospital"),
    "room number": ("Room 101", "Room 202", "Room 303"),
    "vehicle make": ("Toyota", "Ford", "Honda"),
    "vehicle model": ("Corolla", "Focus", "Civic"),
    "vehicle color": ("silver", "white", "obsidian"),
    "vehicle year": ("2018 model", "2020 model", "2022 model"),
    "license plate": ("ABC123", "DEF456", "GHI789"),
    "flight number": ("BA101", "AC202", "SQ303"),
    "frequent flyer number": ("FF10001", "FF20002", "FF30003"),
    "hotel name": ("Grand Hotel", "Park Hotel", "Harbour Hotel"),
    "room preference": ("window seat", "high floor", "quiet room"),
    "check-in date": ("1 April 2024", "10 May 2024", "20 June 2024"),
    "check-out date": ("5 April 2024", "15 May 2024", "25 June 2024"),
    "package carrier": ("FedEx", "DHL", "UPS"),
    "tracking number": ("TRK10001", "TRK20002", "TRK30003"),
    "favorite website": ("Wikipedia", "YouTube", "GitHub"),
    "operating system": ("Windows", "macOS", "Linux"),
    "device model": ("ThinkPad X1", "MacBook Air", "Surface Laptop"),
    "screen size": ("13 inch", "15 inch", "27 inch"),
    "internet provider": ("Comcast", "Bell", "SingTel"),
    "router model": ("Archer C7", "Eero 6", "Orbi RBR"),
    "wifi network": ("HomeNet", "OfficeNet", "GuestNet"),
    "project name": ("Apollo", "Beacon", "Atlas"),
    "project code": ("PRJ100", "PRJ200", "PRJ300"),
    "team name": ("Red Team", "Blue Team", "Green Team"),
    "manager name": ("Jordan Smith", "Taylor Lee", "Morgan Chen"),
    "office floor": ("Floor 2", "Floor 5", "Floor 8"),
    "desk number": ("Desk 12", "Desk 34", "Desk 56"),
    "building name": ("Oak Building", "Maple Building", "Cedar Building"),
    "meeting room": ("Room Alpha", "Room Bravo", "Room Charlie"),
    "access level": ("standard", "privileged", "administrator"),
    "security question": ("first school", "first pet", "birth city"),
    "preferred contact method": ("email", "phone", "post"),
    "customer tier": ("bronze", "silver tier", "gold"),
    "subscription plan": ("basic", "professional", "enterprise"),
    "account status": ("active", "pending", "suspended"),
}


def position_index(label: str, size: int) -> int:
    """Return the deterministic index for a named target position."""
    if label == "beginning":
        return 0
    if label == "25th_percentile":
        return size // 4
    if label == "50th_percentile":
        return size // 2
    if label == "75th_percentile":
        return (3 * size) // 4
    if label == "end":
        return size - 1
    raise ValueError(f"Unknown position label: {label}")


def value_for(key_type: str, index: int, include_suffix: bool = True) -> str:
    """Generate a human-readable value with a unique 128-bit suffix."""
    if key_type == "date of birth":
        if include_suffix:
            return (date(1950, 1, 1) + timedelta(days=index)).strftime("%-d %B %Y")
        return BASE_VALUES[key_type][index % 3]
    if key_type in KEY_TYPES:
        suffix = hashlib.blake2b(
            f"{key_type}:{index}".encode("utf-8"), digest_size=16
        ).hexdigest()
        prefix = BASE_VALUES[key_type][index % 3]
        return f"{prefix}-{suffix}" if include_suffix else prefix
    raise ValueError(f"Unknown key type: {key_type}")


def shuffled_values(
    keys: list[str],
    example_index: int,
    rng: random.Random,
    include_suffix: bool = True,
) -> list[str]:
    """Load generated values and shuffle them without changing their key type."""
    values_by_key: dict[str, list[str]] = {key: [] for key in KEY_TYPES}
    for index, key in enumerate(keys):
        values_by_key[key].append(
            value_for(key, example_index * len(keys) + index, include_suffix)
        )
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
    subjects = [
        "Subject_" + hashlib.blake2b(
            f"{seed}:subject:{i}".encode("utf-8"), digest_size=16
        ).hexdigest()
        for i in range(size)
    ]
    rng.shuffle(subjects)
    if condition == "high":
        # High similarity uses every key type across examples, but one key type
        # consistently appears on every line within an individual haystack.
        key_type = KEY_TYPES[example_index % len(KEY_TYPES)]
        keys = [key_type] * size
    else:
        # Use every category when possible, with no category appearing more
        # than ceil(size / len(KEY_TYPES)) times.
        keys = [KEY_TYPES[i % len(KEY_TYPES)] for i in range(size)]
    values = shuffled_values(
        keys, example_index, rng, include_suffix=condition == "high"
    )
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
    positions: Iterable[str] = POSITIONS,
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
                        default=list(POSITIONS))
    args = parser.parse_args()
    paths = generate_dataset(args.output_dir, examples_per_position=args.examples_per_position,
                             seed=args.seed, positions=args.positions)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
