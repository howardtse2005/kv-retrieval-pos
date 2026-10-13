import json
import random
import re
from collections import Counter

from generate_dataset import BASE_VALUES, KEY_TYPES, POSITIONS, generate_dataset, shuffled_values, value_for
from run_benchmark import SYSTEM_PROMPT, build_prompt, score_response


def test_dataset_has_target_at_declared_position(tmp_path):
    paths = generate_dataset(tmp_path, sizes=(75,), examples_per_position=6, seed=42)
    assert len(paths) == 10
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert len(rows) == 6
    assert paths[0].name.endswith("_beginning.jsonl")
    for row in rows:
        subjects = [record[0] for record in row["ordered_kv_records"]]
        assert len(subjects) == len(set(subjects))
        assert all(re.fullmatch(r"Subject_[0-9a-f]{32}", subject) for subject in subjects)
        assert subjects != sorted(subjects)
        assert row["ordered_kv_records"][row["position"]][0:2] == row["target_key"]
        assert sum(record[2] == row["target_value"] for record in row["ordered_kv_records"]) == 1
        assert len({record[1] for record in row["ordered_kv_records"]}) == 1


def test_high_similarity_covers_all_key_types_and_shuffles_values(tmp_path):
    assert len(KEY_TYPES) == 100
    assert all(len(values) == 3 for values in BASE_VALUES.values())
    assert BASE_VALUES["favorite color"] == ("red", "purple", "green")
    paths = generate_dataset(tmp_path, sizes=(75,), examples_per_position=100, seed=42)
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert {row["ordered_kv_records"][0][1] for row in rows} == set(KEY_TYPES)
    for row in rows:
        values = [record[2] for record in row["ordered_kv_records"]]
        assert len(values) == len(set(values))

    keys = ["phone number"] * 75
    generated_order = [value_for("phone number", i) for i in range(75)]
    assert shuffled_values(keys, example_index=0, rng=random.Random(42)) != generated_order


def test_five_positions_and_128_bit_non_date_values(tmp_path):
    paths = generate_dataset(tmp_path, sizes=(75,), examples_per_position=2, seed=42)
    assert len(paths) == 10
    assert any(path.name.endswith("_25th_percentile.jsonl") for path in paths)
    assert any(path.name.endswith("_50th_percentile.jsonl") for path in paths)
    assert any(path.name.endswith("_75th_percentile.jsonl") for path in paths)
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    assert {row["position_label"] for row in rows} == set(POSITIONS)
    for path in paths:
        is_high = "_high_" in path.name
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for subject, key, value in row["ordered_kv_records"]:
                if key != "date of birth" and is_high:
                    assert re.fullmatch(r".+-[0-9a-f]{32}", value)
                    prefix, suffix = value.rsplit("-", 1)
                    assert prefix in BASE_VALUES[key]
                    assert len(suffix) == 32
                elif key != "date of birth":
                    assert value in BASE_VALUES[key]
                    assert not re.search(r"-[0-9a-f]{32}$", value)


def test_low_similarity_uses_all_categories_and_base_values(tmp_path):
    for size, maximum in ((75, 1), (140, 2), (300, 3)):
        paths = generate_dataset(
            tmp_path / str(size), sizes=(size,), conditions=("low",),
            examples_per_position=1, seed=42
        )
        for path in paths:
            row = json.loads(path.read_text().splitlines()[0])
            counts = Counter(record[1] for record in row["ordered_kv_records"])
            assert len(counts) == min(size, len(KEY_TYPES))
            assert max(counts.values()) <= maximum
            assert len({record[2] for record in row["ordered_kv_records"]}) == size
            for _, key, value in row["ordered_kv_records"]:
                assert value in BASE_VALUES[key]
                assert not re.search(r"-[0-9a-f]{32}$", value)


def test_prompt_and_scoring():
    example = {
        "ordered_kv_records": [["Alice", "date of birth", "1 January 2000"]],
        "target_key": ["Alice", "date of birth"],
    }
    assert "Alice's date of birth is 1 January 2000" in build_prompt(example)
    assert "ANSWER: <one answer>" in SYSTEM_PROMPT
    assert "favorite color is red-" in SYSTEM_PROMPT
    assert "'ANSWER: red'" in SYSTEM_PROMPT
    assert "ANSWER:" not in build_prompt(example)
    assert score_response("ANSWER: 1 January 2000", "1 January 2000") == (True, "correct")
    assert score_response("ANSWER: 1 January 1999", "1 January 2000") == (False, "wrong_answer")
    assert score_response("The answer is 1 January 2000.", "1 January 2000") == (False, "wrong_format")
    assert score_response("I don't know.", "1 January 2000") == (False, "no_answer")
    assert score_response("ANSWER: 1 January 2000\nANSWER: extra", "1 January 2000") == (
        False, "wrong_format"
    )
