import json
import random

from generate_dataset import generate_dataset, shuffled_values, value_for
from run_benchmark import build_prompt, score_response


def test_dataset_has_target_at_declared_position(tmp_path):
    paths = generate_dataset(tmp_path, sizes=(75,), examples_per_position=6, seed=42)
    assert len(paths) == 6
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert len(rows) == 6
    assert paths[0].name.endswith("_beginning.jsonl")
    for row in rows:
        subjects = [record[0] for record in row["ordered_kv_records"]]
        assert len(subjects) == len(set(subjects))
        assert subjects != [f"Subject_{i:04d}" for i in range(row["size"])]
        assert row["ordered_kv_records"][row["position"]][0:2] == row["target_key"]
        assert sum(record[2] == row["target_value"] for record in row["ordered_kv_records"]) == 1
        assert len({record[1] for record in row["ordered_kv_records"]}) == 1


def test_high_similarity_covers_all_key_types_and_shuffles_values(tmp_path):
    paths = generate_dataset(tmp_path, sizes=(75,), examples_per_position=5, seed=42)
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert {row["ordered_kv_records"][0][1] for row in rows} == {
        "date of birth", "favorite color", "pet name", "phone number", "employee ID"
    }
    for row in rows:
        values = [record[2] for record in row["ordered_kv_records"]]
        assert len(values) == len(set(values))

    keys = ["phone number"] * 75
    generated_order = [value_for("phone number", i) for i in range(75)]
    assert shuffled_values(keys, example_index=0, rng=random.Random(42)) != generated_order


def test_prompt_and_scoring():
    example = {
        "ordered_kv_records": [["Alice", "date of birth", "1 January 2000"]],
        "target_key": ["Alice", "date of birth"],
    }
    assert "Alice's date of birth is 1 January 2000" in build_prompt(example)
    assert score_response("ANSWER: 1 January 2000", "1 January 2000") == (True, "correct")
    assert score_response("ANSWER: 1 January 1999", "1 January 2000") == (False, "wrong_answer")
    assert score_response("The answer is 1 January 2000.", "1 January 2000") == (False, "wrong_format")
    assert score_response("I don't know.", "1 January 2000") == (False, "no_answer")
    assert score_response("ANSWER: 1 January 2000\nANSWER: extra", "1 January 2000") == (
        False, "wrong_format"
    )
