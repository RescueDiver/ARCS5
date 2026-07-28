from __future__ import annotations

from typing import Any

from learned_skills.color_mapping import SKILL
from observe.observer import observe


Grid = list[list[int]]


def predict_grid(
    train_pairs: list[dict[str, Any]],
    input_grid: Grid,
    label: str = "GRID",
) -> Grid:
    """
    Learn from the supplied training pairs and predict one grid.
    """
    observations = observe(input_grid)

    print()
    print("=" * 60)
    print(f"OBSERVATIONS — {label}")
    print("=" * 60)

    for key, value in observations.items():
        print(f"{key:<30}: {value}")

    learned_mapping = SKILL.learn(train_pairs)

    print()
    print("=" * 60)
    print("LEARNED SKILLS")
    print("=" * 60)

    if learned_mapping is None:
        print("color_mapping                : not applicable")
        print("prediction                   : unchanged input")

        return [
            row[:]
            for row in input_grid
        ]

    print("color_mapping                : applicable")
    print(f"learned_mapping              : {learned_mapping}")

    return SKILL.apply(
        input_grid=input_grid,
        learned_mapping=learned_mapping,
    )


def validate_skill(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """
    Honest leave-one-out validation.

    Each training pair is hidden once. The skill must learn from the remaining
    pairs and correctly reconstruct the hidden output.
    """
    if len(train_pairs) < 2:
        print()
        print("Validation requires at least two training pairs.")
        return False

    all_correct = True

    print()
    print("=" * 60)
    print("HONEST SKILL VALIDATION")
    print("=" * 60)

    for hidden_index, hidden_pair in enumerate(train_pairs):
        learning_pairs = [
            pair
            for pair_index, pair in enumerate(train_pairs)
            if pair_index != hidden_index
        ]

        prediction = predict_grid(
            train_pairs=learning_pairs,
            input_grid=hidden_pair.get("input", []),
            label=f"HIDDEN TRAIN {hidden_index + 1}",
        )

        expected = hidden_pair.get("output", [])
        correct = prediction == expected

        print()
        print(
            f"Train pair {hidden_index + 1}: "
            f"{'CORRECT' if correct else 'WRONG'}"
        )

        if not correct:
            all_correct = False

    print()
    print("=" * 60)
    print(
        "VALIDATION RESULT: "
        f"{'PASSED' if all_correct else 'FAILED'}"
    )
    print("=" * 60)

    return all_correct