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
    Learn from the task's training pairs, then predict one unknown grid.
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

    prediction = SKILL.apply(
        input_grid=input_grid,
        learned_mapping=learned_mapping,
    )

    return prediction