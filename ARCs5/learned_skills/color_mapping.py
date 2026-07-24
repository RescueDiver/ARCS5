from __future__ import annotations

from collections import defaultdict
from typing import Any


Grid = list[list[int]]


class ColorMappingSkill:
    name = "color_mapping"

    def learn(
        self,
        train_pairs: list[dict[str, Any]],
    ) -> dict[int, int] | None:
        """
        Learn whether each input color consistently becomes one output color.

        Only works when every training input and output have the same shape.
        Returns None when the evidence is inconsistent.
        """
        observed_mappings: dict[int, set[int]] = defaultdict(set)

        for pair in train_pairs:
            input_grid = pair.get("input", [])
            output_grid = pair.get("output", [])

            if not self._same_shape(input_grid, output_grid):
                return None

            for row_index, input_row in enumerate(input_grid):
                for column_index, input_color in enumerate(input_row):
                    output_color = output_grid[row_index][column_index]

                    observed_mappings[input_color].add(output_color)

        learned_mapping: dict[int, int] = {}

        for input_color, possible_outputs in observed_mappings.items():
            if len(possible_outputs) != 1:
                return None

            learned_mapping[input_color] = next(
                iter(possible_outputs)
            )

        return learned_mapping

    def apply(
        self,
        input_grid: Grid,
        learned_mapping: dict[int, int],
    ) -> Grid:
        return [
            [
                learned_mapping.get(color, color)
                for color in row
            ]
            for row in input_grid
        ]

    @staticmethod
    def _same_shape(
        first_grid: Grid,
        second_grid: Grid,
    ) -> bool:
        if len(first_grid) != len(second_grid):
            return False

        return all(
            len(first_row) == len(second_row)
            for first_row, second_row in zip(
                first_grid,
                second_grid,
            )
        )


SKILL = ColorMappingSkill()