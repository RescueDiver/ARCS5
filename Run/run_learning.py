from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from typing import Any

from observe.observer import observe


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "data.json"


# =============================================================================
# ARC COLORS
# =============================================================================

ARC_COLORS = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}


Grid = list[list[int]]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_file: Path) -> dict[str, Any]:
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find data file:\n{data_file}"
        )

    with data_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected data.json to contain a dictionary of task IDs."
        )

    return data


def choose_task(
    data: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    task_id = input("Task ID (blank = first): ").strip()

    if not task_id:
        task_id = next(iter(data))

    if task_id not in data:
        raise ValueError(
            f"Unknown task ID: {task_id}"
        )

    return task_id, data[task_id]


# =============================================================================
# GRID DISPLAY
# =============================================================================

def grid_dimensions(
    grid: Grid,
) -> tuple[int, int]:
    if not grid:
        return 0, 0

    height = len(grid)

    width = max(
        (len(row) for row in grid),
        default=0,
    )

    return height, width


def choose_cell_size(
    grid: Grid,
) -> int:
    height, width = grid_dimensions(grid)
    largest_dimension = max(height, width, 1)

    if largest_dimension <= 10:
        return 34

    if largest_dimension <= 15:
        return 26

    if largest_dimension <= 20:
        return 20

    if largest_dimension <= 30:
        return 15

    return 10


def draw_grid(
    parent: tk.Widget,
    grid: Grid,
    title: str,
) -> tk.Frame:
    outer_frame = tk.Frame(
        parent,
        background="#202020",
        padx=8,
        pady=8,
    )

    title_label = tk.Label(
        outer_frame,
        text=title,
        background="#202020",
        foreground="white",
        font=("Arial", 12, "bold"),
    )
    title_label.pack(pady=(0, 6))

    height, width = grid_dimensions(grid)
    cell_size = choose_cell_size(grid)

    canvas = tk.Canvas(
        outer_frame,
        width=max(width * cell_size, 1),
        height=max(height * cell_size, 1),
        background="#111111",
        highlightthickness=0,
    )
    canvas.pack()

    for row_index, row in enumerate(grid):
        for column_index, color_number in enumerate(row):
            x1 = column_index * cell_size
            y1 = row_index * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=ARC_COLORS.get(
                    color_number,
                    "#FFFFFF",
                ),
                outline="#303030",
                width=1,
            )

    dimensions_label = tk.Label(
        outer_frame,
        text=f"{height} × {width}",
        background="#202020",
        foreground="#BBBBBB",
        font=("Arial", 9),
    )
    dimensions_label.pack(pady=(5, 0))

    return outer_frame


# =============================================================================
# SCROLLABLE WINDOW
# =============================================================================

def create_scrollable_area(
    root: tk.Tk,
) -> tuple[tk.Canvas, tk.Frame]:
    canvas = tk.Canvas(
        root,
        background="#181818",
        highlightthickness=0,
    )

    vertical_scrollbar = tk.Scrollbar(
        root,
        orient="vertical",
        command=canvas.yview,
    )

    horizontal_scrollbar = tk.Scrollbar(
        root,
        orient="horizontal",
        command=canvas.xview,
    )

    canvas.configure(
        yscrollcommand=vertical_scrollbar.set,
        xscrollcommand=horizontal_scrollbar.set,
    )

    vertical_scrollbar.pack(
        side="right",
        fill="y",
    )

    horizontal_scrollbar.pack(
        side="bottom",
        fill="x",
    )

    canvas.pack(
        side="left",
        fill="both",
        expand=True,
    )

    content_frame = tk.Frame(
        canvas,
        background="#181818",
    )

    canvas_window = canvas.create_window(
        (0, 0),
        window=content_frame,
        anchor="nw",
    )

    def update_scroll_region(
        _event: tk.Event,
    ) -> None:
        canvas.configure(
            scrollregion=canvas.bbox("all")
        )

    def resize_content(
        _event: tk.Event,
    ) -> None:
        requested_width = content_frame.winfo_reqwidth()
        visible_width = canvas.winfo_width()

        canvas.itemconfigure(
            canvas_window,
            width=max(
                requested_width,
                visible_width,
            ),
        )

    def mouse_wheel(
        event: tk.Event,
    ) -> None:
        canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units",
        )

    content_frame.bind(
        "<Configure>",
        update_scroll_region,
    )

    canvas.bind(
        "<Configure>",
        resize_content,
    )

    canvas.bind_all(
        "<MouseWheel>",
        mouse_wheel,
    )

    return canvas, content_frame


# =============================================================================
# TASK VIEWER
# =============================================================================

def show_task(
    task_id: str,
    task: dict[str, Any],
) -> None:
    root = tk.Tk()

    root.title(
        f"ARCs5 Task Viewer — {task_id}"
    )

    root.geometry("1300x850")
    root.configure(background="#181818")

    _, content_frame = create_scrollable_area(root)

    heading = tk.Label(
        content_frame,
        text=f"TASK {task_id}",
        background="#181818",
        foreground="white",
        font=("Arial", 18, "bold"),
    )
    heading.pack(pady=(15, 5))

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    summary = tk.Label(
        content_frame,
        text=(
            f"Train pairs: {len(train_pairs)}    "
            f"Test pairs: {len(test_pairs)}"
        ),
        background="#181818",
        foreground="#BBBBBB",
        font=("Arial", 11),
    )
    summary.pack(pady=(0, 15))

    for pair_number, pair in enumerate(
        train_pairs,
        start=1,
    ):
        section = tk.Frame(
            content_frame,
            background="#282828",
            padx=12,
            pady=12,
        )

        section.pack(
            fill="x",
            padx=15,
            pady=8,
        )

        section_title = tk.Label(
            section,
            text=f"TRAIN PAIR {pair_number}",
            background="#282828",
            foreground="white",
            font=("Arial", 14, "bold"),
        )
        section_title.pack(pady=(0, 10))

        grids_frame = tk.Frame(
            section,
            background="#282828",
        )
        grids_frame.pack()

        input_frame = draw_grid(
            grids_frame,
            pair.get("input", []),
            "INPUT",
        )

        input_frame.grid(
            row=0,
            column=0,
            padx=20,
            pady=5,
            sticky="n",
        )

        arrow = tk.Label(
            grids_frame,
            text="→",
            background="#282828",
            foreground="white",
            font=("Arial", 24, "bold"),
        )

        arrow.grid(
            row=0,
            column=1,
            padx=10,
        )

        output_frame = draw_grid(
            grids_frame,
            pair.get("output", []),
            "EXPECTED OUTPUT",
        )

        output_frame.grid(
            row=0,
            column=2,
            padx=20,
            pady=5,
            sticky="n",
        )

    for test_number, test_pair in enumerate(
        test_pairs,
        start=1,
    ):
        section = tk.Frame(
            content_frame,
            background="#303030",
            padx=12,
            pady=12,
        )

        section.pack(
            fill="x",
            padx=15,
            pady=8,
        )

        section_title = tk.Label(
            section,
            text=f"TEST {test_number}",
            background="#303030",
            foreground="white",
            font=("Arial", 14, "bold"),
        )
        section_title.pack(pady=(0, 10))

        test_grid = draw_grid(
            section,
            test_pair.get("input", []),
            "TEST INPUT",
        )
        test_grid.pack(pady=5)

    close_button = tk.Button(
        content_frame,
        text="Close and Print Observations",
        command=root.destroy,
        font=("Arial", 11, "bold"),
        padx=25,
        pady=6,
    )
    close_button.pack(pady=20)

    root.mainloop()


# =============================================================================
# OBSERVATION REPORT
# =============================================================================

def format_value(
    value: Any,
) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"

        parts = [
            f"{key}={item}"
            for key, item in value.items()
        ]

        return ", ".join(parts)

    if isinstance(value, list):
        if not value:
            return "[]"

        return ", ".join(
            str(item)
            for item in value
        )

    if isinstance(value, set):
        return str(sorted(value))

    return str(value)


def print_object_view(
    title: str,
    objects: list[dict],
    detection_available: bool,
    detection_reason: str | None,
) -> None:
    print()
    print(title)
    print("-" * 72)

    if not detection_available:
        print(
            "Object detection unavailable: "
            f"{detection_reason}"
        )
        return

    print(f"Found {len(objects)} candidate objects")

    for object_record in objects:
        object_id = object_record["id"]
        bbox = object_record["bbox"]

        print()
        print(f"Object {object_id}")
        print("-" * 40)

        print(
            f"connectivity        : "
            f"{object_record['connectivity']}"
        )

        print(
            f"color               : "
            f"{object_record['color']}"
        )

        print(
            "bbox                : "
            f"({bbox['top']},{bbox['left']})"
            f"-"
            f"({bbox['bottom']},{bbox['right']})"
        )

        print(
            f"pixel_count         : "
            f"{object_record['pixel_count']}"
        )

        print(
            f"height              : "
            f"{object_record['height']}"
        )

        print(
            f"width               : "
            f"{object_record['width']}"
        )

        print(
            f"bounding_box_area   : "
            f"{object_record['bounding_box_area']}"
        )

        print(
            f"fills_bounding_box  : "
            f"{object_record['fills_bounding_box']}"
        )

        print(
            f"touches_border      : "
            f"{object_record['touches_border']}"
        )

        print(
            f"touches_top         : "
            f"{object_record['touches_top']}"
        )

        print(
            f"touches_bottom      : "
            f"{object_record['touches_bottom']}"
        )

        print(
            f"touches_left        : "
            f"{object_record['touches_left']}"
        )

        print(
            f"touches_right       : "
            f"{object_record['touches_right']}"
        )

        print(
            f"cells               : "
            f"{object_record['cells']}"
        )


def print_observations(
    title: str,
    grid: Grid,
) -> None:
    observations = observe(grid)

    print()
    print(title)
    print("-" * 72)

    hidden_keys = {
        "color_4_objects",
        "color_4_object_count",
        "color_8_objects",
        "color_8_object_count",
    }

    for key, value in observations.items():
        if key in hidden_keys:
            continue

        print(
            f"{key:<32}: "
            f"{format_value(value)}"
        )

    print_object_view(
        title="COLOR OBJECTS — 4 CONNECTED",
        objects=observations.get(
            "color_4_objects",
            [],
        ),
        detection_available=observations.get(
            "object_detection_available",
            False,
        ),
        detection_reason=observations.get(
            "object_detection_reason",
        ),
    )

    print_object_view(
        title="COLOR OBJECTS — 8 CONNECTED",
        objects=observations.get(
            "color_8_objects",
            [],
        ),
        detection_available=observations.get(
            "object_detection_available",
            False,
        ),
        detection_reason=observations.get(
            "object_detection_reason",
        ),
    )
def observe_task(
    task_id: str,
    task: dict[str, Any],
) -> None:
    print()
    print("=" * 72)
    print(f"ARCs5 OBSERVATIONS — TASK {task_id}")
    print("=" * 72)

    train_pairs = task.get("train", [])

    for pair_number, pair in enumerate(
        train_pairs,
        start=1,
    ):
        print()
        print("=" * 72)
        print(f"TRAIN PAIR {pair_number}")
        print("=" * 72)

        print_observations(
            title="INPUT OBSERVATIONS",
            grid=pair.get("input", []),
        )

        print_observations(
            title="OUTPUT OBSERVATIONS",
            grid=pair.get("output", []),
        )

    test_pairs = task.get("test", [])

    for test_number, test_pair in enumerate(
        test_pairs,
        start=1,
    ):
        print()
        print("=" * 72)
        print(f"TEST {test_number}")
        print("=" * 72)

        print_observations(
            title="TEST INPUT OBSERVATIONS",
            grid=test_pair.get("input", []),
        )

    print()
    print("=" * 72)
    print("OBSERVATION COMPLETE")
    print("=" * 72)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    data = load_data(DATA_FILE)

    task_id, task = choose_task(data)

    print()
    print(f"Loaded task: {task_id}")
    print(
        f"Train pairs: "
        f"{len(task.get('train', []))}"
    )
    print(
        f"Test pairs : "
        f"{len(task.get('test', []))}"
    )

    print()
    print("Opening visual task viewer...")

    show_task(
        task_id=task_id,
        task=task,
    )

    observe_task(
        task_id=task_id,
        task=task,
    )


if __name__ == "__main__":
    main()