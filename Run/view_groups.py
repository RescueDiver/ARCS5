import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any


# =============================================================================
# PATHS
# =============================================================================

# Expected location:
#
# ARCs5/
#     Run/
#         view_groups.py
#
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUP_ROOT = PROJECT_ROOT / "task_groups"


# =============================================================================
# ARC COLORS
# =============================================================================

ARC_COLORS = {
    0: "#000000",  # black
    1: "#0074D9",  # blue
    2: "#FF4136",  # red
    3: "#2ECC40",  # green
    4: "#FFDC00",  # yellow
    5: "#AAAAAA",  # gray
    6: "#F012BE",  # magenta
    7: "#FF851B",  # orange
    8: "#7FDBFF",  # light blue
    9: "#870C25",  # maroon
}


# =============================================================================
# FILE DISCOVERY
# =============================================================================

SKIP_FILE_NAMES = {
    "task_observations.json",
    "task_inventory.json",
}


def find_group_files() -> list[Path]:
    """
    Find every group JSON file recursively inside task_groups.
    """
    if not GROUP_ROOT.exists():
        return []

    group_files: list[Path] = []

    for file_path in GROUP_ROOT.rglob("*.json"):
        if file_path.name in SKIP_FILE_NAMES:
            continue

        group_files.append(file_path)

    return sorted(
        group_files,
        key=lambda path: str(
            path.relative_to(GROUP_ROOT)
        ).lower(),
    )


def display_name(file_path: Path) -> str:
    return str(file_path.relative_to(GROUP_ROOT))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_group(file_path: Path) -> dict[str, dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"{file_path.name} does not contain a task dictionary."
        )

    tasks: dict[str, dict[str, Any]] = {}

    for task_id, task in data.items():
        if isinstance(task, dict) and "train" in task:
            tasks[str(task_id)] = task

    return tasks


# =============================================================================
# GRID HELPERS
# =============================================================================

def grid_shape(grid: Any) -> tuple[int, int]:
    if not isinstance(grid, list) or not grid:
        return 0, 0

    height = len(grid)
    width = max(
        (
            len(row)
            for row in grid
            if isinstance(row, list)
        ),
        default=0,
    )

    return height, width


def is_grid(grid: Any) -> bool:
    if not isinstance(grid, list) or not grid:
        return False

    return all(
        isinstance(row, list)
        for row in grid
    )


def choose_shared_cell_size(
    grids: list[Any],
    maximum_width: int = 340,
    maximum_height: int = 300,
) -> int:
    """
    Choose one cell size for Input, Expected, and Prediction so the three
    panels in a train pair use the same visual scale.
    """
    valid_shapes = [
        grid_shape(grid)
        for grid in grids
        if is_grid(grid)
    ]

    if not valid_shapes:
        return 12

    maximum_grid_height = max(
        height
        for height, _width in valid_shapes
    )

    maximum_grid_width = max(
        width
        for _height, width in valid_shapes
    )

    if maximum_grid_height == 0 or maximum_grid_width == 0:
        return 12

    width_size = maximum_width // maximum_grid_width
    height_size = maximum_height // maximum_grid_height

    return max(
        4,
        min(
            24,
            width_size,
            height_size,
        ),
    )


def draw_grid(
    parent: tk.Widget,
    grid: Any,
    cell_size: int,
) -> tk.Canvas:
    height, width = grid_shape(grid)

    canvas_width = max(1, width * cell_size)
    canvas_height = max(1, height * cell_size)

    canvas = tk.Canvas(
        parent,
        width=canvas_width,
        height=canvas_height,
        background="#202020",
        highlightthickness=1,
        highlightbackground="#666666",
    )

    if not is_grid(grid):
        return canvas

    for row_index, row in enumerate(grid):
        for column_index, color_number in enumerate(row):
            color = ARC_COLORS.get(
                color_number,
                "#FFFFFF",
            )

            x1 = column_index * cell_size
            y1 = row_index * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                outline="#303030",
                width=1,
            )

    return canvas


def get_prediction(pair: dict[str, Any]) -> Any:
    """
    Read a prediction without inventing one.

    The first supported key should be:
        pair["prediction"]

    The extra names are accepted so the viewer can tolerate older experiments.
    """
    for key in (
        "prediction",
        "predicted",
        "predicted_output",
    ):
        prediction = pair.get(key)

        if is_grid(prediction):
            return prediction

    return None


# =============================================================================
# PANEL DRAWING
# =============================================================================

def add_grid_panel(
    parent: tk.Widget,
    title: str,
    grid: Any,
    cell_size: int,
) -> None:
    panel = ttk.Frame(
        parent,
        padding=6,
    )

    panel.pack(
        side="left",
        anchor="n",
    )

    ttk.Label(
        panel,
        text=title,
        font=("Segoe UI", 10, "bold"),
    ).pack(
        anchor="center",
        pady=(0, 5),
    )

    grid_canvas = draw_grid(
        panel,
        grid,
        cell_size,
    )

    grid_canvas.pack(
        anchor="center",
    )


def add_prediction_panel(
    parent: tk.Widget,
    prediction: Any,
    cell_size: int,
) -> None:
    panel = ttk.Frame(
        parent,
        padding=6,
    )

    panel.pack(
        side="left",
        anchor="n",
    )

    ttk.Label(
        panel,
        text="PREDICTION",
        font=("Segoe UI", 10, "bold"),
    ).pack(
        anchor="center",
        pady=(0, 5),
    )

    if is_grid(prediction):
        prediction_canvas = draw_grid(
            panel,
            prediction,
            cell_size,
        )

        prediction_canvas.pack(
            anchor="center",
        )

        return

    placeholder = tk.Frame(
        panel,
        width=180,
        height=90,
        background="#202020",
        highlightthickness=1,
        highlightbackground="#666666",
    )

    placeholder.pack(
        anchor="center",
    )

    placeholder.pack_propagate(False)

    tk.Label(
        placeholder,
        text="NOT GENERATED",
        background="#202020",
        foreground="#DDDDDD",
        font=("Segoe UI", 10, "bold"),
    ).place(
        relx=0.5,
        rely=0.5,
        anchor="center",
    )


def add_arrow(
    parent: tk.Widget,
) -> None:
    ttk.Label(
        parent,
        text="→",
        font=("Segoe UI", 22, "bold"),
    ).pack(
        side="left",
        anchor="center",
        padx=8,
        pady=(28, 0),
    )


# =============================================================================
# SCROLLABLE TASK VIEW
# =============================================================================

class ScrollableFrame(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
    ) -> None:
        super().__init__(parent)

        self.canvas = tk.Canvas(
            self,
            highlightthickness=0,
        )

        self.vertical_scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
        )

        self.horizontal_scrollbar = ttk.Scrollbar(
            self,
            orient="horizontal",
            command=self.canvas.xview,
        )

        self.content = ttk.Frame(self.canvas)

        self.content.bind(
            "<Configure>",
            self._update_scroll_region,
        )

        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.content,
            anchor="nw",
        )

        self.canvas.configure(
            yscrollcommand=self.vertical_scrollbar.set,
            xscrollcommand=self.horizontal_scrollbar.set,
        )

        self.canvas.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.vertical_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        self.horizontal_scrollbar.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        self.grid_rowconfigure(
            0,
            weight=1,
        )

        self.grid_columnconfigure(
            0,
            weight=1,
        )

        self.canvas.bind(
            "<Configure>",
            self._resize_content_width,
        )

        self.canvas.bind_all(
            "<MouseWheel>",
            self._mouse_wheel,
        )

    def _update_scroll_region(
        self,
        _event: tk.Event,
    ) -> None:
        self.canvas.configure(
            scrollregion=self.canvas.bbox("all"),
        )

    def _resize_content_width(
        self,
        event: tk.Event,
    ) -> None:
        requested_width = self.content.winfo_reqwidth()
        width = max(event.width, requested_width)

        self.canvas.itemconfigure(
            self.canvas_window,
            width=width,
        )

    def _mouse_wheel(
        self,
        event: tk.Event,
    ) -> None:
        self.canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units",
        )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class GroupViewer:
    def __init__(
        self,
        root: tk.Tk,
    ) -> None:
        self.root = root

        self.root.title("ARCs5 Group Viewer")
        self.root.geometry("1500x900")

        self.group_files = find_group_files()
        self.file_lookup = {
            display_name(file_path): file_path
            for file_path in self.group_files
        }

        self._build_controls()
        self._build_viewer()

        if self.group_files:
            first_name = display_name(
                self.group_files[0]
            )

            self.group_variable.set(first_name)
            self.load_selected_group()
        else:
            self.status_variable.set(
                f"No group JSON files found in {GROUP_ROOT}"
            )

    def _build_controls(self) -> None:
        control_frame = ttk.Frame(
            self.root,
            padding=10,
        )

        control_frame.pack(
            fill="x",
        )

        ttk.Label(
            control_frame,
            text="Group:",
        ).pack(
            side="left",
            padx=(0, 8),
        )

        self.group_variable = tk.StringVar()

        self.group_box = ttk.Combobox(
            control_frame,
            textvariable=self.group_variable,
            values=list(self.file_lookup),
            state="readonly",
            width=90,
        )

        self.group_box.pack(
            side="left",
            fill="x",
            expand=True,
        )

        self.group_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.load_selected_group(),
        )

        ttk.Button(
            control_frame,
            text="Refresh",
            command=self.refresh_groups,
        ).pack(
            side="left",
            padx=(10, 0),
        )

        self.status_variable = tk.StringVar()

        ttk.Label(
            self.root,
            textvariable=self.status_variable,
            padding=(10, 0, 10, 8),
        ).pack(
            fill="x",
        )

    def _build_viewer(self) -> None:
        self.scrollable = ScrollableFrame(
            self.root,
        )

        self.scrollable.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=(0, 10),
        )

    def refresh_groups(self) -> None:
        current_selection = self.group_variable.get()

        self.group_files = find_group_files()
        self.file_lookup = {
            display_name(file_path): file_path
            for file_path in self.group_files
        }

        names = list(self.file_lookup)

        self.group_box.configure(
            values=names,
        )

        if current_selection in self.file_lookup:
            self.group_variable.set(current_selection)
        elif names:
            self.group_variable.set(names[0])
        else:
            self.group_variable.set("")

        self.load_selected_group()

    def clear_view(self) -> None:
        for child in self.scrollable.content.winfo_children():
            child.destroy()

    def load_selected_group(self) -> None:
        self.clear_view()

        selected_name = self.group_variable.get()

        if not selected_name:
            self.status_variable.set(
                f"No group JSON files found in {GROUP_ROOT}"
            )
            return

        file_path = self.file_lookup.get(selected_name)

        if file_path is None:
            return

        try:
            tasks = load_group(file_path)
        except Exception as error:
            messagebox.showerror(
                "Unable to load group",
                str(error),
            )
            return

        self.status_variable.set(
            f"{selected_name} | {len(tasks)} tasks"
        )

        for task_number, (task_id, task) in enumerate(
            tasks.items(),
            start=1,
        ):
            self.add_task(
                task_number=task_number,
                task_id=task_id,
                task=task,
            )

    def add_task(
        self,
        task_number: int,
        task_id: str,
        task: dict[str, Any],
    ) -> None:
        task_frame = ttk.Frame(
            self.scrollable.content,
            padding=(8, 10),
        )

        task_frame.pack(
            fill="x",
            anchor="w",
        )

        ttk.Label(
            task_frame,
            text=f"{task_number}. Task {task_id}",
            font=("Segoe UI", 12, "bold"),
        ).pack(
            anchor="w",
            pady=(0, 8),
        )

        train_pairs = task.get("train", [])

        for pair_number, pair in enumerate(
            train_pairs,
            start=1,
        ):
            if not isinstance(pair, dict):
                continue

            input_grid = pair.get("input", [])
            expected_grid = pair.get("output", [])
            prediction_grid = get_prediction(pair)

            cell_size = choose_shared_cell_size(
                [
                    input_grid,
                    expected_grid,
                    prediction_grid,
                ]
            )

            pair_frame = ttk.LabelFrame(
                task_frame,
                text=f"Train Pair {pair_number}",
                padding=8,
            )

            pair_frame.pack(
                fill="x",
                anchor="w",
                pady=(0, 10),
            )

            comparison_row = ttk.Frame(
                pair_frame,
            )

            comparison_row.pack(
                anchor="w",
            )

            add_grid_panel(
                comparison_row,
                "INPUT",
                input_grid,
                cell_size,
            )

            add_arrow(
                comparison_row,
            )

            add_grid_panel(
                comparison_row,
                "EXPECTED",
                expected_grid,
                cell_size,
            )

            add_arrow(
                comparison_row,
            )

            add_prediction_panel(
                comparison_row,
                prediction_grid,
                cell_size,
            )

        ttk.Separator(
            self.scrollable.content,
            orient="horizontal",
        ).pack(
            fill="x",
            padx=8,
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    root = tk.Tk()
    GroupViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()

