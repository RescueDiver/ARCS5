from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "data.json"
TASK_INDEX_FILE = PROJECT_ROOT / "task_groups" / "task_index.json"


# =============================================================================
# DISPLAY SETTINGS
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

UNKNOWN_COLOR = "#FFFFFF"
GRID_LINE_COLOR = "#444444"
BACKGROUND_COLOR = "#202124"
PANEL_BACKGROUND = "#2A2B2E"
TEXT_COLOR = "#F1F3F4"
SUBTEXT_COLOR = "#BDC1C6"
ACCENT_COLOR = "#8AB4F8"

MIN_CELL_SIZE = 10
MAX_CELL_SIZE = 28
DEFAULT_CELL_SIZE = 22


# =============================================================================
# LOADING
# =============================================================================

def load_json(file_path: Path) -> Any:
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find:\n{file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_project_data() -> tuple[dict[str, Any], dict[str, Any]]:
    data = load_json(DATA_FILE)
    index_file = load_json(TASK_INDEX_FILE)

    if not isinstance(data, dict):
        raise ValueError("data.json must contain a dictionary of tasks.")

    if not isinstance(index_file, dict):
        raise ValueError("task_index.json must contain a dictionary.")

    tasks = index_file.get("tasks")

    if not isinstance(tasks, dict):
        raise ValueError(
            "task_index.json is missing its top-level 'tasks' dictionary."
        )

    return data, tasks


# =============================================================================
# SMALL HELPERS
# =============================================================================

def grid_shape(grid: Any) -> tuple[int, int]:
    if not isinstance(grid, list) or not grid:
        return 0, 0

    rows = [row for row in grid if isinstance(row, list)]

    if not rows:
        return 0, 0

    return len(rows), max((len(row) for row in rows), default=0)


def format_shape(grid: Any) -> str:
    height, width = grid_shape(grid)
    return f"{height}x{width}"


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def nested_value(record: dict[str, Any], key: str, default: str = "unknown") -> str:
    value = record.get(key, default)
    return str(value)


def unique_sorted(values: list[str]) -> list[str]:
    return sorted(set(values), key=lambda value: (value.lower(), value))


# =============================================================================
# FILTERING
# =============================================================================

def classification_matches(
    classification: dict[str, Any],
    size_group: str,
    scene_group: str,
    background_group: str,
    output_size_group: str,
    structure_tag: str,
    measurement_tag: str,
    search_text: str,
) -> bool:
    if size_group != "All":
        if classification.get("size_group") != size_group:
            return False

    if scene_group != "All":
        if classification.get("scene_group") != scene_group:
            return False

    if background_group != "All":
        if classification.get("background_group") != background_group:
            return False

    if output_size_group != "All":
        if classification.get("output_size_group") != output_size_group:
            return False

    if structure_tag != "All":
        if structure_tag not in safe_list(classification.get("structure_tags")):
            return False

    if measurement_tag != "All":
        if measurement_tag not in safe_list(classification.get("measurement_tags")):
            return False

    search_text = search_text.strip().lower()

    if search_text:
        task_id = str(classification.get("task_id", "")).lower()

        searchable_parts = [
            task_id,
            str(classification.get("size_group", "")),
            str(classification.get("scene_group", "")),
            str(classification.get("background_group", "")),
            str(classification.get("output_size_group", "")),
            " ".join(str(tag) for tag in safe_list(classification.get("structure_tags"))),
            " ".join(str(tag) for tag in safe_list(classification.get("measurement_tags"))),
        ]

        searchable_text = " ".join(searchable_parts).lower()

        if search_text not in searchable_text:
            return False

    return True


# =============================================================================
# GRID DRAWING
# =============================================================================

class GridPanel(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        grid: Any,
        cell_size: int = DEFAULT_CELL_SIZE,
    ) -> None:
        super().__init__(
            parent,
            bg=PANEL_BACKGROUND,
            highlightthickness=1,
            highlightbackground="#3C4043",
        )

        self.grid_data = grid if isinstance(grid, list) else []
        self.cell_size = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, cell_size))

        title_label = tk.Label(
            self,
            text=f"{title}   ({format_shape(self.grid_data)})",
            bg=PANEL_BACKGROUND,
            fg=TEXT_COLOR,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        title_label.pack(fill="x", padx=8, pady=(7, 4))

        canvas_holder = tk.Frame(self, bg=PANEL_BACKGROUND)
        canvas_holder.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.canvas = tk.Canvas(
            canvas_holder,
            bg="#111111",
            highlightthickness=0,
        )

        horizontal_scrollbar = ttk.Scrollbar(
            canvas_holder,
            orient="horizontal",
            command=self.canvas.xview,
        )

        vertical_scrollbar = ttk.Scrollbar(
            canvas_holder,
            orient="vertical",
            command=self.canvas.yview,
        )

        self.canvas.configure(
            xscrollcommand=horizontal_scrollbar.set,
            yscrollcommand=vertical_scrollbar.set,
        )

        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        canvas_holder.grid_rowconfigure(0, weight=1)
        canvas_holder.grid_columnconfigure(0, weight=1)

        self.draw_grid()

    def draw_grid(self) -> None:
        self.canvas.delete("all")

        height, width = grid_shape(self.grid_data)

        if height == 0 or width == 0:
            self.canvas.create_text(
                15,
                15,
                text="No grid",
                anchor="nw",
                fill=TEXT_COLOR,
                font=("Segoe UI", 10),
            )
            self.canvas.configure(scrollregion=(0, 0, 180, 80))
            return

        for row_index, row in enumerate(self.grid_data):
            if not isinstance(row, list):
                continue

            for column_index, color in enumerate(row):
                left = column_index * self.cell_size
                top = row_index * self.cell_size
                right = left + self.cell_size
                bottom = top + self.cell_size

                fill_color = ARC_COLORS.get(color, UNKNOWN_COLOR)

                self.canvas.create_rectangle(
                    left,
                    top,
                    right,
                    bottom,
                    fill=fill_color,
                    outline=GRID_LINE_COLOR,
                    width=1,
                )

        total_width = width * self.cell_size
        total_height = height * self.cell_size

        visible_width = min(total_width + 2, 720)
        visible_height = min(total_height + 2, 520)

        self.canvas.configure(
            width=max(150, visible_width),
            height=max(120, visible_height),
            scrollregion=(0, 0, total_width, total_height),
        )


# =============================================================================
# TASK REVIEW WINDOW
# =============================================================================

class TaskReviewWindow(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Widget,
        data: dict[str, Any],
        task_index: dict[str, Any],
        task_ids: list[str],
        start_index: int = 0,
    ) -> None:
        super().__init__(parent)

        self.data = data
        self.task_index = task_index
        self.task_ids = task_ids
        self.current_index = max(0, min(start_index, len(task_ids) - 1))

        self.title("ARCs5 Visual Task Review")
        self.configure(bg=BACKGROUND_COLOR)
        self.geometry("1500x900")
        self.minsize(1000, 700)

        self.bind("<Left>", lambda event: self.show_previous())
        self.bind("<Right>", lambda event: self.show_next())
        self.bind("<Escape>", lambda event: self.destroy())

        self.build_window()
        self.show_current_task()

    def build_window(self) -> None:
        header = tk.Frame(self, bg=BACKGROUND_COLOR)
        header.pack(fill="x", padx=12, pady=(10, 5))

        self.position_label = tk.Label(
            header,
            text="",
            bg=BACKGROUND_COLOR,
            fg=ACCENT_COLOR,
            font=("Segoe UI", 12, "bold"),
        )
        self.position_label.pack(side="left")

        self.task_label = tk.Label(
            header,
            text="",
            bg=BACKGROUND_COLOR,
            fg=TEXT_COLOR,
            font=("Consolas", 15, "bold"),
        )
        self.task_label.pack(side="left", padx=(18, 0))

        navigation = tk.Frame(header, bg=BACKGROUND_COLOR)
        navigation.pack(side="right")

        previous_button = ttk.Button(
            navigation,
            text="◀ Previous",
            command=self.show_previous,
        )
        previous_button.pack(side="left", padx=4)

        next_button = ttk.Button(
            navigation,
            text="Next ▶",
            command=self.show_next,
        )
        next_button.pack(side="left", padx=4)

        info_frame = tk.Frame(
            self,
            bg=PANEL_BACKGROUND,
            highlightthickness=1,
            highlightbackground="#3C4043",
        )
        info_frame.pack(fill="x", padx=12, pady=5)

        self.classification_label = tk.Label(
            info_frame,
            text="",
            bg=PANEL_BACKGROUND,
            fg=TEXT_COLOR,
            justify="left",
            anchor="w",
            font=("Segoe UI", 10),
        )
        self.classification_label.pack(fill="x", padx=10, pady=(8, 3))

        self.tags_label = tk.Label(
            info_frame,
            text="",
            bg=PANEL_BACKGROUND,
            fg=SUBTEXT_COLOR,
            justify="left",
            anchor="w",
            wraplength=1400,
            font=("Segoe UI", 9),
        )
        self.tags_label.pack(fill="x", padx=10, pady=(0, 8))

        content_holder = tk.Frame(self, bg=BACKGROUND_COLOR)
        content_holder.pack(fill="both", expand=True, padx=12, pady=(5, 12))

        self.outer_canvas = tk.Canvas(
            content_holder,
            bg=BACKGROUND_COLOR,
            highlightthickness=0,
        )

        outer_vertical_scrollbar = ttk.Scrollbar(
            content_holder,
            orient="vertical",
            command=self.outer_canvas.yview,
        )

        outer_horizontal_scrollbar = ttk.Scrollbar(
            content_holder,
            orient="horizontal",
            command=self.outer_canvas.xview,
        )

        self.outer_canvas.configure(
            yscrollcommand=outer_vertical_scrollbar.set,
            xscrollcommand=outer_horizontal_scrollbar.set,
        )

        self.outer_canvas.grid(row=0, column=0, sticky="nsew")
        outer_vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        outer_horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        content_holder.grid_rowconfigure(0, weight=1)
        content_holder.grid_columnconfigure(0, weight=1)

        self.task_content = tk.Frame(
            self.outer_canvas,
            bg=BACKGROUND_COLOR,
        )

        self.canvas_window = self.outer_canvas.create_window(
            (0, 0),
            window=self.task_content,
            anchor="nw",
        )

        self.task_content.bind(
            "<Configure>",
            lambda event: self.outer_canvas.configure(
                scrollregion=self.outer_canvas.bbox("all")
            ),
        )

        self.outer_canvas.bind(
            "<Configure>",
            self.resize_content_width,
        )

        self.outer_canvas.bind_all(
            "<MouseWheel>",
            self.on_mouse_wheel,
        )

    def resize_content_width(self, event: tk.Event) -> None:
        requested_width = self.task_content.winfo_reqwidth()
        target_width = max(event.width, requested_width)
        self.outer_canvas.itemconfigure(
            self.canvas_window,
            width=target_width,
        )

    def on_mouse_wheel(self, event: tk.Event) -> None:
        self.outer_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def show_previous(self) -> None:
        if not self.task_ids:
            return

        self.current_index = (self.current_index - 1) % len(self.task_ids)
        self.show_current_task()

    def show_next(self) -> None:
        if not self.task_ids:
            return

        self.current_index = (self.current_index + 1) % len(self.task_ids)
        self.show_current_task()

    def classification_text(
        self,
        classification: dict[str, Any],
    ) -> tuple[str, str]:
        size_group = nested_value(classification, "size_group")
        scene_group = nested_value(classification, "scene_group")
        background_group = nested_value(classification, "background_group")
        output_size_group = nested_value(classification, "output_size_group")

        background_candidates = safe_list(
            classification.get("background_candidates")
        )

        summary = (
            f"Size: {size_group}    |    "
            f"Scene: {scene_group}    |    "
            f"Background: {background_group} "
            f"{background_candidates}    |    "
            f"Output size: {output_size_group}"
        )

        structure_tags = safe_list(classification.get("structure_tags"))
        measurement_tags = safe_list(classification.get("measurement_tags"))

        tags = (
            "Structure: "
            + (", ".join(str(tag) for tag in structure_tags) or "none")
            + "\nMeasurement: "
            + (", ".join(str(tag) for tag in measurement_tags) or "none")
        )

        return summary, tags

    def clear_task_content(self) -> None:
        for child in self.task_content.winfo_children():
            child.destroy()

    def show_current_task(self) -> None:
        if not self.task_ids:
            return

        task_id = self.task_ids[self.current_index]
        task = safe_dict(self.data.get(task_id))
        classification = safe_dict(self.task_index.get(task_id))

        self.position_label.configure(
            text=f"{self.current_index + 1} / {len(self.task_ids)}"
        )
        self.task_label.configure(text=task_id)

        summary, tags = self.classification_text(classification)
        self.classification_label.configure(text=summary)
        self.tags_label.configure(text=tags)

        self.clear_task_content()

        train_pairs = safe_list(task.get("train"))
        test_pairs = safe_list(task.get("test"))

        row_number = 0

        for pair_index, pair in enumerate(train_pairs, start=1):
            if not isinstance(pair, dict):
                continue

            pair_header = tk.Label(
                self.task_content,
                text=f"TRAIN PAIR {pair_index}",
                bg=BACKGROUND_COLOR,
                fg=ACCENT_COLOR,
                font=("Segoe UI", 11, "bold"),
                anchor="w",
            )
            pair_header.grid(
                row=row_number,
                column=0,
                columnspan=2,
                sticky="w",
                padx=4,
                pady=(10, 5),
            )
            row_number += 1

            input_grid = pair.get("input")
            output_grid = pair.get("output")

            input_panel = GridPanel(
                self.task_content,
                title="INPUT",
                grid=input_grid,
            )
            input_panel.grid(
                row=row_number,
                column=0,
                sticky="nsew",
                padx=(4, 6),
                pady=(0, 8),
            )

            output_panel = GridPanel(
                self.task_content,
                title="EXPECTED OUTPUT",
                grid=output_grid,
            )
            output_panel.grid(
                row=row_number,
                column=1,
                sticky="nsew",
                padx=(6, 4),
                pady=(0, 8),
            )

            row_number += 1

        for pair_index, pair in enumerate(test_pairs, start=1):
            if not isinstance(pair, dict):
                continue

            pair_header = tk.Label(
                self.task_content,
                text=f"TEST PAIR {pair_index}",
                bg=BACKGROUND_COLOR,
                fg="#F28B82",
                font=("Segoe UI", 11, "bold"),
                anchor="w",
            )
            pair_header.grid(
                row=row_number,
                column=0,
                columnspan=2,
                sticky="w",
                padx=4,
                pady=(12, 5),
            )
            row_number += 1

            input_grid = pair.get("input")
            output_grid = pair.get("output")

            input_panel = GridPanel(
                self.task_content,
                title="TEST INPUT",
                grid=input_grid,
            )
            input_panel.grid(
                row=row_number,
                column=0,
                sticky="nsew",
                padx=(4, 6),
                pady=(0, 8),
            )

            if output_grid is not None:
                output_panel = GridPanel(
                    self.task_content,
                    title="TEST OUTPUT",
                    grid=output_grid,
                )
            else:
                output_panel = GridPanel(
                    self.task_content,
                    title="TEST OUTPUT NOT PROVIDED",
                    grid=[],
                )

            output_panel.grid(
                row=row_number,
                column=1,
                sticky="nsew",
                padx=(6, 4),
                pady=(0, 8),
            )

            row_number += 1

        self.task_content.grid_columnconfigure(0, weight=1)
        self.task_content.grid_columnconfigure(1, weight=1)

        self.outer_canvas.xview_moveto(0)
        self.outer_canvas.yview_moveto(0)


# =============================================================================
# MAIN BROWSER
# =============================================================================

class TaskBrowserApp(tk.Tk):
    def __init__(
        self,
        data: dict[str, Any],
        task_index: dict[str, Any],
    ) -> None:
        super().__init__()

        self.data = data
        self.task_index = task_index
        self.filtered_task_ids: list[str] = []

        self.title("ARCs5 Task Browser")
        self.configure(bg=BACKGROUND_COLOR)
        self.geometry("1250x780")
        self.minsize(1000, 650)

        self.style = ttk.Style(self)
        self.configure_styles()

        self.build_filter_values()
        self.build_window()
        self.apply_filters()

    def configure_styles(self) -> None:
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.style.configure(
            "Treeview",
            background="#25262A",
            fieldbackground="#25262A",
            foreground=TEXT_COLOR,
            rowheight=27,
            font=("Segoe UI", 9),
        )

        self.style.configure(
            "Treeview.Heading",
            background="#3C4043",
            foreground=TEXT_COLOR,
            font=("Segoe UI", 9, "bold"),
        )

        self.style.map(
            "Treeview",
            background=[("selected", "#445A75")],
            foreground=[("selected", "#FFFFFF")],
        )

    def build_filter_values(self) -> None:
        classifications = list(self.task_index.values())

        self.size_values = ["All", *unique_sorted([
            str(item.get("size_group", "unknown"))
            for item in classifications
        ])]

        self.scene_values = ["All", *unique_sorted([
            str(item.get("scene_group", "uncertain"))
            for item in classifications
        ])]

        self.background_values = ["All", *unique_sorted([
            str(item.get("background_group", "uncertain"))
            for item in classifications
        ])]

        self.output_values = ["All", *unique_sorted([
            str(item.get("output_size_group", "unknown"))
            for item in classifications
        ])]

        self.structure_values = ["All", *unique_sorted([
            str(tag)
            for item in classifications
            for tag in safe_list(item.get("structure_tags"))
        ])]

        self.measurement_values = ["All", *unique_sorted([
            str(tag)
            for item in classifications
            for tag in safe_list(item.get("measurement_tags"))
        ])]

    def build_window(self) -> None:
        title_label = tk.Label(
            self,
            text="ARCs5 TASK BROWSER",
            bg=BACKGROUND_COLOR,
            fg=TEXT_COLOR,
            font=("Segoe UI", 18, "bold"),
        )
        title_label.pack(anchor="w", padx=16, pady=(14, 4))

        subtitle_label = tk.Label(
            self,
            text=(
                "Filter the first-look classifications, then visually inspect "
                "the actual train and test grids."
            ),
            bg=BACKGROUND_COLOR,
            fg=SUBTEXT_COLOR,
            font=("Segoe UI", 10),
        )
        subtitle_label.pack(anchor="w", padx=16, pady=(0, 12))

        filter_frame = tk.Frame(
            self,
            bg=PANEL_BACKGROUND,
            highlightthickness=1,
            highlightbackground="#3C4043",
        )
        filter_frame.pack(fill="x", padx=16, pady=(0, 10))

        self.size_var = tk.StringVar(value="All")
        self.scene_var = tk.StringVar(value="All")
        self.background_var = tk.StringVar(value="All")
        self.output_var = tk.StringVar(value="All")
        self.structure_var = tk.StringVar(value="All")
        self.measurement_var = tk.StringVar(value="All")
        self.search_var = tk.StringVar(value="")

        self.add_filter(
            filter_frame,
            row=0,
            column=0,
            label="Size",
            variable=self.size_var,
            values=self.size_values,
        )

        self.add_filter(
            filter_frame,
            row=0,
            column=1,
            label="Scene",
            variable=self.scene_var,
            values=self.scene_values,
        )

        self.add_filter(
            filter_frame,
            row=0,
            column=2,
            label="Background",
            variable=self.background_var,
            values=self.background_values,
        )

        self.add_filter(
            filter_frame,
            row=0,
            column=3,
            label="Output size",
            variable=self.output_var,
            values=self.output_values,
        )

        self.add_filter(
            filter_frame,
            row=1,
            column=0,
            label="Structure",
            variable=self.structure_var,
            values=self.structure_values,
        )

        self.add_filter(
            filter_frame,
            row=1,
            column=1,
            label="Measurement",
            variable=self.measurement_var,
            values=self.measurement_values,
        )

        search_holder = tk.Frame(filter_frame, bg=PANEL_BACKGROUND)
        search_holder.grid(
            row=1,
            column=2,
            columnspan=2,
            sticky="ew",
            padx=8,
            pady=8,
        )

        search_label = tk.Label(
            search_holder,
            text="Search",
            bg=PANEL_BACKGROUND,
            fg=TEXT_COLOR,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        )
        search_label.pack(fill="x")

        search_entry = ttk.Entry(
            search_holder,
            textvariable=self.search_var,
        )
        search_entry.pack(fill="x", pady=(3, 0))
        search_entry.bind("<Return>", lambda event: self.apply_filters())

        for column in range(4):
            filter_frame.grid_columnconfigure(column, weight=1)

        button_frame = tk.Frame(self, bg=BACKGROUND_COLOR)
        button_frame.pack(fill="x", padx=16, pady=(0, 10))

        apply_button = ttk.Button(
            button_frame,
            text="Apply Filters",
            command=self.apply_filters,
        )
        apply_button.pack(side="left", padx=(0, 6))

        clear_button = ttk.Button(
            button_frame,
            text="Clear Filters",
            command=self.clear_filters,
        )
        clear_button.pack(side="left", padx=6)

        open_button = ttk.Button(
            button_frame,
            text="Open Selected Task",
            command=self.open_selected_task,
        )
        open_button.pack(side="left", padx=6)

        review_button = ttk.Button(
            button_frame,
            text="Review Filtered Set",
            command=self.review_filtered_set,
        )
        review_button.pack(side="left", padx=6)

        self.result_count_label = tk.Label(
            button_frame,
            text="",
            bg=BACKGROUND_COLOR,
            fg=ACCENT_COLOR,
            font=("Segoe UI", 10, "bold"),
        )
        self.result_count_label.pack(side="right")

        table_holder = tk.Frame(self, bg=BACKGROUND_COLOR)
        table_holder.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        columns = (
            "task_id",
            "shape",
            "size",
            "scene",
            "background",
            "output",
        )

        self.tree = ttk.Treeview(
            table_holder,
            columns=columns,
            show="headings",
            selectmode="browse",
        )

        headings = {
            "task_id": "Task ID",
            "shape": "Largest Input",
            "size": "Size",
            "scene": "Scene",
            "background": "Background",
            "output": "Output Size",
        }

        widths = {
            "task_id": 110,
            "shape": 105,
            "size": 130,
            "scene": 100,
            "background": 115,
            "output": 280,
        }

        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(
                column,
                width=widths[column],
                minwidth=80,
                anchor="w",
            )

        vertical_scrollbar = ttk.Scrollbar(
            table_holder,
            orient="vertical",
            command=self.tree.yview,
        )

        horizontal_scrollbar = ttk.Scrollbar(
            table_holder,
            orient="horizontal",
            command=self.tree.xview,
        )

        self.tree.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )

        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        table_holder.grid_rowconfigure(0, weight=1)
        table_holder.grid_columnconfigure(0, weight=1)

        self.tree.bind("<Double-1>", lambda event: self.open_selected_task())
        self.tree.bind("<Return>", lambda event: self.open_selected_task())

    def add_filter(
        self,
        parent: tk.Widget,
        row: int,
        column: int,
        label: str,
        variable: tk.StringVar,
        values: list[str],
    ) -> None:
        holder = tk.Frame(parent, bg=PANEL_BACKGROUND)
        holder.grid(
            row=row,
            column=column,
            sticky="ew",
            padx=8,
            pady=8,
        )

        label_widget = tk.Label(
            holder,
            text=label,
            bg=PANEL_BACKGROUND,
            fg=TEXT_COLOR,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        )
        label_widget.pack(fill="x")

        combo = ttk.Combobox(
            holder,
            textvariable=variable,
            values=values,
            state="readonly",
        )
        combo.pack(fill="x", pady=(3, 0))
        combo.bind("<<ComboboxSelected>>", lambda event: self.apply_filters())

    def clear_filters(self) -> None:
        self.size_var.set("All")
        self.scene_var.set("All")
        self.background_var.set("All")
        self.output_var.set("All")
        self.structure_var.set("All")
        self.measurement_var.set("All")
        self.search_var.set("")
        self.apply_filters()

    def apply_filters(self) -> None:
        matching_ids = []

        for task_id, classification in sorted(self.task_index.items()):
            if not isinstance(classification, dict):
                continue

            if classification_matches(
                classification=classification,
                size_group=self.size_var.get(),
                scene_group=self.scene_var.get(),
                background_group=self.background_var.get(),
                output_size_group=self.output_var.get(),
                structure_tag=self.structure_var.get(),
                measurement_tag=self.measurement_var.get(),
                search_text=self.search_var.get(),
            ):
                matching_ids.append(task_id)

        self.filtered_task_ids = matching_ids
        self.refresh_table()

    def refresh_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        for task_id in self.filtered_task_ids:
            classification = safe_dict(self.task_index.get(task_id))

            largest_input = safe_dict(
                classification.get("largest_train_input")
            )

            shape = (
                f"{largest_input.get('height', 0)}x"
                f"{largest_input.get('width', 0)}"
            )

            self.tree.insert(
                "",
                "end",
                iid=task_id,
                values=(
                    task_id,
                    shape,
                    classification.get("size_group", "unknown"),
                    classification.get("scene_group", "uncertain"),
                    classification.get("background_group", "uncertain"),
                    classification.get("output_size_group", "unknown"),
                ),
            )

        self.result_count_label.configure(
            text=f"{len(self.filtered_task_ids)} task(s)"
        )

        if self.filtered_task_ids:
            first_id = self.filtered_task_ids[0]
            self.tree.selection_set(first_id)
            self.tree.focus(first_id)

    def open_selected_task(self) -> None:
        selection = self.tree.selection()

        if not selection:
            messagebox.showinfo(
                "No task selected",
                "Select a task first.",
                parent=self,
            )
            return

        selected_id = selection[0]

        try:
            start_index = self.filtered_task_ids.index(selected_id)
        except ValueError:
            start_index = 0

        TaskReviewWindow(
            parent=self,
            data=self.data,
            task_index=self.task_index,
            task_ids=self.filtered_task_ids,
            start_index=start_index,
        )

    def review_filtered_set(self) -> None:
        if not self.filtered_task_ids:
            messagebox.showinfo(
                "No matching tasks",
                "The current filters found no tasks.",
                parent=self,
            )
            return

        TaskReviewWindow(
            parent=self,
            data=self.data,
            task_index=self.task_index,
            task_ids=self.filtered_task_ids,
            start_index=0,
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    try:
        data, task_index = load_project_data()
    except Exception as error:
        root = tk.Tk()
        root.withdraw()

        messagebox.showerror(
            "ARCs5 Task Browser",
            str(error),
            parent=root,
        )

        root.destroy()
        return

    app = TaskBrowserApp(
        data=data,
        task_index=task_index,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
