"""
Pipeline Panel — Manages UI for operation history, step display, and pipeline controls.
Includes Undo & Reset buttons wired to pipeline state management.
"""

import customtkinter as ctk
import tkinter as tk


FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_SMALL = ("Segoe UI", 10)
FONT_MONO  = ("Courier New", 10)
TEXT_DIM   = "#888888"
ACCENT     = "#1f6aa5"


class PipelinePanel:
    """
    Manages the pipeline UI panel displayed on the right side of the main window.
    Displays operation history, step count, Undo & Reset buttons.
    """

    def __init__(self, parent, on_undo=None, on_reset=None):
        """
        Initialize the pipeline panel.
        
        Args:
            parent: Parent CTk widget (usually the main window).
            on_undo: Callback function for Undo button.
            on_reset: Callback function for Reset button.
        """
        self.parent = parent
        self.on_undo = on_undo or (lambda: None)
        self.on_reset = on_reset or (lambda: None)
        self._build_ui()

    def _build_ui(self):
        """Build the pipeline UI components."""
        # ── Undo & Reset Buttons ───────────────────────────────
        btn_row = ctk.CTkFrame(self.parent, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=8)
        
        ctk.CTkButton(btn_row, text="↩  Undo", width=110, height=34,
                      command=self.on_undo,
                      fg_color="#5a3e8e", corner_radius=6).pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="🔄  Reset", width=110, height=34,
                      command=self.on_reset,
                      fg_color="#8b2500", corner_radius=6).pack(side="left", padx=2)

        self._divider()

        # ── Pipeline Steps Display ─────────────────────────────
        ctk.CTkLabel(self.parent, text="⚙  PIPELINE STEPS", font=FONT_TITLE).pack(
            anchor="w", padx=12, pady=(12, 2))

        self._pipe_box = ctk.CTkTextbox(self.parent, height=350, font=FONT_MONO,
                                        state="disabled")
        self._pipe_box.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(self.parent, text="Step count:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._step_count_var = tk.StringVar(value="0")
        ctk.CTkLabel(self.parent, textvariable=self._step_count_var,
                     font=FONT_TITLE).pack(anchor="w", padx=12)

    def _divider(self):
        """Add a visual divider line."""
        ctk.CTkFrame(self.parent, height=2, fg_color="#0f3460").pack(
            fill="x", padx=8, pady=8)

    def update_display(self, pipeline):
        """
        Update the pipeline steps display and step count.
        
        Args:
            pipeline: The Pipeline instance to display.
        """
        self._pipe_box.configure(state="normal")
        self._pipe_box.delete("1.0", "end")

        for i, step in enumerate(pipeline.steps):
            marker = "▶" if i == pipeline.step_count - 1 else "  "
            self._pipe_box.insert("end", f"{marker} {i}. {step}\n")

        self._pipe_box.see("end")
        self._pipe_box.configure(state="disabled")
        self._step_count_var.set(str(pipeline.step_count))
