import customtkinter as ctk
import tkinter as tk
from gui.theme import (
    ACCENT_BLUE, ACCENT_PURPLE, BG_ELEVATED, BG_TEXTBOX,
    BORDER_CYAN, FONT_MONO, FONT_SMALL, FONT_TITLE, SUCCESS, TEXT_DIM, TEXT_MAIN,
)


class PipelinePanel:
    """Right-side panel: Undo / Redo / Reset / Save Log + pipeline history."""

    def __init__(self, parent, on_undo=None, on_reset=None,
                 on_redo=None, on_save_log=None):
        self.parent      = parent
        self.on_undo     = on_undo     or (lambda: None)
        self.on_reset    = on_reset    or (lambda: None)
        self.on_redo     = on_redo     or (lambda: None)
        self.on_save_log = on_save_log or (lambda: None)
        self._build_ui()

    def _build_ui(self):
        self._card = ctk.CTkFrame(
            self.parent,
            fg_color=BG_ELEVATED,
            corner_radius=10,
            border_width=1,
            border_color=BORDER_CYAN,
        )
        self._card.pack(fill="x", padx=10, pady=(0, 8))

        # ── 2×2 button grid (equal width) ──
        btn_frame = ctk.CTkFrame(self._card, fg_color="transparent")
        btn_frame.pack(fill="x", padx=12, pady=(8, 8))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ctk.CTkButton(btn_frame, text="↩  Undo", height=34,
                      command=self.on_undo,
                      fg_color=ACCENT_PURPLE, hover_color="#9F67D8",
                      corner_radius=6).grid(row=0, column=0, sticky="ew", padx=2, pady=2)

        self._redo_btn = ctk.CTkButton(
            btn_frame, text="↪  Redo", height=34,
            command=self.on_redo,
            fg_color=ACCENT_BLUE, hover_color="#3A63CC",
            corner_radius=6,
        )
        self._redo_btn.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        ctk.CTkButton(btn_frame, text="↻  Reset", height=34,
                      command=self.on_reset,
                      fg_color="#B94A57", hover_color="#9E3E49",
                      corner_radius=6).grid(row=1, column=0, sticky="ew", padx=2, pady=2)

        ctk.CTkButton(btn_frame, text="💾  Save", height=34,
                      command=self.on_save_log,
                      fg_color=SUCCESS, hover_color="#0C6541",
                      corner_radius=6).grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        self._divider()

        # ── Pipeline history ──
        ctk.CTkLabel(self._card, text="⚙  PIPELINE STEPS",
                     font=FONT_TITLE, text_color=TEXT_MAIN).pack(
                         anchor="w", padx=12, pady=(12, 2))

        self._pipe_box = ctk.CTkTextbox(
            self._card,
            height=300,
            font=FONT_MONO,
            state="disabled",
            fg_color=BG_TEXTBOX,
            text_color=TEXT_MAIN,
            border_width=1,
            border_color=BORDER_CYAN,
        )
        self._pipe_box.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(self._card, text="Step count:",
                     font=FONT_SMALL, text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._step_count_var = tk.StringVar(value="0")
        ctk.CTkLabel(self._card, textvariable=self._step_count_var,
                     font=FONT_TITLE).pack(anchor="w", padx=12)

    def _divider(self):
        ctk.CTkFrame(self.parent, height=2, fg_color=BORDER_CYAN).pack(
            fill="x", padx=8, pady=8)

    def update_display(self, pipeline):
        """Refresh history box and button states after every pipeline change."""
        self._pipe_box.configure(state="normal")
        self._pipe_box.delete("1.0", "end")

        for i, step in enumerate(pipeline.steps):
            marker = "▶" if i == pipeline.step_count - 1 else "  "
            self._pipe_box.insert("end", f"{marker} {i}. {step}\n")

        self._pipe_box.see("end")
        self._pipe_box.configure(state="disabled")
        self._step_count_var.set(str(pipeline.step_count))

        # Enable/disable Redo based on whether there's anything to redo
        if pipeline.can_redo:
            self._redo_btn.configure(state="normal", fg_color=ACCENT_BLUE)
        else:
            self._redo_btn.configure(state="disabled", fg_color="#2A3A5C")
