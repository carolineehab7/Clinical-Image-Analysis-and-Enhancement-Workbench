import customtkinter as ctk
import tkinter as tk
from gui.theme import ACCENT_BLUE, ACCENT_PURPLE, BG_ELEVATED, BG_TEXTBOX, BORDER_CYAN, FONT_MONO, FONT_SMALL, FONT_TITLE, TEXT_DIM, TEXT_MAIN


class PipelinePanel:
    # Right-side panel for Undo/Reset and pipeline history.
    
    def __init__(self, parent, on_undo=None, on_reset=None):
        # Keep callbacks optional so this widget is safe to reuse.
        self.parent = parent
        self.on_undo = on_undo or (lambda: None)      # Run when user clicks Undo
        self.on_reset = on_reset or (lambda: None)    # Run when user clicks Reset
        self._build_ui()

    def _build_ui(self):
        # Row for Undo and Reset.
        self._card = ctk.CTkFrame(
            self.parent,
            fg_color=BG_ELEVATED,
            corner_radius=10,
            border_width=1,
            border_color=BORDER_CYAN,
        )
        self._card.pack(fill="x", padx=10, pady=(0, 8))

        btn_row = ctk.CTkFrame(self._card, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=8)

        # Undo goes back one step in history.
        ctk.CTkButton(btn_row, text="↩  Undo", width=110, height=34,
                      command=self.on_undo,
                      fg_color=ACCENT_PURPLE, hover_color="#9F67D8", corner_radius=6).pack(side="left", padx=2)

        # Reset clears the whole pipeline and returns to the original image.
        ctk.CTkButton(btn_row, text="↻  Reset", width=110, height=34,
                      command=self.on_reset,
                      fg_color="#B94A57", hover_color="#9E3E49", corner_radius=6).pack(side="left", padx=2)

        self._divider()

        # Section title and read-only list of applied steps.
        ctk.CTkLabel(self._card, text="⚙  PIPELINE STEPS", font=FONT_TITLE, text_color=TEXT_MAIN).pack(
            anchor="w", padx=12, pady=(12, 2))

        self._pipe_box = ctk.CTkTextbox(
            self._card,
            height=350,
            font=FONT_MONO,
            state="disabled",
            fg_color=BG_TEXTBOX,
            text_color=TEXT_MAIN,
            border_width=1,
            border_color=BORDER_CYAN,
        )
        self._pipe_box.pack(fill="x", padx=8, pady=4)

        # Small counter for total active steps.
        ctk.CTkLabel(self._card, text="Step count:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._step_count_var = tk.StringVar(value="0")
        ctk.CTkLabel(self._card, textvariable=self._step_count_var,
                     font=FONT_TITLE).pack(anchor="w", padx=12)

    def _divider(self):
        # Visual separator between controls and the steps area.
        ctk.CTkFrame(self.parent, height=2, fg_color=BORDER_CYAN).pack(
            fill="x", padx=8, pady=8)

    def update_display(self, pipeline):
        # Refresh the history box after every pipeline change.
        self._pipe_box.configure(state="normal")
        self._pipe_box.delete("1.0", "end")

        # List steps and mark the current one.
        for i, step in enumerate(pipeline.steps):
            marker = "▶" if i == pipeline.step_count - 1 else "  "
            self._pipe_box.insert("end", f"{marker} {i}. {step}\n")

        # Keep the latest update visible.
        self._pipe_box.see("end")
        self._pipe_box.configure(state="disabled")

        # Sync the numeric step counter.
        self._step_count_var.set(str(pipeline.step_count))
