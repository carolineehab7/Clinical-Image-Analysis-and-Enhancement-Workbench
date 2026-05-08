import tkinter as tk
import customtkinter as ctk

from gui.theme import (
    ACCENT_CYAN, BG_SIDEBAR, BORDER_CYAN, ERROR, FONT_SMALL,
    FONT_TITLE, SUCCESS, TEXT_DIM, TEXT_MAIN,
)

class ROIPanel:
    def __init__(self, parent, on_toggle_roi_mode, on_clear_roi, on_analyze_roi):
        self._on_toggle  = on_toggle_roi_mode
        self._on_clear   = on_clear_roi
        self._on_analyze = on_analyze_roi
        self._active     = False

        ctk.CTkLabel(parent, text="ROI ANALYSIS",
                     font=FONT_TITLE, text_color=TEXT_MAIN).pack(
                         anchor="w", padx=12, pady=(12, 2))

        ctk.CTkLabel(
            parent,
            text="Click-drag on the image to draw\na region of interest, then analyze\nits local histogram and statistics.",
            font=FONT_SMALL, text_color=TEXT_DIM, justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 6))

        self._draw_btn = ctk.CTkButton(
            parent,
            text="Draw ROI  [OFF]",
            command=self._toggle,
            fg_color=BG_SIDEBAR,
            border_width=1,
            border_color=BORDER_CYAN,
            text_color=TEXT_MAIN,
            hover_color="#1D2838",
        )
        self._draw_btn.pack(padx=12, pady=2, fill="x")

        btn_row = ctk.CTkFrame(parent, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(btn_row, text="Clear ROI", width=105,
                      fg_color=ERROR, hover_color="#9E3E49",
                      command=self._on_clear).pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="Analyze ROI", width=105,
                      fg_color=SUCCESS, hover_color="#0C6541",
                      command=self._on_analyze).pack(side="right", padx=2)

        self._status_var = tk.StringVar(value="No ROI selected.")
        ctk.CTkLabel(parent, textvariable=self._status_var,
                     font=FONT_SMALL, text_color=TEXT_DIM,
                     wraplength=220, justify="left").pack(
                         anchor="w", padx=12, pady=(0, 4))

        ctk.CTkFrame(parent, height=2, fg_color=BORDER_CYAN).pack(
            fill="x", padx=8, pady=8)

    #makes the active flag
    def set_active(self, active: bool):
        self._active = active
        self._refresh_btn()

    #sets the status text
    def set_status(self, text: str):
        self._status_var.set(text)


    def _toggle(self):
        self._active = not self._active
        self._on_toggle(self._active)
        self._refresh_btn()

    #to update button appearance based on active state
    def _refresh_btn(self):
        if self._active:
            self._draw_btn.configure(
                text="Draw ROI  [ON]",
                fg_color=ACCENT_CYAN,
                text_color="#000000",
                hover_color="#007A8F",
            )
        else:
            self._draw_btn.configure(
                text="Draw ROI  [OFF]",
                fg_color=BG_SIDEBAR,
                text_color=TEXT_MAIN,
                hover_color="#1D2838",
            )
