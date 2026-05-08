"""Local histogram and statistics display popup (Member 4)."""
import tkinter as tk
import customtkinter as ctk

from gui.theme import (
    ACCENT_CYAN, ACCENT_PURPLE, BG_CANVAS, BG_SURFACE, BORDER_CYAN,
    FONT_SMALL, FONT_TITLE, TEXT_DIM, TEXT_MAIN, WARNING,
)
from core.analysis.statistics import compute_local_histogram, compute_mean, compute_variance


class StatisticsWindow(ctk.CTkToplevel):
    """Bar-chart histogram + mean/variance for a selected ROI."""

    def __init__(self, parent, roi, roi_label: str = "Selected ROI"):
        super().__init__(parent)
        self.title(f"ROI Statistics — {roi_label}")
        self.geometry("900x560")
        self.resizable(True, True)
        self.configure(fg_color=BG_SURFACE)

        # Member 4 — compute from scratch
        self._bins     = compute_local_histogram(roi)
        self._mean     = compute_mean(roi)
        self._variance = compute_variance(roi)
        self._n_pixels = roi.size

        self._build(roi_label)
        self.after(60, self._draw)

    # ──────────────────────────────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────────────────────────────

    def _build(self, roi_label: str):
        ctk.CTkLabel(self, text=f"Local Histogram — {roi_label}",
                     font=FONT_TITLE, text_color=TEXT_MAIN).pack(pady=(10, 2))

        # Stats row
        stats = ctk.CTkFrame(self, fg_color="transparent")
        stats.pack(pady=(2, 8))

        ctk.CTkLabel(stats,
                     text=f"Mean: {self._mean:.4f}",
                     font=("Segoe UI", 13, "bold"),
                     text_color=ACCENT_CYAN).pack(side="left", padx=20)

        ctk.CTkLabel(stats,
                     text=f"Variance: {self._variance:.4f}",
                     font=("Segoe UI", 13, "bold"),
                     text_color=ACCENT_PURPLE).pack(side="left", padx=20)

        ctk.CTkLabel(stats,
                     text=f"Pixels: {self._n_pixels}",
                     font=FONT_SMALL, text_color=TEXT_DIM).pack(side="left", padx=20)

        # Histogram canvas
        frame = ctk.CTkFrame(self, fg_color=BG_SURFACE,
                              border_width=1, border_color=BORDER_CYAN)
        frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._canvas = tk.Canvas(frame, bg=BG_CANVAS, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self._canvas.bind("<Configure>", lambda _e: self._draw())

    # ──────────────────────────────────────────────────────────────────────
    # Drawing
    # ──────────────────────────────────────────────────────────────────────

    def _draw(self):
        self.update_idletasks()
        W = max(self._canvas.winfo_width(),  700)
        H = max(self._canvas.winfo_height(), 360)
        self._canvas.delete("all")

        L, T, R, B = 70, 24, W - 20, H - 50
        pw, ph = R - L, B - T
        bw = pw / 256.0

        # Border
        self._canvas.create_rectangle(L, T, R, B, outline=BORDER_CYAN, width=1)

        max_count = max(self._bins) if self._bins else 1
        max_count = max(max_count, 1)

        # Y-axis ticks
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            val = int(frac * max_count)
            y = B - frac * ph
            self._canvas.create_line(L - 5, y, L, y, fill=TEXT_DIM, width=1)
            self._canvas.create_text(L - 8, y, text=str(val),
                                      fill=TEXT_DIM, anchor="e", font=("Segoe UI", 8))

        # Bars (Member 4 histogram)
        for i, count in enumerate(self._bins):
            if count == 0:
                continue
            bar_h = (count / max_count) * ph
            x0 = L + i * bw
            x1 = L + (i + 1) * bw
            self._canvas.create_rectangle(x0, B - bar_h, x1, B,
                                           fill=ACCENT_CYAN, outline="")

        # X-axis labels
        for tick in (0, 64, 128, 192, 255):
            x = L + (tick / 256.0) * pw
            self._canvas.create_line(x, B, x, B + 4, fill=TEXT_DIM, width=1)
            self._canvas.create_text(x, B + 14, text=str(tick),
                                      fill=TEXT_DIM, anchor="n", font=("Segoe UI", 8))

        # Axis labels
        self._canvas.create_text(L + pw / 2, H - 10,
                                  text="Pixel Intensity (0–255)",
                                  fill=TEXT_DIM, font=("Segoe UI", 9, "bold"))
        self._canvas.create_text(14, T + ph / 2,
                                  text="Count", fill=TEXT_DIM,
                                  angle=90, font=("Segoe UI", 9, "bold"))

        # Mean line (yellow dashed)
        mean_x = L + (self._mean / 255.0) * pw
        self._canvas.create_line(mean_x, T, mean_x, B,
                                  fill=WARNING, width=1, dash=(4, 4))
        self._canvas.create_text(mean_x + 2, T + 6, anchor="w",
                                  text=f"μ={self._mean:.1f}",
                                  fill=WARNING, font=("Segoe UI", 8))
