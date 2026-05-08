import tkinter as tk

import customtkinter as ctk
import numpy as np
from gui.theme import ACCENT_CYAN, ACCENT_PURPLE, BG_CANVAS, BG_SURFACE, BORDER_CYAN, FONT_TITLE, TEXT_DIM, TEXT_MAIN

BG_MID = BG_SURFACE


class HistogramWindow(ctk.CTkToplevel):
    def __init__(self, parent, histogram, title="Image Histogram", histogram_after=None, title_after=None):
        super().__init__(parent)
        self.title(title)
        self.configure(fg_color=BG_SURFACE)
        self._hist_before = np.asarray(histogram, dtype=np.int64)
        self._hist_after  = np.asarray(histogram_after, dtype=np.int64) if histogram_after is not None else None
        self._scale_var   = tk.StringVar(value="Log")
        self._interp_var  = tk.StringVar(value="Bars")   # "Bars" | "Smooth"

        if histogram_after is not None:
            self.geometry("1800x750")
            self.minsize(1400, 700)
        else:
            self.geometry("1300x750")
            self.minsize(1100, 700)

        self.resizable(True, True)

        ctk.CTkLabel(self, text=title, font=FONT_TITLE, text_color=TEXT_MAIN).pack(pady=(10, 6))

        self._summary_var = tk.StringVar()
        ctk.CTkLabel(self, textvariable=self._summary_var, text_color=TEXT_DIM).pack(pady=(0, 4))

        # ── Controls row ──────────────────────────────────────────────
        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(fill="x", padx=12, pady=(0, 6))

        ctk.CTkLabel(controls, text="Scale:", text_color=TEXT_DIM).pack(side="left")
        ctk.CTkSegmentedButton(
            controls,
            values=["Linear", "Log"],
            variable=self._scale_var,
            command=lambda _v: self._redraw(),
            width=180,
        ).pack(side="left", padx=8)

        ctk.CTkLabel(controls, text="Display:", text_color=TEXT_DIM).pack(side="left", padx=(16, 0))
        ctk.CTkSegmentedButton(
            controls,
            values=["Bars", "Smooth"],
            variable=self._interp_var,
            command=lambda _v: self._redraw(),
            width=180,
        ).pack(side="left", padx=8)

        # ── Chart area ────────────────────────────────────────────────
        frame = ctk.CTkFrame(self, fg_color=BG_SURFACE, border_width=1, border_color=BORDER_CYAN)
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        if histogram_after is not None:
            left_frame = ctk.CTkFrame(frame, fg_color=BG_SURFACE, border_width=1, border_color=BORDER_CYAN)
            left_frame.pack(side="left", fill="both", expand=True, padx=5)
            right_frame = ctk.CTkFrame(frame, fg_color=BG_SURFACE, border_width=1, border_color=BORDER_CYAN)
            right_frame.pack(side="right", fill="both", expand=True, padx=5)

            ctk.CTkLabel(left_frame, text="Before Equalization",
                         font=("Segoe UI", 11, "bold"), text_color=TEXT_MAIN).pack(pady=4)
            self._canvas_before = tk.Canvas(left_frame, bg=BG_CANVAS, highlightthickness=0)
            self._canvas_before.pack(fill="both", expand=True, padx=5, pady=5)
            self._canvas_before.bind("<Configure>", lambda _e: self._redraw())

            ctk.CTkLabel(right_frame, text=title_after or "After Equalization",
                         font=("Segoe UI", 11, "bold"), text_color=TEXT_MAIN).pack(pady=4)
            self._canvas_after = tk.Canvas(right_frame, bg=BG_CANVAS, highlightthickness=0)
            self._canvas_after.pack(fill="both", expand=True, padx=5, pady=5)
            self._canvas_after.bind("<Configure>", lambda _e: self._redraw())

            self._draw_histogram_comparison(self._hist_before, self._hist_after)
        else:
            self._canvas = tk.Canvas(frame, bg=BG_CANVAS, highlightthickness=0)
            self._canvas.pack(fill="both", expand=True, padx=10, pady=10)
            self._canvas.bind("<Configure>", lambda _e: self._redraw())
            self._draw_histogram(self._hist_before)

    # ── Redraw dispatcher ─────────────────────────────────────────────

    def _redraw(self):
        if self._hist_after is not None:
            self._draw_histogram_comparison(self._hist_before, self._hist_after)
        else:
            self._draw_histogram(self._hist_before)

    # ── Single-canvas draw (used for the non-comparison case) ─────────

    def _draw_histogram(self, histogram):
        self.update_idletasks()
        width  = max(self._canvas.winfo_width(),  900)
        height = max(self._canvas.winfo_height(), 500)
        self._canvas.delete("all")

        L, T, R, B = 70, 30, width - 30, height - 50
        max_count  = max(int(histogram.max()) if histogram.size else 1, 1)
        total      = int(histogram.sum()) if histogram.size else 0
        peak_bin   = int(histogram.argmax()) if histogram.size else 0

        self._summary_var.set(
            f"Total pixels: {total}   Peak: {max_count} @ intensity {peak_bin}"
            f"   Scale: {self._scale_var.get()}   Display: {self._interp_var.get()}"
        )

        self._canvas.create_rectangle(L, T, R, B, outline=BORDER_CYAN, width=2)
        self._draw_y_ticks(self._canvas, L, T, B, max_count)
        self._draw_x_ticks(self._canvas, L, R, B, height)

        if self._interp_var.get() == "Smooth":
            self._draw_smooth(self._canvas, histogram, L, T, R, B, max_count, ACCENT_CYAN)
        else:
            self._draw_bars(self._canvas, histogram, L, B, R, max_count, ACCENT_CYAN)

        self._canvas.create_text(L + (R - L) / 2, height - 15,
                                 text="Pixel Intensity (0-255)", fill=TEXT_DIM,
                                 font=("Segoe UI", 10, "bold"))
        self._canvas.create_text(20, T + (B - T) / 2,
                                 text="Pixel Count", fill=TEXT_DIM, angle=90,
                                 font=("Segoe UI", 10, "bold"))

    # ── Side-by-side comparison draw ──────────────────────────────────

    def _draw_histogram_comparison(self, hist_before, hist_after):
        self.update_idletasks()
        self._draw_single_histogram_on_canvas(self._canvas_before, hist_before, ACCENT_CYAN)
        self._draw_single_histogram_on_canvas(self._canvas_after,  hist_after,  ACCENT_PURPLE)

        bc, ac = int(hist_before.sum()), int(hist_after.sum())
        bp, ap = int(hist_before.argmax()), int(hist_after.argmax())
        bm, am = int(hist_before.max()),   int(hist_after.max())
        self._summary_var.set(
            f"Before: {bc} px (peak {bm} @ {bp})   "
            f"After: {ac} px (peak {am} @ {ap})   "
            f"Scale: {self._scale_var.get()}   Display: {self._interp_var.get()}"
        )

    def _draw_single_histogram_on_canvas(self, canvas, histogram, color):
        canvas.delete("all")
        width  = max(canvas.winfo_width(),  820)
        height = max(canvas.winfo_height(), 560)

        L, T, R, B = 70, 30, width - 30, height - 50
        max_count  = max(int(histogram.max()) if histogram.size else 1, 1)

        canvas.create_rectangle(L, T, R, B, outline=BORDER_CYAN, width=2)
        self._draw_y_ticks(canvas, L, T, B, max_count)
        self._draw_x_ticks(canvas, L, R, B, height)

        if self._interp_var.get() == "Smooth":
            self._draw_smooth(canvas, histogram, L, T, R, B, max_count, color)
        else:
            self._draw_bars(canvas, histogram, L, B, R, max_count, color)

        canvas.create_text(L + (R - L) / 2, height - 15,
                           text="Pixel Intensity (0-255)", fill=TEXT_DIM,
                           font=("Segoe UI", 10, "bold"))
        canvas.create_text(20, T + (B - T) / 2,
                           text=f"Pixel Count ({self._scale_var.get()})", fill=TEXT_DIM,
                           angle=90, font=("Segoe UI", 10, "bold"))

    # ── Drawing primitives ────────────────────────────────────────────

    def _draw_bars(self, canvas, histogram, L, B, R, max_count, color):
        pw = R - L
        ph = B - 30        # plot height (top = 30)
        bw = pw / 256.0
        for i, count in enumerate(histogram):
            bar_h = self._scaled_ratio(int(count), max_count) * ph
            x0 = L + i * bw
            x1 = L + (i + 1) * bw
            canvas.create_rectangle(x0, B - bar_h, x1, B, fill=color, outline="")

    def _draw_smooth(self, canvas, histogram, L, T, R, B, max_count, color):
        """Interpolated smooth curve: filled polygon + bright outline."""
        pw = R - L
        ph = B - T
        bw = pw / 256.0

        # Polygon points: bottom-left → curve tops → bottom-right
        pts = [L, B]
        for i, count in enumerate(histogram):
            h = self._scaled_ratio(int(count), max_count) * ph
            pts.extend([L + (i + 0.5) * bw, B - h])
        pts.extend([R, B])

        # Filled area
        canvas.create_polygon(pts, fill=color, outline="")

        # Smooth outline on top for clarity
        curve_pts = pts[2:-2]   # strip the two bottom-corner points
        if len(curve_pts) >= 4:
            canvas.create_line(curve_pts, fill="white", width=1, smooth=True)

    def _draw_y_ticks(self, canvas, L, T, B, max_count):
        for tick in self._build_tick_values(max_count):
            y = B - self._scaled_ratio(tick, max_count) * (B - T)
            canvas.create_line(L - 8, y, L, y, fill=TEXT_DIM, width=1)
            canvas.create_text(L - 12, y, text=str(tick),
                               fill=TEXT_DIM, anchor="e", font=("Segoe UI", 9))

    def _draw_x_ticks(self, canvas, L, R, B, height):
        pw = R - L
        for tick_pos in (0, 64, 128, 192, 255):
            x = L + (tick_pos / 256.0) * pw
            canvas.create_line(x, B, x, B + 5, fill=TEXT_DIM, width=1)
            canvas.create_text(x, B + 18, text=str(tick_pos),
                               fill=TEXT_DIM, anchor="n", font=("Segoe UI", 9))

    # ── Scale helpers ─────────────────────────────────────────────────

    def _scaled_ratio(self, count: int, max_count: int) -> float:
        if max_count <= 0:
            return 0.0
        if self._scale_var.get() == "Log":
            return float(np.log1p(max(count, 0)) / np.log1p(max_count))
        return float(max(count, 0) / max_count)

    def _build_tick_values(self, max_count: int):
        if self._scale_var.get() != "Log":
            return sorted({0, max_count // 4, max_count // 2, (max_count * 3) // 4, max_count})
        ticks, p = [0], 0
        while 10 ** p < max_count:
            ticks.append(10 ** p)
            p += 1
        ticks.append(max_count)
        return sorted(set(ticks))
