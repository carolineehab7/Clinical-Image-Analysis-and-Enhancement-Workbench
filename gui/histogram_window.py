import tkinter as tk

import customtkinter as ctk
import numpy as np


FONT_TITLE = ("Segoe UI", 13, "bold")
TEXT_DIM = "#888888"
BG_MID = "#16213e"


class HistogramWindow(ctk.CTkToplevel):
    def __init__(self, parent, histogram, title="Image Histogram", histogram_after=None, title_after=None):
        super().__init__(parent)
        self.title(title)
        self._hist_before = np.asarray(histogram, dtype=np.int64)
        self._hist_after = np.asarray(histogram_after, dtype=np.int64) if histogram_after is not None else None
        self._scale_var = tk.StringVar(value="Log")
        
        # If comparing two histograms, use much wider window
        if histogram_after is not None:
            self.geometry("1800x750")
            self.minsize(1400, 700)
        else:
            self.geometry("1300x750")
            self.minsize(1100, 700)
        
        self.resizable(True, True)

        ctk.CTkLabel(self, text=title, font=FONT_TITLE).pack(pady=(10, 6))

        self._summary_var = tk.StringVar()
        ctk.CTkLabel(self, textvariable=self._summary_var, text_color=TEXT_DIM).pack(pady=(0, 8))

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

        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        if histogram_after is not None:
            # Side-by-side comparison
            left_frame = ctk.CTkFrame(frame)
            left_frame.pack(side="left", fill="both", expand=True, padx=5)
            
            right_frame = ctk.CTkFrame(frame)
            right_frame.pack(side="right", fill="both", expand=True, padx=5)
            
            # Left: original
            ctk.CTkLabel(left_frame, text="Before Equalization", font=("Segoe UI", 11, "bold")).pack(pady=4)
            self._canvas_before = tk.Canvas(left_frame, bg=BG_MID, highlightthickness=0)
            self._canvas_before.pack(fill="both", expand=True, padx=5, pady=5)
            self._canvas_before.bind("<Configure>", lambda _e: self._redraw())
            
            # Right: after
            ctk.CTkLabel(right_frame, text=title_after or "After Equalization", font=("Segoe UI", 11, "bold")).pack(pady=4)
            self._canvas_after = tk.Canvas(right_frame, bg=BG_MID, highlightthickness=0)
            self._canvas_after.pack(fill="both", expand=True, padx=5, pady=5)
            self._canvas_after.bind("<Configure>", lambda _e: self._redraw())
            
            self._draw_histogram_comparison(self._hist_before, self._hist_after)
        else:
            # Single histogram
            self._canvas = tk.Canvas(frame, bg=BG_MID, highlightthickness=0)
            self._canvas.pack(fill="both", expand=True, padx=10, pady=10)
            self._canvas.bind("<Configure>", lambda _e: self._redraw())
            self._draw_histogram(self._hist_before)

    def _redraw(self):
        if self._hist_after is not None:
            self._draw_histogram_comparison(self._hist_before, self._hist_after)
        else:
            self._draw_histogram(self._hist_before)

    def _draw_histogram(self, histogram):
        self.update_idletasks()
        width = max(self._canvas.winfo_width(), 900)
        height = max(self._canvas.winfo_height(), 500)
        self._canvas.delete("all")

        left = 70
        top = 30
        bottom = height - 50
        right = width - 30

        self._canvas.create_rectangle(left, top, right, bottom, outline="#3a4a5a", width=2)

        max_count = int(histogram.max()) if histogram.size else 1
        max_count = max(max_count, 1)
        total_count = int(histogram.sum()) if histogram.size else 0
        peak_bin = int(histogram.argmax()) if histogram.size else 0
        self._summary_var.set(
            f"Total pixels: {total_count}   Peak: {max_count} pixels at intensity {peak_bin}   Scale: {self._scale_var.get()}"
        )

        plot_width = right - left
        plot_height = bottom - top
        bin_width = plot_width / 256.0

        # Y-axis ticks and labels
        for tick in self._build_tick_values(max_count):
            y = bottom - self._scaled_ratio(tick, max_count) * plot_height
            self._canvas.create_line(left - 8, y, left, y, fill=TEXT_DIM, width=1)
            self._canvas.create_text(left - 12, y, text=str(tick), fill=TEXT_DIM, anchor="e", font=("Segoe UI", 9))

        # Draw bars
        for i, count in enumerate(histogram):
            bar_height = self._scaled_ratio(int(count), max_count) * plot_height
            x0 = left + i * bin_width
            x1 = left + (i + 1) * bin_width
            y0 = bottom - bar_height
            self._canvas.create_rectangle(x0, y0, x1, bottom, fill="#1f6aa5", outline="")

        # X-axis labels and ticks
        tick_positions = [0, 64, 128, 192, 255]
        for tick_pos in tick_positions:
            x = left + (tick_pos / 256.0) * plot_width
            self._canvas.create_line(x, bottom, x, bottom + 5, fill=TEXT_DIM, width=1)
            self._canvas.create_text(x, bottom + 18, text=str(tick_pos), fill=TEXT_DIM, anchor="n", font=("Segoe UI", 9))

        # Axis labels
        self._canvas.create_text(left + plot_width / 2, height - 15,
                                text="Pixel Intensity (0-255)", fill=TEXT_DIM, font=("Segoe UI", 10, "bold"))
        self._canvas.create_text(20, top + plot_height / 2,
                                text="Pixel Count", fill=TEXT_DIM, angle=90, font=("Segoe UI", 10, "bold"))

    def _draw_histogram_comparison(self, histogram_before, histogram_after):
        """Draw two histograms side by side for comparison."""
        self.update_idletasks()
        
        # Draw before on left canvas
        self._draw_single_histogram_on_canvas(
            self._canvas_before, 
            histogram_before,
            color="#1f6aa5"
        )
        
        # Draw after on right canvas
        self._draw_single_histogram_on_canvas(
            self._canvas_after,
            histogram_after,
            color="#2ecc71"
        )
        
        # Update summary
        before_count = int(histogram_before.sum()) if histogram_before.size else 0
        after_count = int(histogram_after.sum()) if histogram_after.size else 0
        before_peak_idx = int(histogram_before.argmax()) if histogram_before.size else 0
        after_peak_idx = int(histogram_after.argmax()) if histogram_after.size else 0
        before_peak_count = int(histogram_before.max()) if histogram_before.size else 0
        after_peak_count = int(histogram_after.max()) if histogram_after.size else 0
        self._summary_var.set(
            "Before: "
            f"{before_count} px (peak {before_peak_count} @ {before_peak_idx})   "
            "After: "
            f"{after_count} px (peak {after_peak_count} @ {after_peak_idx})   "
            f"Scale: {self._scale_var.get()}"
        )

    def _draw_single_histogram_on_canvas(self, canvas, histogram, color):
        """Draw a single histogram on a given canvas."""
        canvas.delete("all")
        width = max(canvas.winfo_width(), 820)
        height = max(canvas.winfo_height(), 560)

        left = 70
        top = 30
        bottom = height - 50
        right = width - 30

        canvas.create_rectangle(left, top, right, bottom, outline="#3a4a5a", width=2)

        max_count = int(histogram.max()) if histogram.size else 1
        max_count = max(max_count, 1)

        plot_width = right - left
        plot_height = bottom - top
        bin_width = plot_width / 256.0

        # Y-axis ticks and labels
        for tick in self._build_tick_values(max_count):
            y = bottom - self._scaled_ratio(tick, max_count) * plot_height
            canvas.create_line(left - 8, y, left, y, fill=TEXT_DIM, width=1)
            canvas.create_text(left - 12, y, text=str(tick), fill=TEXT_DIM, anchor="e", font=("Segoe UI", 9))

        # Draw bars
        for i, count in enumerate(histogram):
            bar_height = self._scaled_ratio(int(count), max_count) * plot_height
            x0 = left + i * bin_width
            x1 = left + (i + 1) * bin_width
            y0 = bottom - bar_height
            canvas.create_rectangle(x0, y0, x1, bottom, fill=color, outline="")

        # X-axis labels and ticks
        tick_positions = [0, 64, 128, 192, 255]
        for tick_pos in tick_positions:
            x = left + (tick_pos / 256.0) * plot_width
            canvas.create_line(x, bottom, x, bottom + 5, fill=TEXT_DIM, width=1)
            canvas.create_text(x, bottom + 18, text=str(tick_pos), fill=TEXT_DIM, anchor="n", font=("Segoe UI", 9))
        
        # Axis labels
        canvas.create_text(left + plot_width / 2, height - 15,
                          text="Pixel Intensity (0-255)", fill=TEXT_DIM, font=("Segoe UI", 10, "bold"))
        canvas.create_text(20, top + plot_height / 2,
                          text=f"Pixel Count ({self._scale_var.get()})", fill=TEXT_DIM, angle=90, font=("Segoe UI", 10, "bold"))

    def _scaled_ratio(self, count: int, max_count: int) -> float:
        if max_count <= 0:
            return 0.0
        if self._scale_var.get() == "Log":
            return float(np.log1p(max(count, 0)) / np.log1p(max_count))
        return float(max(count, 0) / max_count)

    def _build_tick_values(self, max_count: int):
        if self._scale_var.get() != "Log":
            return sorted(set([0, max_count // 4, max_count // 2, (max_count * 3) // 4, max_count]))

        ticks = [0]
        p = 0
        while 10 ** p < max_count:
            ticks.append(10 ** p)
            p += 1
        ticks.append(max_count)
        return sorted(set(ticks))