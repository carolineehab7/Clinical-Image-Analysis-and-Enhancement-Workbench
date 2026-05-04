import tkinter as tk

import customtkinter as ctk


FONT_TITLE = ("Segoe UI", 13, "bold")
TEXT_DIM = "#888888"
BG_MID = "#16213e"


class HistogramWindow(ctk.CTkToplevel):
    def __init__(self, parent, histogram, title="Image Histogram"):
        super().__init__(parent)
        self.title(title)
        self.geometry("1100x620")
        self.minsize(980, 560)
        self.resizable(True, True)

        ctk.CTkLabel(self, text=title, font=FONT_TITLE).pack(pady=(10, 6))

        self._summary_var = tk.StringVar()
        ctk.CTkLabel(self, textvariable=self._summary_var, text_color=TEXT_DIM).pack(pady=(0, 8))

        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        self._canvas = tk.Canvas(frame, bg=BG_MID, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self._draw_histogram(histogram)

    def _draw_histogram(self, histogram):
        self.update_idletasks()
        width = max(self._canvas.winfo_width(), 920)
        height = max(self._canvas.winfo_height(), 480)
        self._canvas.delete("all")

        left = 65
        top = 30
        bottom = height - 60
        right = width - 25

        self._canvas.create_rectangle(left, top, right, bottom, outline="#3a4a5a")

        max_count = int(histogram.max()) if histogram.size else 1
        max_count = max(max_count, 1)
        total_count = int(histogram.sum()) if histogram.size else 0
        peak_bin = int(histogram.argmax()) if histogram.size else 0
        self._summary_var.set(f"Total pixels: {total_count}   Peak count: {max_count} at intensity {peak_bin}")

        plot_width = right - left
        plot_height = bottom - top
        bin_width = plot_width / 256.0

        tick_values = [0, max_count // 4, max_count // 2, (max_count * 3) // 4, max_count]
        tick_values = sorted(set(tick_values))
        for tick in tick_values:
            y = bottom - (tick / max_count) * plot_height if max_count else bottom
            self._canvas.create_line(left - 6, y, left, y, fill=TEXT_DIM)
            self._canvas.create_text(left - 10, y, text=str(tick), fill=TEXT_DIM, anchor="e")

        for i, count in enumerate(histogram):
            bar_height = (count / max_count) * plot_height if max_count else 0
            x0 = left + i * bin_width
            x1 = left + (i + 1) * bin_width
            y0 = bottom - bar_height
            self._canvas.create_rectangle(x0, y0, x1, bottom, fill="#1f6aa5", outline="")

        self._canvas.create_text(left, bottom + 18, text="0", fill=TEXT_DIM, anchor="w")
        self._canvas.create_text(right, bottom + 18, text="255", fill=TEXT_DIM, anchor="e")
        self._canvas.create_text(left + plot_width / 2, height - 22,
                                 text="Pixel intensity", fill=TEXT_DIM)
        self._canvas.create_text(18, top + plot_height / 2,
                                 text="Count", fill=TEXT_DIM, angle=90)