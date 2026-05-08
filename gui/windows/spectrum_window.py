"""Spectrum viewer + notch filter UI (Members 1 & 2)."""
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from gui.theme import (
    ACCENT_CYAN, BG_CANVAS, BG_SIDEBAR, BG_SURFACE,
    BORDER_CYAN, ERROR, FONT_SMALL, FONT_TITLE, SUCCESS, TEXT_DIM, TEXT_MAIN,
)
from core.frequency.fft import compute_fft
from core.frequency.notch import (
    apply_notch_filter,
    butterworth_notch_mask,
    gaussian_notch_mask,
    ideal_notch_mask,
)

_NOTCH_COLOR = "#FF4444"
_CONJ_COLOR  = "#4488FF"
_DOT_R       = 6   # visual dot radius in canvas pixels


class SpectrumWindow(ctk.CTkToplevel):
    """Side-by-side original / log-magnitude spectrum.  Click spectrum to place notches."""

    def __init__(self, parent, pipeline, on_image_updated, on_pipeline_updated, on_status):
        super().__init__(parent)
        self.title("FFT Spectrum & Notch Filter")
        self.geometry("1100x720")
        self.resizable(True, True)
        self.configure(fg_color=BG_SURFACE)

        self._pipeline           = pipeline
        self._on_image_updated   = on_image_updated
        self._on_pipeline_updated= on_pipeline_updated
        self._on_status          = on_status

        # Member 1 — compute FFT on current pipeline image
        self._original              = pipeline.current_image.copy()
        self._F_shifted, self._mag  = compute_fft(self._original)   # shared complex array
        self._rows, self._cols      = self._F_shifted.shape

        # Member 2 — notch list
        self._notches: list[tuple[int, int]] = []

        # Display geometry (filled when canvases are drawn)
        self._orig_x0 = self._orig_y0 = self._orig_sc = 0.0
        self._spec_x0 = self._spec_y0 = self._spec_sc = 0.0

        # Keep PhotoImage refs alive
        self._orig_photo = self._spec_photo = None

        self._build()
        self.after(60, self._redraw_all)

    # ──────────────────────────────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────────────────────────────

    def _build(self):
        ctk.CTkLabel(self, text="FFT Spectrum & Notch Filter",
                     font=FONT_TITLE, text_color=TEXT_MAIN).pack(pady=(10, 4))

        panels = ctk.CTkFrame(self, fg_color=BG_SURFACE)
        panels.pack(fill="both", expand=True, padx=10, pady=4)

        # ── Left: original image ──
        lf = ctk.CTkFrame(panels, fg_color=BG_SURFACE,
                          border_width=1, border_color=BORDER_CYAN)
        lf.pack(side="left", fill="both", expand=True, padx=(0, 4))
        ctk.CTkLabel(lf, text="Original Image",
                     font=FONT_SMALL, text_color=TEXT_DIM).pack(pady=2)
        self._orig_canvas = tk.Canvas(lf, bg=BG_CANVAS, highlightthickness=0)
        self._orig_canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self._orig_canvas.bind("<Configure>", lambda _e: self._redraw_all())

        # ── Right: log-magnitude spectrum (clickable) ──
        rf = ctk.CTkFrame(panels, fg_color=BG_SURFACE,
                          border_width=1, border_color=BORDER_CYAN)
        rf.pack(side="right", fill="both", expand=True, padx=(4, 0))
        ctk.CTkLabel(rf, text="Log-Magnitude Spectrum  (click to place notch)",
                     font=FONT_SMALL, text_color=TEXT_DIM).pack(pady=2)
        self._spec_canvas = tk.Canvas(rf, bg=BG_CANVAS, highlightthickness=0,
                                       cursor="crosshair")
        self._spec_canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self._spec_canvas.bind("<Configure>",    lambda _e: self._redraw_all())
        self._spec_canvas.bind("<ButtonPress-1>", self._on_spectrum_click)

        # ── Controls ──
        ctrl = ctk.CTkFrame(self, fg_color=BG_SIDEBAR,
                             border_width=1, border_color=BORDER_CYAN)
        ctrl.pack(fill="x", padx=10, pady=(0, 10))

        row1 = ctk.CTkFrame(ctrl, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(row1, text="Filter Type:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(side="left", padx=(0, 4))
        self._filter_var = tk.StringVar(value="Ideal")
        ctk.CTkOptionMenu(row1, variable=self._filter_var,
                          values=["Ideal", "Butterworth", "Gaussian"],
                          width=130).pack(side="left", padx=(0, 20))

        ctk.CTkLabel(row1, text="D₀ radius:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(side="left", padx=(0, 4))
        self._d0_var = tk.StringVar(value="20")
        ctk.CTkEntry(row1, textvariable=self._d0_var, width=70).pack(side="left", padx=(0, 20))

        ctk.CTkLabel(row1, text="Order n (BW):", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(side="left", padx=(0, 4))
        self._n_var = tk.StringVar(value="2")
        ctk.CTkEntry(row1, textvariable=self._n_var, width=60).pack(side="left")

        row2 = ctk.CTkFrame(ctrl, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(4, 10))

        ctk.CTkLabel(row2, text="Notches:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(side="left", padx=(0, 6))
        self._notch_label_var = tk.StringVar(value="None")
        ctk.CTkLabel(row2, textvariable=self._notch_label_var,
                     font=FONT_SMALL, text_color=ACCENT_CYAN,
                     wraplength=500, justify="left").pack(side="left", padx=(0, 16))

        ctk.CTkButton(row2, text="Clear Notches", width=110,
                      fg_color=ERROR, hover_color="#9E3E49",
                      command=self._clear_notches).pack(side="right", padx=(4, 0))
        ctk.CTkButton(row2, text="Apply Notch Filter", width=145,
                      fg_color=SUCCESS, hover_color="#0C6541",
                      command=self._apply_notch).pack(side="right", padx=(0, 4))

    # ──────────────────────────────────────────────────────────────────────
    # Drawing helpers
    # ──────────────────────────────────────────────────────────────────────

    def _redraw_all(self):
        self._draw_array(self._orig_canvas, self._original,
                         "_orig_photo", "_orig_x0", "_orig_y0", "_orig_sc")
        self._draw_array(self._spec_canvas, self._mag,
                         "_spec_photo", "_spec_x0", "_spec_y0", "_spec_sc")
        self._draw_notch_markers()

    def _draw_array(self, canvas, array, photo_attr, x0_attr, y0_attr, sc_attr):
        """Render a numpy array on a Canvas and record display geometry."""
        self.update_idletasks()
        cw = max(canvas.winfo_width(),  200)
        ch = max(canvas.winfo_height(), 200)
        canvas.delete("all")

        h, w = array.shape[:2]
        sc = min(cw / w, ch / h)
        dw, dh = int(w * sc), int(h * sc)

        arr8 = np.clip(array, 0, 255).astype(np.uint8)
        pil  = Image.fromarray(arr8, mode="L" if arr8.ndim == 2 else "RGB")
        pil  = pil.resize((dw, dh), Image.NEAREST)
        photo = ImageTk.PhotoImage(pil)
        setattr(self, photo_attr, photo)

        cx, cy = cw // 2, ch // 2
        canvas.create_image(cx, cy, anchor="center", image=photo)
        setattr(self, x0_attr, cx - dw / 2)
        setattr(self, y0_attr, cy - dh / 2)
        setattr(self, sc_attr,  sc)

    # ──────────────────────────────────────────────────────────────────────
    # Coordinate mapping
    # ──────────────────────────────────────────────────────────────────────

    def _freq_to_canvas(self, u, v):
        """FFT frequency (u, v) → spectrum canvas (cx, cy)."""
        cx = self._spec_x0 + (v + self._cols // 2) * self._spec_sc
        cy = self._spec_y0 + (u + self._rows // 2) * self._spec_sc
        return cx, cy

    def _canvas_to_freq(self, cx, cy):
        """Spectrum canvas (cx, cy) → FFT frequency (u, v)."""
        sc = self._spec_sc if self._spec_sc > 0 else 1.0
        v = round((cx - self._spec_x0) / sc - self._cols // 2)
        u = round((cy - self._spec_y0) / sc - self._rows // 2)
        return u, v

    # ──────────────────────────────────────────────────────────────────────
    # Notch placement (Member 2)
    # ──────────────────────────────────────────────────────────────────────

    def _on_spectrum_click(self, event):
        u, v = self._canvas_to_freq(event.x, event.y)
        if (u, v) in self._notches or (-u, -v) in self._notches:
            return
        self._notches.append((u, v))
        self._update_notch_label()
        # Redraw spectrum + markers without re-computing FFT
        self._draw_array(self._spec_canvas, self._mag,
                         "_spec_photo", "_spec_x0", "_spec_y0", "_spec_sc")
        self._draw_notch_markers()

    def _draw_notch_markers(self):
        try:
            D0 = float(self._d0_var.get())
        except ValueError:
            D0 = 20.0
        D0_px = D0 * self._spec_sc

        for (u, v) in self._notches:
            # Primary notch (red)
            cx, cy = self._freq_to_canvas(u, v)
            self._spec_canvas.create_oval(cx - _DOT_R, cy - _DOT_R,
                                           cx + _DOT_R, cy + _DOT_R,
                                           outline=_NOTCH_COLOR, width=2)
            if D0_px > 0:
                self._spec_canvas.create_oval(cx - D0_px, cy - D0_px,
                                               cx + D0_px, cy + D0_px,
                                               outline=_NOTCH_COLOR, width=1, dash=(3, 3))
            # Conjugate (blue)
            cx2, cy2 = self._freq_to_canvas(-u, -v)
            self._spec_canvas.create_oval(cx2 - _DOT_R, cy2 - _DOT_R,
                                           cx2 + _DOT_R, cy2 + _DOT_R,
                                           outline=_CONJ_COLOR, width=2)
            if D0_px > 0:
                self._spec_canvas.create_oval(cx2 - D0_px, cy2 - D0_px,
                                               cx2 + D0_px, cy2 + D0_px,
                                               outline=_CONJ_COLOR, width=1, dash=(3, 3))

    def _update_notch_label(self):
        if not self._notches:
            self._notch_label_var.set("None")
        else:
            self._notch_label_var.set("  ".join(f"({u},{v})" for u, v in self._notches))

    def _clear_notches(self):
        self._notches.clear()
        self._update_notch_label()
        self._draw_array(self._spec_canvas, self._mag,
                         "_spec_photo", "_spec_x0", "_spec_y0", "_spec_sc")

    # ──────────────────────────────────────────────────────────────────────
    # Notch filter application (Member 2)
    # ──────────────────────────────────────────────────────────────────────

    def _apply_notch(self):
        if not self._notches:
            self._on_status("Place at least one notch on the spectrum first.", "warn")
            return

        try:
            D0 = float(self._d0_var.get())
        except ValueError:
            D0 = 20.0
        try:
            n = int(self._n_var.get())
        except ValueError:
            n = 2

        ftype = self._filter_var.get()
        shape = self._F_shifted.shape

        if ftype == "Ideal":
            mask = ideal_notch_mask(shape, self._notches, D0)
        elif ftype == "Butterworth":
            mask = butterworth_notch_mask(shape, self._notches, D0, n)
        else:
            mask = gaussian_notch_mask(shape, self._notches, D0)

        result = apply_notch_filter(self._F_shifted, mask)

        suffix = f", n={n}" if ftype == "Butterworth" else ""
        desc   = f"Notch Filter — {ftype} (D₀={D0}{suffix})"
        self._pipeline.push(result, desc)
        self._on_image_updated(result)
        self._on_pipeline_updated()
        self._on_status(f"Applied: {desc}", "ok")
        self.destroy()
