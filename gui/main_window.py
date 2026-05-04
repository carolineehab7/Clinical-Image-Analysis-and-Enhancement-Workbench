"""
Clinical Image Analysis Workbench — Main GUI
Built with CustomTkinter for a modern dark-themed desktop interface.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from core.image_io import load_image, save_image
from core.pipeline import Pipeline
from core.interpolation import nearest_neighbor_zoom, bilinear_zoom
from core.filters import (average_filter, gaussian_filter,
                           sobel_filter, prewitt_filter, median_filter)
from core.histogram import local_histogram_equalization
from gui.pipeline_panel import PipelinePanel


# ──────────────────────────────────────────────────────────────
# Theme
# ──────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT     = "#1f6aa5"
BG_DARK    = "#1a1a2e"
BG_MID     = "#16213e"
BG_PANEL   = "#0f3460"
TEXT_MAIN  = "#e0e0e0"
TEXT_DIM   = "#888888"
SUCCESS    = "#4caf50"
WARNING    = "#ff9800"

FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_LABEL = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 10)
FONT_MONO  = ("Courier New", 10)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def parse_kernel_size(s: str) -> int:
    """Convert '3x3' → 3."""
    return int(s.split("x")[0])


def parse_block_size(s: str) -> int:
    """Convert '8x8' → 8."""
    return int(s.split("x")[0])


def array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a numpy uint8 array to a PIL Image."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode='RGB')
    elif arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr[:, :, :3], mode='RGB')
    return Image.fromarray(arr)


# ──────────────────────────────────────────────────────────────
# Edge Results Popup
# ──────────────────────────────────────────────────────────────

class EdgeResultsWindow(ctk.CTkToplevel):
    """Show horizontal, vertical, and magnitude edge maps side-by-side."""

    def __init__(self, parent, gx, gy, mag, detector_name):
        super().__init__(parent)
        self.title(f"{detector_name} Edge Detection Results")
        self.geometry("900x380")
        self.resizable(False, False)

        ctk.CTkLabel(self, text=f"{detector_name} Edge Detection",
                     font=FONT_TITLE).pack(pady=8)

        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        thumb_size = (260, 260)
        images = [("Horizontal (Gx)", gx),
                  ("Vertical (Gy)", gy),
                  ("Magnitude (Combined)", mag)]

        self._photos = []
        for label_text, arr in images:
            col = ctk.CTkFrame(frame)
            col.pack(side="left", expand=True, fill="both", padx=5, pady=5)

            ctk.CTkLabel(col, text=label_text, font=FONT_SMALL).pack(pady=3)

            pil = array_to_pil(arr).resize(thumb_size, Image.NEAREST)
            photo = ImageTk.PhotoImage(pil)
            self._photos.append(photo)

            lbl = tk.Label(col, image=photo, bg=BG_MID)
            lbl.pack()


# ──────────────────────────────────────────────────────────────
# Main Application Window
# ──────────────────────────────────────────────────────────────

class MainWindow(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("🏥 Clinical Image Analysis & Enhancement Workbench")
        self.geometry("1380x820")
        self.minsize(1000, 650)

        # State
        self.pipeline    = Pipeline()
        self.metadata    = {}
        self._photo_ref  = None          # Keep PhotoImage alive
        self._edge_cache = None          # (gx, gy, mag, detector_name)
        self._zoom_level = 1.0           # Accumulated zoom scale
        self._pipeline_panel = None      # Will be created in _build_layout

        self._build_layout()
        self._set_status("Welcome! Open a DICOM, JPEG, or BMP image to begin.", "info")

    # ──────────────────────────────────────────────────────
    # Layout construction
    # ──────────────────────────────────────────────────────

    def _build_layout(self):
        """Construct the three-column layout."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._build_top_bar()
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()

    # ── Top bar ──────────────────────────────────────────

    def _build_top_bar(self):
        bar = ctk.CTkFrame(self, height=52, corner_radius=0)
        bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        bar.grid_propagate(False)

        # Logo / title
        ctk.CTkLabel(bar, text="🏥  Clinical Image Workbench",
                     font=("Segoe UI", 14, "bold")).pack(side="left", padx=15)

        # Action buttons
        btn_cfg = dict(width=110, height=34, corner_radius=6)
        ctk.CTkButton(bar, text="📂  Open",  command=self._open_image,
                      fg_color=ACCENT,      **btn_cfg).pack(side="left", padx=4, pady=8)
        ctk.CTkButton(bar, text="💾  Save",  command=self._save_image,
                      fg_color="#2d6a4f",   **btn_cfg).pack(side="left", padx=4, pady=8)

        # Status label (right-aligned)
        self._status_var = tk.StringVar(value="")
        self._status_lbl = ctk.CTkLabel(bar, textvariable=self._status_var,
                                        font=FONT_SMALL, text_color=TEXT_DIM)
        self._status_lbl.pack(side="right", padx=15)

    # ── Left tools panel ─────────────────────────────────

    def _build_left_panel(self):
        panel = ctk.CTkScrollableFrame(self, width=250, corner_radius=0)
        panel.grid(row=1, column=0, sticky="nsew")

        # ── Zoom ──────────────────────────────────────────
        self._section_title(panel, "🔍  ZOOM")

        ctk.CTkLabel(panel, text="Interpolation Method:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._interp_var = tk.StringVar(value="Nearest Neighbor")
        ctk.CTkOptionMenu(panel, variable=self._interp_var,
                          values=["Nearest Neighbor", "Bilinear"],
                          width=226).pack(padx=12, pady=3)

        zoom_row = ctk.CTkFrame(panel, fg_color="transparent")
        zoom_row.pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(zoom_row, text="＋  Zoom In",  width=107,
                      command=self._zoom_in).pack(side="left", padx=2)
        ctk.CTkButton(zoom_row, text="－  Zoom Out", width=107,
                      command=self._zoom_out).pack(side="right", padx=2)

        self._zoom_lbl = ctk.CTkLabel(panel, text="Scale: 1.00×",
                                      font=FONT_SMALL, text_color=TEXT_DIM)
        self._zoom_lbl.pack(pady=2)

        self._divider(panel)

        # ── Smoothing Filters ─────────────────────────────
        self._section_title(panel, "🎛  SMOOTHING FILTERS")

        ctk.CTkLabel(panel, text="Filter Type:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._filter_var = tk.StringVar(value="Average")
        ctk.CTkOptionMenu(panel, variable=self._filter_var,
                          values=["Average", "Gaussian", "Median"],
                          command=self._on_filter_change,
                          width=226).pack(padx=12, pady=3)

        ctk.CTkLabel(panel, text="Kernel Size:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._kernel_var = tk.StringVar(value="3x3")
        ctk.CTkOptionMenu(panel, variable=self._kernel_var,
                          values=["3x3", "5x5", "7x7", "9x9"],
                          width=226).pack(padx=12, pady=3)

        # Gaussian-only sigma entry (hidden by default)
        self._sigma_frame = ctk.CTkFrame(panel, fg_color="transparent")
        ctk.CTkLabel(self._sigma_frame, text="Sigma (σ):",
                     font=FONT_SMALL, text_color=TEXT_DIM).pack(anchor="w")
        self._sigma_entry = ctk.CTkEntry(self._sigma_frame,
                                         placeholder_text="e.g. 1.5", width=226)
        self._sigma_entry.pack()

        ctk.CTkButton(panel, text="▶  Apply Smoothing",
                      command=self._apply_smoothing).pack(padx=12, pady=6, fill="x")

        self._divider(panel)

        # ── Edge Detection ────────────────────────────────
        self._section_title(panel, "📐  EDGE DETECTION")

        ctk.CTkLabel(panel, text="Detector:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._edge_var = tk.StringVar(value="Sobel")
        ctk.CTkOptionMenu(panel, variable=self._edge_var,
                          values=["Sobel", "Prewitt"],
                          width=226).pack(padx=12, pady=3)

        ctk.CTkLabel(panel, text="Apply to Pipeline:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._edge_apply_var = tk.StringVar(value="Magnitude (Combined)")
        ctk.CTkOptionMenu(panel, variable=self._edge_apply_var,
                          values=["Horizontal (Gx)",
                                  "Vertical (Gy)",
                                  "Magnitude (Combined)"],
                          width=226).pack(padx=12, pady=3)

        edge_row = ctk.CTkFrame(panel, fg_color="transparent")
        edge_row.pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(edge_row, text="▶  Apply", width=107,
                      command=self._apply_edge).pack(side="left", padx=2)
        ctk.CTkButton(edge_row, text="👁  All 3", width=107,
                      command=self._show_all_edges).pack(side="right", padx=2)

        self._divider(panel)

        # ── Local Histogram Equalization ──────────────────
        self._section_title(panel, "📊  LOCAL HISTOGRAM EQ.")

        ctk.CTkLabel(panel, text="Block Size:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._block_var = tk.StringVar(value="8x8")
        ctk.CTkOptionMenu(panel, variable=self._block_var,
                          values=["4x4", "8x8", "16x16", "32x32"],
                          width=226).pack(padx=12, pady=3)

        ctk.CTkButton(panel, text="▶  Apply Local HE",
                      command=self._apply_local_he).pack(padx=12, pady=6, fill="x")

    # ── Center image canvas ───────────────────────────────

    def _build_center_panel(self):
        container = ctk.CTkFrame(self, corner_radius=0)
        container.grid(row=1, column=1, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(container, bg=BG_DARK,
                                 highlightthickness=0, cursor="crosshair")
        self._canvas.grid(row=0, column=0, sticky="nsew")

        # Scrollbars
        vsb = tk.Scrollbar(container, orient="vertical",
                           command=self._canvas.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb = tk.Scrollbar(container, orient="horizontal",
                           command=self._canvas.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self._canvas.configure(yscrollcommand=vsb.set,
                               xscrollcommand=hsb.set)

        # Placeholder text
        self._canvas.create_text(
            500, 300,
            text="📂  Open an image to begin\n\nSupported formats: DICOM · JPEG · BMP",
            fill=TEXT_DIM, font=("Segoe UI", 14), justify="center",
            tags="placeholder"
        )

        self._canvas.bind("<Configure>", self._on_canvas_resize)

    # ── Right info panel ──────────────────────────────────

    def _build_right_panel(self):
        panel = ctk.CTkScrollableFrame(self, width=240, corner_radius=0)
        panel.grid(row=1, column=2, sticky="nsew")

        self._section_title(panel, "📋  IMAGE METADATA")
        self._meta_box = ctk.CTkTextbox(panel, height=220, font=FONT_MONO,
                                        state="disabled")
        self._meta_box.pack(fill="x", padx=8, pady=4)
        self._update_metadata_display()

        self._divider(panel)

        # Initialize pipeline panel with undo/reset callbacks
        self._pipeline_panel = PipelinePanel(panel, 
                                             on_undo=self._undo,
                                             on_reset=self._reset)

    # ──────────────────────────────────────────────────────
    # UI helpers
    # ──────────────────────────────────────────────────────

    def _section_title(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=FONT_TITLE).pack(
            anchor="w", padx=12, pady=(12, 2))

    def _divider(self, parent):
        ctk.CTkFrame(parent, height=2, fg_color=BG_PANEL).pack(
            fill="x", padx=8, pady=8)

    def _set_status(self, msg, kind="info"):
        colors = {"info": TEXT_DIM, "ok": SUCCESS, "warn": WARNING, "error": "#e53935"}
        self._status_var.set(msg)
        self._status_lbl.configure(text_color=colors.get(kind, TEXT_DIM))

    # ──────────────────────────────────────────────────────
    # Image display
    # ──────────────────────────────────────────────────────

    def _display_image(self, array: np.ndarray):
        """Render a numpy array on the canvas, fitting it to the visible area."""
        if array is None:
            return

        pil_img = array_to_pil(array)
        img_w, img_h = pil_img.size

        canvas_w = self._canvas.winfo_width()
        canvas_h = self._canvas.winfo_height()
        if canvas_w < 10:
            canvas_w, canvas_h = 700, 560

        # Fit the image inside the canvas (never upscale beyond native for display)
        scale = min(canvas_w / img_w, canvas_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)

        # Resize for display only (PIL LANCZOS — just for screen rendering,
        # not for actual image data processing)
        display_pil = pil_img.resize((disp_w, disp_h), Image.NEAREST)

        self._photo_ref = ImageTk.PhotoImage(display_pil)
        self._canvas.delete("all")
        cx = max(canvas_w, disp_w) // 2
        cy = max(canvas_h, disp_h) // 2
        self._canvas.create_image(cx, cy, anchor="center",
                                  image=self._photo_ref, tags="img")
        self._canvas.configure(scrollregion=(0, 0,
                                             max(canvas_w, disp_w),
                                             max(canvas_h, disp_h)))

    def _on_canvas_resize(self, _event=None):
        if not self.pipeline.is_empty:
            self._display_image(self.pipeline.current_image)

    # ──────────────────────────────────────────────────────
    # Open / Save
    # ──────────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Open Medical Image",
            filetypes=[
                ("All supported", "*.dcm *.dicom *.jpg *.jpeg *.bmp"),
                ("DICOM", "*.dcm *.dicom"),
                ("JPEG",  "*.jpg *.jpeg"),
                ("BMP",   "*.bmp"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return

        try:
            image_array, self.metadata = load_image(path)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return

        self._zoom_level = 1.0
        self._zoom_lbl.configure(text="Scale: 1.00×")
        self._edge_cache = None

        self.pipeline.load(image_array)
        self._display_image(image_array)
        self._update_metadata_display()
        self._update_pipeline_display()

        fname = os.path.basename(path)
        h, w = image_array.shape[:2]
        self._set_status(f"Loaded: {fname}  ({w}×{h})", "ok")

    def _save_image(self):
        if self.pipeline.is_empty:
            messagebox.showwarning("No Image", "Load an image first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"),
                       ("BMP", "*.bmp"), ("TIFF", "*.tif")]
        )
        if not path:
            return

        try:
            save_image(self.pipeline.current_image, path)
            self._set_status(f"Saved: {os.path.basename(path)}", "ok")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    # ──────────────────────────────────────────────────────
    # Pipeline controls
    # ──────────────────────────────────────────────────────

    def _undo(self):
        if self.pipeline.is_empty:
            return
        img = self.pipeline.undo()
        self._display_image(img)
        self._update_pipeline_display()
        self._set_status("Undo — reverted to previous step.", "warn")

    def _reset(self):
        if self.pipeline.is_empty:
            return
        img = self.pipeline.reset()
        self._zoom_level = 1.0
        self._zoom_lbl.configure(text="Scale: 1.00×")
        self._edge_cache = None
        self._display_image(img)
        self._update_pipeline_display()
        self._set_status("Reset to original image.", "warn")

    # ──────────────────────────────────────────────────────
    # Zoom
    # ──────────────────────────────────────────────────────

    def _do_zoom(self, factor: float):
        if self.pipeline.is_empty:
            messagebox.showwarning("No Image", "Load an image first.")
            return

        new_scale = self._zoom_level * factor
        if new_scale < 0.1 or new_scale > 8.0:
            self._set_status("Zoom limit reached (0.1× – 8.0×).", "warn")
            return

        current = self.pipeline.current_image
        interp  = self._interp_var.get()
        step    = "Zoom In" if factor > 1 else "Zoom Out"
        label   = f"{step} ×{factor:.2f} ({interp}) → Scale {new_scale:.2f}×"

        try:
            if interp == "Nearest Neighbor":
                result = nearest_neighbor_zoom(current, factor)
            else:
                result = bilinear_zoom(current, factor)
        except Exception as exc:
            messagebox.showerror("Zoom Error", str(exc))
            return

        self._zoom_level = new_scale
        self._zoom_lbl.configure(text=f"Scale: {new_scale:.2f}×")
        self.pipeline.push(result, label)
        self._display_image(result)
        self._update_pipeline_display()
        self._set_status(label, "ok")

    def _zoom_in(self):
        self._do_zoom(1.25)

    def _zoom_out(self):
        self._do_zoom(0.8)

    # ──────────────────────────────────────────────────────
    # Smoothing filters
    # ──────────────────────────────────────────────────────

    def _on_filter_change(self, choice):
        if choice == "Gaussian":
            self._sigma_frame.pack(padx=12, pady=2, fill="x")
        else:
            self._sigma_frame.pack_forget()

    def _apply_smoothing(self):
        if self.pipeline.is_empty:
            messagebox.showwarning("No Image", "Load an image first.")
            return

        ftype  = self._filter_var.get()
        ksize  = parse_kernel_size(self._kernel_var.get())

        try:
            if ftype == "Average":
                desc = f"Average Filter {ksize}×{ksize}"
                result = self.pipeline.apply(
                    lambda img: average_filter(img, ksize), desc)

            elif ftype == "Gaussian":
                sigma_txt = self._sigma_entry.get().strip() or "1.5"
                sigma = float(sigma_txt)
                if sigma <= 0:
                    raise ValueError("Sigma must be positive.")
                desc = f"Gaussian Filter {ksize}×{ksize}, σ={sigma}"
                result = self.pipeline.apply(
                    lambda img, k=ksize, s=sigma: gaussian_filter(img, k, s), desc)

            elif ftype == "Median":
                desc = f"Median Filter {ksize}×{ksize}"
                result = self.pipeline.apply(
                    lambda img: median_filter(img, ksize), desc)

            else:
                return

        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Filter Error", str(exc))
            return

        self._display_image(result)
        self._update_pipeline_display()
        self._set_status(f"Applied: {desc}", "ok")

    # ──────────────────────────────────────────────────────
    # Edge detection
    # ──────────────────────────────────────────────────────

    def _run_edge_detection(self):
        """Run edge detection and cache the results."""
        if self.pipeline.is_empty:
            messagebox.showwarning("No Image", "Load an image first.")
            return None

        detector = self._edge_var.get()
        current  = self.pipeline.current_image

        try:
            if detector == "Sobel":
                gx, gy, mag = sobel_filter(current)
            else:
                gx, gy, mag = prewitt_filter(current)
        except Exception as exc:
            messagebox.showerror("Edge Detection Error", str(exc))
            return None

        self._edge_cache = (gx, gy, mag, detector)
        return gx, gy, mag, detector

    def _apply_edge(self):
        """Apply one of the three edge maps into the pipeline."""
        result = self._run_edge_detection()
        if result is None:
            return
        gx, gy, mag, detector = result

        choice = self._edge_apply_var.get()
        if "Horizontal" in choice:
            img  = gx
            kind = "Gx"
        elif "Vertical" in choice:
            img  = gy
            kind = "Gy"
        else:
            img  = mag
            kind = "Magnitude"

        desc = f"{detector} Edge — {kind}"
        self.pipeline.push(img, desc)
        self._display_image(img)
        self._update_pipeline_display()
        self._set_status(f"Applied: {desc}", "ok")

    def _show_all_edges(self):
        """Open a popup showing all three edge maps."""
        result = self._run_edge_detection()
        if result is None:
            return
        gx, gy, mag, detector = result
        EdgeResultsWindow(self, gx, gy, mag, detector)

    # ──────────────────────────────────────────────────────
    # Local Histogram Equalization
    # ──────────────────────────────────────────────────────

    def _apply_local_he(self):
        if self.pipeline.is_empty:
            messagebox.showwarning("No Image", "Load an image first.")
            return

        block_size = parse_block_size(self._block_var.get())
        desc = f"Local Hist. Eq. — block {block_size}×{block_size}"

        try:
            result = self.pipeline.apply(
                lambda img, b=block_size: local_histogram_equalization(img, b),
                desc
            )
        except Exception as exc:
            messagebox.showerror("Histogram Error", str(exc))
            return

        self._display_image(result)
        self._update_pipeline_display()
        self._set_status(f"Applied: {desc}", "ok")

    # ──────────────────────────────────────────────────────
    # Info panel updates
    # ──────────────────────────────────────────────────────

    def _update_metadata_display(self):
        self._meta_box.configure(state="normal")
        self._meta_box.delete("1.0", "end")

        if not self.metadata:
            self._meta_box.insert("end", "No image loaded.")
        else:
            fields = [
                ("Format",       self.metadata.get("Format",       "N/A")),
                ("Width",        self.metadata.get("Width",        "N/A") + " px"),
                ("Height",       self.metadata.get("Height",       "N/A") + " px"),
                ("Bit Depth",    self.metadata.get("Bit Depth",    "N/A")),
                ("─────────────", ""),
                ("Modality",     self.metadata.get("Modality",     "N/A")),
                ("Patient Name", self.metadata.get("Patient Name", "N/A")),
                ("Patient Age",  self.metadata.get("Patient Age",  "N/A")),
                ("Body Part",    self.metadata.get("Body Part",    "N/A")),
                ("Study Date",   self.metadata.get("Study Date",   "N/A")),
                ("Institution",  self.metadata.get("Institution",  "N/A")),
            ]
            for key, val in fields:
                if key.startswith("─"):
                    self._meta_box.insert("end", f"\n{key}\n")
                else:
                    self._meta_box.insert("end", f"{key}:\n  {val}\n")

        self._meta_box.configure(state="disabled")

    def _update_pipeline_display(self):
        """Update the pipeline display via the pipeline panel."""
        if self._pipeline_panel:
            self._pipeline_panel.update_display(self.pipeline)
