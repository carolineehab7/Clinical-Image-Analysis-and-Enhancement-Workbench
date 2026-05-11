import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from core.io import load_image, save_image
from core.pipeline import Pipeline
from core.enhancement import (
    compute_histogram,
    local_histogram_equalization_interpolated,
)
from gui.panels.filter_panel import FilterPanel
from gui.windows.histogram_window import HistogramWindow
from gui.panels.noise_panel import NoisePanel
from gui.panels.pipeline_panel import PipelinePanel
from gui.theme import (
    ACCENT_BLUE,
    ACCENT_CYAN,
    ACCENT_TEAL,
    APP_MODE,
    BG_CANVAS,
    BG_DIVIDER,
    BG_ELEVATED,
    BG_CARD,
    BG_SIDEBAR,
    BG_SURFACE,
    BG_TEXTBOX,
    BG_WINDOW,
    BORDER_CYAN,
    COLOR_THEME,
    ERROR,
    FONT_MONO,
    FONT_SMALL,
    FONT_TITLE,
    SUCCESS,
    TEXT_DIM,
    TEXT_MAIN,
    WARNING,
)
from .panels.zoom_panel import zoom_in, zoom_out
from .panels.EdgeDetection_panel import EdgeDetectionPanel
from .panels.fft_panel import FFTPanel
from .panels.roi_panel import ROIPanel
from bonus.morphology_panel import MorphologyPanel


# ──────────────────────────────────────────────────────────────
# Theme
# ──────────────────────────────────────────────────────────────
ctk.set_appearance_mode(APP_MODE)
ctk.set_default_color_theme(COLOR_THEME)

ACCENT = ACCENT_CYAN
BG_DARK = BG_WINDOW
BG_MID = BG_SURFACE
BG_PANEL = BG_DIVIDER


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

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

class CloseImageDialog(ctk.CTkToplevel):
    """Confirmation dialog shown before closing the current image."""

    def __init__(self, parent, on_confirm):
        super().__init__(parent)
        self.configure(fg_color=BG_SURFACE)
        self.title("Close Image")
        self.geometry("420x180")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._on_confirm = on_confirm

        ctk.CTkLabel(
            self,
            text="Are you sure you want to close. Your changes will not be saved",
            font=FONT_TITLE,
            text_color=TEXT_MAIN,
            wraplength=360,
            justify="center",
        ).pack(padx=20, pady=(24, 18))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=8)

        ctk.CTkButton(btn_row, text="Close", width=120,
                      fg_color=ACCENT_BLUE, command=self._dismiss).pack(side="left", padx=8)
        ctk.CTkButton(btn_row, text="Confirm", width=120,
                      fg_color=ERROR, command=self._confirm).pack(side="left", padx=8)

        self.protocol("WM_DELETE_WINDOW", self._dismiss)

    def _dismiss(self):
        self.grab_release()
        self.destroy()

    def _confirm(self):
        self.grab_release()
        self.destroy()
        self._on_confirm()


# ──────────────────────────────────────────────────────────────
# Main Application Window
# ──────────────────────────────────────────────────────────────

class MainWindow(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.configure(fg_color=BG_WINDOW)
        self.title(" Clinical Image Analysis & Enhancement Workbench")
        self.geometry("1380x820")
        self.minsize(1000, 650)

        # State
        self.pipeline    = Pipeline()
        self.metadata    = {}
        self._photo_ref  = None          # Keep PhotoImage alive
        self._edge_cache = None          # (gx, gy, mag, detector_name)
        self._zoom_level = 1.0           # Accumulated zoom scale
        self._pipeline_panel = None      # Will be created in _build_layout

        # ROI state (Member 3)
        self._roi_mode         = False
        self._roi_start        = None    # canvas coords of mouse-press
        self._roi_rect_canvas  = None    # canvas rectangle item ID
        self._roi_image_coords = None    # (x0, y0, x1, y1) in image pixels
        # Display geometry — updated by _display_image for ROI coordinate mapping
        self._canvas_img_x0    = 0.0
        self._canvas_img_y0    = 0.0
        self._display_scale    = 1.0
        self._roi_panel        = None    # set in _build_left_panel

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
        bar = ctk.CTkFrame(
            self,
            height=52,
            corner_radius=0,
            fg_color=BG_SIDEBAR,
            border_width=1,
            border_color=BORDER_CYAN,
        )
        bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        bar.grid_propagate(False)

        # Logo / title
        ctk.CTkLabel(bar, text="  Clinical Image Workbench",
                     font=("Segoe UI", 14, "bold"), text_color=TEXT_MAIN).pack(side="left", padx=15)

        # Action buttons
        btn_cfg = dict(width=110, height=34, corner_radius=6)
        ctk.CTkButton(bar, text="  Open",  command=self._open_image,
                      fg_color=ACCENT_CYAN, hover_color="#00869B", **btn_cfg).pack(side="left", padx=4, pady=8)
        ctk.CTkButton(bar, text="  Save",  command=self._save_image,
                      fg_color=ACCENT_TEAL, hover_color="#0C6541", **btn_cfg).pack(side="left", padx=4, pady=8)
        ctk.CTkButton(bar, text="✖  Close", command=self._ask_close_image,
                  fg_color=ERROR, hover_color="#9E3E49", **btn_cfg).pack(side="left", padx=4, pady=8)

        # Status label (right-aligned)
        self._status_var = tk.StringVar(value="")
        self._status_lbl = ctk.CTkLabel(bar, textvariable=self._status_var,
                                        font=FONT_SMALL, text_color=TEXT_DIM)
        self._status_lbl.pack(side="right", padx=15)

    # ── Left tools panel ─────────────────────────────────

    def _build_left_panel(self):
        panel = ctk.CTkScrollableFrame(self, width=250, corner_radius=0, fg_color=BG_SIDEBAR)
        panel.grid(row=1, column=0, sticky="nsew")

        ctk.CTkLabel(panel, text="SECTIONS", font=FONT_SMALL, text_color=TEXT_DIM).pack(
            anchor="w", padx=12, pady=(10, 2)
        )
        self._left_nav_var = tk.StringVar(value="Spatial")
        ctk.CTkSegmentedButton(
            panel,
            values=["Spatial", "Frequency", "Morphology"],
            variable=self._left_nav_var,
            command=self._on_left_nav_change,
        ).pack(fill="x", padx=12, pady=(0, 10))

        self._page_spatial = ctk.CTkFrame(panel, fg_color="transparent")
        self._page_frequency = ctk.CTkFrame(panel, fg_color="transparent")
        self._page_morphology = ctk.CTkFrame(panel, fg_color="transparent")

        ###### Zoom ######
        zoom_card = self._section_card(self._page_spatial)
        self._section_title(zoom_card, "ZOOM")

        ctk.CTkLabel(zoom_card, text="Interpolation Method:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._interp_var = tk.StringVar(value="Nearest Neighbor")
        ctk.CTkOptionMenu(zoom_card, variable=self._interp_var,
                          values=["Nearest Neighbor", "Bilinear"],
                          width=226).pack(padx=12, pady=3)

        zoom_row = ctk.CTkFrame(zoom_card, fg_color="transparent")
        zoom_row.pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(zoom_row, text="＋  Zoom In",  width=107,
                      command=lambda: zoom_in(self)).pack(side="left", padx=2)
        ctk.CTkButton(zoom_row, text="－  Zoom Out", width=107,
                      command=lambda: zoom_out(self)).pack(side="right", padx=2)

        self._zoom_lbl = ctk.CTkLabel(zoom_card, text="Scale: 1.00×",
                                      font=FONT_SMALL, text_color=TEXT_DIM)
        self._zoom_lbl.pack(pady=2)

        # Group: Spatial filters
        self._group_title(self._page_spatial, "SPATIAL FILTERS")

        # ── Smoothing Filters ────────────
        self._filter_panel = FilterPanel(
            self._page_spatial,
            pipeline=self.pipeline,
            on_image_updated=self._display_image,
            on_pipeline_updated=self._update_pipeline_display,
            on_status=self._set_status,
        )

         ###### Edge Detection ######
        self._edge_panel = EdgeDetectionPanel(
            self._page_spatial,
            pipeline=self.pipeline,
            on_image_updated=self._display_image,
            on_pipeline_updated=self._update_pipeline_display,
            on_status=self._set_status,
        )

        # ── Noise Generation ─────────────────────────────
        self._noise_panel = NoisePanel(
            self._page_spatial,
            pipeline=self.pipeline,
            on_image_updated=self._display_image,
            on_pipeline_updated=self._update_pipeline_display,
            on_status=self._set_status,
        )

        # ── Local Histogram Equalization ──────────────────
        he_card = self._section_card(self._page_spatial)
        self._section_title(he_card, "LOCAL HISTOGRAM EQ.")

        ctk.CTkLabel(he_card, text="Block Size:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._block_var = tk.StringVar(value="8x8")
        ctk.CTkOptionMenu(he_card, variable=self._block_var,
                          values=["4x4", "8x8", "16x16", "32x32"],
                          width=226).pack(padx=12, pady=3)

        ctk.CTkButton(he_card, text="▶  Apply Local HE",
                      command=self._apply_local_he).pack(padx=12, pady=6, fill="x")

        ctk.CTkButton(he_card, text="Before/After HE",
                  command=self._show_histogram_comparison).pack(padx=12, pady=(2, 6), fill="x")

        # Group: Frequency filters
        self._group_title(self._page_frequency, "FREQUENCY FILTERS")

        # ── FFT & Notch Filter (Members 1 & 2) ───────────────
        fft_card = self._section_card(self._page_frequency)
        FFTPanel(
            fft_card,
            pipeline=self.pipeline,
            on_image_updated=self._display_image,
            on_pipeline_updated=self._update_pipeline_display,
            on_status=self._set_status,
        )

        # ── ROI Analysis (Members 3 & 4) ──────────────────────
        roi_card = self._section_card(self._page_frequency)
        self._roi_panel = ROIPanel(
            roi_card,
            on_toggle_roi_mode=self._set_roi_mode,
            on_clear_roi=self._clear_roi,
            on_analyze_roi=self._analyze_roi,
        )

        # ── Morphological Engine (Bonus) ───────────────────────
        self._group_title(self._page_morphology, "MORPHOLOGY")
        morph_card = self._section_card(self._page_morphology)
        MorphologyPanel(
            morph_card,
            pipeline=self.pipeline,
            on_image_updated=self._display_image,
            on_pipeline_updated=self._update_pipeline_display,
            on_status=self._set_status,
        )

        self._show_left_page("Spatial")

    # ── Center image canvas ───────────────────────────────

    def _build_center_panel(self):
        container = ctk.CTkFrame(self, corner_radius=0, fg_color=BG_WINDOW)
        container.grid(row=1, column=1, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(container, bg=BG_CANVAS,
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
            text="Open an image to begin\n\nSupported formats: DICOM · JPEG · BMP",
            fill=TEXT_DIM, font=("Segoe UI", 14), justify="center",
            tags="placeholder"
        )

        self._canvas.bind("<Configure>",      self._on_canvas_resize)
        self._canvas.bind("<ButtonPress-1>",   self._on_roi_press)
        self._canvas.bind("<B1-Motion>",       self._on_roi_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_roi_release)

    # ── Right info panel ──────────────────────────────────

    def _build_right_panel(self):
        panel = ctk.CTkScrollableFrame(self, width=240, corner_radius=0, fg_color=BG_SIDEBAR)
        panel.grid(row=1, column=2, sticky="nsew")

        meta_card = self._section_card(panel)
        self._section_title(meta_card, "IMAGE METADATA")
        self._meta_box = ctk.CTkTextbox(
            meta_card,
            height=220,
            font=FONT_MONO,
            state="disabled",
            fg_color=BG_TEXTBOX,
            text_color=TEXT_MAIN,
            border_width=1,
            border_color=BORDER_CYAN,
        )
        self._meta_box.pack(fill="x", padx=8, pady=4)
        self._update_metadata_display()

        # Initialize pipeline panel with undo/redo/reset/save-log callbacks
        self._pipeline_panel = PipelinePanel(
            panel,
            on_undo=self._undo,
            on_redo=self._redo,
            on_reset=self._reset,
            on_save_log=self._save_pipeline_log,
        )

    # ──────────────────────────────────────────────────────
    # UI helpers
    # ──────────────────────────────────────────────────────

    def _section_title(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=FONT_TITLE, text_color=TEXT_MAIN).pack(
            anchor="w", padx=12, pady=(12, 2))

    def _group_title(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=FONT_SMALL, text_color=TEXT_DIM).pack(
            anchor="w", padx=12, pady=(6, 2))

    def _section_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=BG_CARD,
            corner_radius=10,
            border_width=1,
            border_color=BORDER_CYAN,
        )
        card.pack(fill="x", padx=10, pady=(0, 8))
        return card

    def _on_left_nav_change(self, value):
        self._show_left_page(value)

    def _show_left_page(self, name):
        pages = {
            "Spatial": self._page_spatial,
            "Frequency": self._page_frequency,
            "Morphology": self._page_morphology,
        }
        for page in pages.values():
            page.pack_forget()

        page = pages.get(name)
        if page is not None:
            page.pack(fill="x", padx=0, pady=0)

    def _divider(self, parent):
        ctk.CTkFrame(parent, height=2, fg_color=BORDER_CYAN).pack(
            fill="x", padx=8, pady=8)

    def _set_status(self, msg, kind="info"):
        colors = {"info": TEXT_DIM, "ok": SUCCESS, "warn": WARNING, "error": ERROR}
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

        # Store display geometry for ROI coordinate mapping
        self._canvas_img_x0 = cx - disp_w / 2
        self._canvas_img_y0 = cy - disp_h / 2
        self._display_scale  = scale

        # Redraw ROI overlay on top of the new image
        if self._roi_image_coords is not None:
            self._draw_roi_rectangle()

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

    def _ask_close_image(self):
        if self.pipeline.is_empty:
            self._set_status("No image is currently open.", "warn")
            return

        CloseImageDialog(self, self._close_image)

    def _close_image(self):
        self.pipeline.clear()
        self.metadata = {}
        self._photo_ref = None
        self._edge_cache = None
        self._zoom_level = 1.0
        self._zoom_lbl.configure(text="Scale: 1.00×")
        self._roi_mode         = False
        self._roi_image_coords = None
        self._roi_rect_canvas  = None
        if self._roi_panel:
            self._roi_panel.set_active(False)
            self._roi_panel.set_status("No ROI selected.")

        self._canvas.delete("all")
        self._canvas.create_text(
            500, 300,
            text="  Open an image to begin\n\nSupported formats: DICOM · JPEG · BMP",
            fill=TEXT_DIM, font=("Segoe UI", 14), justify="center",
            tags="placeholder"
        )

        self._update_metadata_display()
        self._update_pipeline_display()
        self._set_status("Image closed. Open another file to continue.", "warn")

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

    def _redo(self):
        if self.pipeline.is_empty or not self.pipeline.can_redo:
            return
        img = self.pipeline.redo()
        self._display_image(img)
        self._update_pipeline_display()
        self._set_status("Redo — re-applied step.", "ok")

    def _save_pipeline_log(self):
        if self.pipeline.is_empty:
            self._set_status("No pipeline to save.", "warn")
            return
        path = filedialog.asksaveasfilename(
            title="Save Pipeline Log",
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        from datetime import datetime
        lines = [
            "Clinical Image Workbench — Pipeline Log",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "─" * 42,
        ]
        for i, step in enumerate(self.pipeline.steps):
            marker = "▶" if i == self.pipeline.step_count - 1 else " "
            lines.append(f"{marker} Step {i:>2}: {step}")
        lines += [
            "─" * 42,
            f"Total steps: {self.pipeline.step_count}",
        ]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self._set_status(f"Pipeline log saved: {os.path.basename(path)}", "ok")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def _reset(self):
        if self.pipeline.is_empty:
            return
        img = self.pipeline.reset()
        self._zoom_level = 1.0
        self._zoom_lbl.configure(text="Scale: 1.00×")
        self._edge_cache = None
        self._roi_image_coords = None
        self._roi_rect_canvas  = None
        if self._roi_panel:
            self._roi_panel.set_status("No ROI selected.")
        self._display_image(img)
        self._update_pipeline_display()
        self._set_status("Reset to original image.", "warn")

    # ──────────────────────────────────────────────────────
    # Local Histogram Equalization
    # ──────────────────────────────────────────────────────

    def _apply_local_he(self):
        if self.pipeline.is_empty:
            messagebox.showwarning("No Image", "Load an image first.")
            return

        block_size = parse_block_size(self._block_var.get())
        desc = f"Local Hist. Eq. (Interpolated) — block {block_size}×{block_size}"

        try:
            result = self.pipeline.apply(
                lambda img, b=block_size: local_histogram_equalization_interpolated(img, b),
                desc
            )
        except Exception as exc:
            messagebox.showerror("Histogram Error", str(exc))
            return

        self._display_image(result)
        self._update_pipeline_display()
        self._set_status(f"Applied: {desc}", "ok")

    def _show_histogram_comparison(self):
        """Show before and after local histogram equalization comparison."""
        if self.pipeline.is_empty:
            messagebox.showwarning("No Image", "Load an image first.")
            return

        # Get a fresh copy of the current image (don't modify pipeline)
        current_image = self.pipeline.current_image.copy()
        histogram_before = compute_histogram(current_image.flatten().astype("uint8"))

        # Apply local histogram equalization with selected block size
        block_size = parse_block_size(self._block_var.get())
        image_equalized = local_histogram_equalization(current_image.copy(), block_size)
        histogram_after = compute_histogram(image_equalized.flatten().astype("uint8"))
        
        # Show comparison window
        HistogramWindow(
            self, 
            histogram_before, 
            title="Histogram Comparison: Local Histogram Equalization",
            histogram_after=histogram_after,
            title_after=f"After Local HE ({block_size}x{block_size})"
        )

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

    # ──────────────────────────────────────────────────────
    # ROI drawing on main canvas (Member 3)
    # ──────────────────────────────────────────────────────

    def _set_roi_mode(self, active: bool):
        self._roi_mode = active

    def _on_roi_press(self, event):
        if not self._roi_mode or self.pipeline.is_empty:
            return
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        self._roi_start = (cx, cy)
        if self._roi_rect_canvas is not None:
            self._canvas.delete(self._roi_rect_canvas)
        self._roi_rect_canvas = self._canvas.create_rectangle(
            cx, cy, cx, cy, outline=ACCENT_CYAN, width=2, dash=(4, 4)
        )

    def _on_roi_drag(self, event):
        if not self._roi_mode or self._roi_start is None:
            return
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        self._canvas.coords(
            self._roi_rect_canvas,
            self._roi_start[0], self._roi_start[1], cx, cy,
        )

    def _on_roi_release(self, event):
        if not self._roi_mode or self._roi_start is None:
            return
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        coords = self._canvas_to_image(
            self._roi_start[0], self._roi_start[1], cx, cy
        )
        self._roi_start = None
        if coords is None:
            return
        self._roi_image_coords = coords
        x0, y0, x1, y1 = coords
        if self._roi_panel:
            self._roi_panel.set_status(f"ROI: ({x0},{y0}) → ({x1},{y1})")

    def _canvas_to_image(self, cx0, cy0, cx1, cy1):
        """Convert canvas pixel coords → image pixel coords, clamped to image bounds."""
        if self.pipeline.is_empty:
            return None
        s = self._display_scale if self._display_scale > 0 else 1.0
        h, w = self.pipeline.current_image.shape[:2]
        x0 = max(0, min(w, round((cx0 - self._canvas_img_x0) / s)))
        y0 = max(0, min(h, round((cy0 - self._canvas_img_y0) / s)))
        x1 = max(0, min(w, round((cx1 - self._canvas_img_x0) / s)))
        y1 = max(0, min(h, round((cy1 - self._canvas_img_y0) / s)))
        return x0, y0, x1, y1

    def _draw_roi_rectangle(self):
        """Redraw the ROI rectangle on the canvas (called from _display_image)."""
        if self._roi_image_coords is None:
            return
        x0, y0, x1, y1 = self._roi_image_coords
        s = self._display_scale
        cx0 = self._canvas_img_x0 + x0 * s
        cy0 = self._canvas_img_y0 + y0 * s
        cx1 = self._canvas_img_x0 + x1 * s
        cy1 = self._canvas_img_y0 + y1 * s
        self._roi_rect_canvas = self._canvas.create_rectangle(
            cx0, cy0, cx1, cy1, outline=ACCENT_CYAN, width=2, dash=(4, 4)
        )

    def _clear_roi(self):
        self._roi_image_coords = None
        if self._roi_rect_canvas is not None:
            self._canvas.delete(self._roi_rect_canvas)
            self._roi_rect_canvas = None
        if self._roi_panel:
            self._roi_panel.set_status("No ROI selected.")

    # ──────────────────────────────────────────────────────
    # ROI analysis → statistics popup (Members 3 & 4)
    # ──────────────────────────────────────────────────────

    def _analyze_roi(self):
        if self.pipeline.is_empty:
            self._set_status("Load an image first.", "warn")
            return
        if self._roi_image_coords is None:
            self._set_status("Draw an ROI on the image first.", "warn")
            return
        from core.analysis.roi import extract_roi
        from gui.windows.statistics_window import StatisticsWindow
        x0, y0, x1, y1 = self._roi_image_coords
        roi = extract_roi(self.pipeline.current_image, x0, y0, x1, y1)
        if roi is None or roi.size == 0:
            self._set_status("ROI is empty — draw a larger region.", "warn")
            return
        label = f"({x0},{y0})→({x1},{y1})"
        StatisticsWindow(self, roi, label)
        self._set_status(f"Analyzing ROI {label}", "ok")
