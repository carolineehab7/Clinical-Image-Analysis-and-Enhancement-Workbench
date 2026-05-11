import tkinter as tk
import customtkinter as ctk

from gui.theme import (
    ACCENT_CYAN, ACCENT_PURPLE, BG_SIDEBAR, BORDER_CYAN,
    ERROR, FONT_SMALL, FONT_TITLE, SUCCESS, TEXT_DIM, TEXT_MAIN,
)


class MorphologyPanel:

    def __init__(self, parent, pipeline, on_image_updated, on_pipeline_updated, on_status):
        """Build the Morphological Engine panel and hook up callbacks.

        Parameters
        ----------
        parent : widget
            Parent container for the panel.
        pipeline : Pipeline
            Shared pipeline holding the current image and history.
        on_image_updated : callable
            Callback to refresh the main image display.
        on_pipeline_updated : callable
            Callback to refresh the pipeline history UI.
        on_status : callable
            Status bar callback for messages.
        """
        self._pipeline            = pipeline
        self._on_image_updated    = on_image_updated
        self._on_pipeline_updated = on_pipeline_updated
        self._on_status           = on_status

        # ── Section header ──
        ctk.CTkLabel(parent, text="MORPHOLOGICAL ENGINE",
                     font=FONT_TITLE, text_color=ACCENT_CYAN).pack(
                         anchor="w", padx=12, pady=(12, 0))

        # ── Threshold ──
        ctk.CTkLabel(parent, text="Threshold value (0–255):",
                     font=FONT_SMALL, text_color=TEXT_DIM).pack(anchor="w", padx=12)

        thresh_row = ctk.CTkFrame(parent, fg_color="transparent")
        thresh_row.pack(fill="x", padx=12, pady=(2, 2))

        self._thresh_var = tk.IntVar(value=128)
        ctk.CTkSlider(
            thresh_row, from_=0, to=255,
            variable=self._thresh_var, number_of_steps=255,
            width=166,
        ).pack(side="left", padx=(0, 6))

        self._thresh_lbl = ctk.CTkLabel(thresh_row, text="128",
                                         font=FONT_SMALL, text_color=TEXT_MAIN, width=30)
        self._thresh_lbl.pack(side="left")
        self._thresh_var.trace_add("write", self._sync_thresh_label)

        ctk.CTkButton(parent, text="Binarize Image",
                      command=self._binarize,
                      fg_color=ACCENT_CYAN, hover_color="#007A8F").pack(
                          padx=12, pady=(2, 8), fill="x")

        # ── Structuring Element ──
        ctk.CTkLabel(parent, text="SE Size:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._se_size_var = tk.StringVar(value="3×3")
        ctk.CTkOptionMenu(parent, variable=self._se_size_var,
                          values=["3×3", "5×5", "7×7"], width=226).pack(
                              padx=12, pady=2)

        ctk.CTkLabel(parent, text="SE Shape:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12, pady=(4, 0))
        self._se_shape_var = tk.StringVar(value="Square")
        ctk.CTkOptionMenu(parent, variable=self._se_shape_var,
                          values=["Square", "Cross"], width=226).pack(
                              padx=12, pady=(2, 6))

        # ── Core operations ──
        row1 = ctk.CTkFrame(parent, fg_color="transparent")
        row1.pack(fill="x", padx=12, pady=2)
        ctk.CTkButton(row1, text="Erode",  width=107,
                      command=lambda: self._apply("Erode")).pack(side="left",  padx=2)
        ctk.CTkButton(row1, text="Dilate", width=107,
                      command=lambda: self._apply("Dilate")).pack(side="right", padx=2)

        # opening and closing
        row2 = ctk.CTkFrame(parent, fg_color="transparent")
        row2.pack(fill="x", padx=12, pady=2)
        ctk.CTkButton(row2, text="Opening", width=107,
                      fg_color=SUCCESS, hover_color="#0C6541",
                      command=lambda: self._apply("Opening")).pack(side="left",  padx=2)
        ctk.CTkButton(row2, text="Closing", width=107,
                      fg_color=SUCCESS, hover_color="#0C6541",
                      command=lambda: self._apply("Closing")).pack(side="right", padx=2)

        # boundary extraction
        ctk.CTkButton(parent, text="Boundary Extraction",
                      fg_color=ACCENT_PURPLE, hover_color="#7E22CE",
                      command=lambda: self._apply("Boundary")).pack(
                          padx=12, pady=(2, 6), fill="x")

        ctk.CTkFrame(parent, height=2, fg_color=BORDER_CYAN).pack(
            fill="x", padx=8, pady=8)

    # ── internals ───────────────────────────────────────────────────────

    def _sync_thresh_label(self, *_):
        """Keep the numeric label in sync with the slider value."""
        self._thresh_lbl.configure(text=str(self._thresh_var.get()))

    def _get_se(self):
        """Return the structuring element selected in the UI."""
        size = int(self._se_size_var.get().split("×")[0])
        shape = self._se_shape_var.get().lower()
        from bonus.morphology import make_structuring_element
        return make_structuring_element(size, shape)

    def _binarize(self):
        """Threshold the current image into a binary mask."""
        if self._pipeline.is_empty:
            self._on_status("Load an image first.", "warn")
            return
        from bonus.morphology import threshold
        try:
            T = self._thresh_var.get()
            result = threshold(self._pipeline.current_image, T)
        except Exception as exc:
            self._on_status(f"Threshold error: {exc}", "error")
            return
        self._pipeline.push(result, f"Threshold T={T}")
        self._on_image_updated(result)
        self._on_pipeline_updated()
        self._on_status(f"Binarized at T={T}", "ok")

    def _apply(self, op: str):
        """Apply the selected morphology operation using the current SE."""
        if self._pipeline.is_empty:
            self._on_status("Load an image first.", "warn")
            return
        from bonus.morphology import erode, dilate, opening, closing, boundary_extraction
        ops = {
            "Erode":    (erode,              "Erosion"),
            "Dilate":   (dilate,             "Dilation"),
            "Opening":  (opening,            "Opening"),
            "Closing":  (closing,            "Closing"),
            "Boundary": (boundary_extraction,"Boundary Extraction"),
        }
        fn, label = ops[op]
        try:
            structuring_element = self._get_se()
        except Exception as exc:
            self._on_status(f"Invalid structuring element: {exc}", "error")
            return
        structuring_element_info = f"{self._se_size_var.get()} {self._se_shape_var.get()}"
        try:
            result = fn(self._pipeline.current_image, structuring_element)
        except Exception as exc:
            self._on_status(f"Morphology error: {exc}", "error")
            return
        desc = f"{label} ({structuring_element_info})"
        self._pipeline.push(result, desc)
        self._on_image_updated(result)
        self._on_pipeline_updated()
        self._on_status(f"Applied: {desc}", "ok")
