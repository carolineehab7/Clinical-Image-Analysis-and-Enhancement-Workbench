"""FFT panel for the left sidebar (Member 1 — Show Spectrum button)."""
import customtkinter as ctk

from gui.theme import BORDER_CYAN, FONT_SMALL, FONT_TITLE, TEXT_DIM, TEXT_MAIN


class FFTPanel:
    """Section in the left sidebar that opens the SpectrumWindow."""

    def __init__(self, parent, pipeline, on_image_updated, on_pipeline_updated, on_status):
        self._parent              = parent
        self._pipeline            = pipeline
        self._on_image_updated    = on_image_updated
        self._on_pipeline_updated = on_pipeline_updated
        self._on_status           = on_status
        self._win                 = None   # current SpectrumWindow instance

        ctk.CTkLabel(parent, text="FFT & NOTCH FILTER",
                     font=FONT_TITLE, text_color=TEXT_MAIN).pack(
                         anchor="w", padx=12, pady=(12, 2))

        ctk.CTkLabel(
            parent,
            text="View the frequency spectrum and\napply notch filters to remove\nperiodic noise patterns.",
            font=FONT_SMALL, text_color=TEXT_DIM, justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 6))

        ctk.CTkButton(parent, text="Show Spectrum / Apply Notch",
                      command=self._open_spectrum).pack(padx=12, pady=4, fill="x")

        ctk.CTkFrame(parent, height=2, fg_color=BORDER_CYAN).pack(
            fill="x", padx=8, pady=8)

    def _open_spectrum(self):
        if self._pipeline.is_empty:
            self._on_status("Load an image first.", "warn")
            return

        # Re-use existing window if still open
        try:
            if self._win and self._win.winfo_exists():
                self._win.lift()
                return
        except Exception:
            pass

        from gui.spectrum_window import SpectrumWindow
        root = self._parent.winfo_toplevel()
        self._win = SpectrumWindow(
            root,
            self._pipeline,
            self._on_image_updated,
            self._on_pipeline_updated,
            self._on_status,
        )
