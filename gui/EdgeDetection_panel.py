import tkinter as tk
from tkinter import messagebox

import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np

from core.filters import sobel_filter
from gui.theme import ACCENT_CYAN, ACCENT_PURPLE, BG_ELEVATED, BG_SURFACE, BORDER_CYAN, FONT_SMALL, FONT_TITLE, TEXT_DIM, TEXT_MAIN

BG_MID = BG_SURFACE


def _array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8) 
    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode='RGB')
    elif arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr[:, :, :3], mode='RGB')
    return Image.fromarray(arr)

# show the three effects in one window
class EdgeResultsWindow(ctk.CTkToplevel):
    def __init__(self, parent, gx, gy, mag, detector_name): #constructor
        super().__init__(parent)
        self.title(f"{detector_name} Edge Detection Results")
        self.geometry("900x380")
        self.resizable(False, False)
        self.configure(fg_color=BG_SURFACE)

        ctk.CTkLabel(self, text=f"{detector_name} Edge Detection",
                 font=FONT_TITLE, text_color=TEXT_MAIN).pack(pady=8)

        frame = ctk.CTkFrame(self, fg_color=BG_SURFACE)
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        thumb_size = (260, 260)
        images = [("Horizontal (Gx)", gx), ("Vertical (Gy)", gy), ("Magnitude (Combined)", mag)]

        self._photos = []
        for label_text, arr in images:
            col = ctk.CTkFrame(frame, fg_color=BG_SURFACE)
            col.pack(side="left", expand=True, fill="both", padx=5, pady=5)

            ctk.CTkLabel(col, text=label_text, font=FONT_SMALL, text_color=TEXT_MAIN).pack(pady=3)

            pil = _array_to_pil(arr).resize(thumb_size, Image.NEAREST)
            photo = ImageTk.PhotoImage(pil)
            self._photos.append(photo)

            lbl = tk.Label(col, image=photo, bg=BG_MID)
            lbl.pack()


class EdgeDetectionPanel:

    def __init__(self, parent, pipeline, on_image_updated, on_pipeline_updated, on_status):
        self.parent = parent
        self.pipeline = pipeline
        self.on_image_updated = on_image_updated
        self.on_pipeline_updated = on_pipeline_updated
        self.on_status = on_status
        self._edge_cache = None
        self._build_ui()
    
    #UI
    def _build_ui(self):
        self._card = ctk.CTkFrame(
            self.parent,
            fg_color=BG_ELEVATED,
            corner_radius=10,
            border_width=1,
            border_color=BORDER_CYAN,
        )
        self._card.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(self._card, text="EDGE DETECTION", font=FONT_TITLE, text_color=TEXT_MAIN).pack(
            anchor="w", padx=12, pady=(12, 2)
        )

        ctk.CTkLabel(self._card, text="Detector:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self.detector_var = tk.StringVar(value="Sobel")
        ctk.CTkOptionMenu(self._card, variable=self.detector_var,
                          values=["Sobel"], width=226).pack(padx=12, pady=3)

        ctk.CTkLabel(self._card, text="Apply to Pipeline:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self.apply_var = tk.StringVar(value="Magnitude (Combined)")
        ctk.CTkOptionMenu(self._card, variable=self.apply_var,
                          values=["Horizontal (Gx)", "Vertical (Gy)", "Magnitude (Combined)"],
                          width=226).pack(padx=12, pady=3)

        edge_row = ctk.CTkFrame(self._card, fg_color="transparent")
        edge_row.pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(edge_row, text="▶  Apply", width=107,
                      command=self._apply_edge,
                      fg_color=ACCENT_CYAN,
                      hover_color="#00869B").pack(side="left", padx=2)
        ctk.CTkButton(edge_row, text=" All 3", width=107,
                      command=self._show_all_edges,
                      fg_color=ACCENT_PURPLE,
                      hover_color="#9F67D8").pack(side="right", padx=2)

        self._divider()

    def _divider(self):
        ctk.CTkFrame(self._card, height=2, fg_color=BORDER_CYAN).pack(fill="x", padx=8, pady=8)

    # calculation
    def _run_edge_detection(self):
        if self.pipeline.is_empty:
            messagebox.showwarning("No Image", "Load an image first.")
            return None

        detector = self.detector_var.get()
        current = self.pipeline.current_image

        try:
            if detector == "Sobel":
                gx, gy, mag = sobel_filter(current)
            else:
                return None
        except Exception as exc:
            messagebox.showerror("Edge Detection Error", str(exc))
            return None

        self._edge_cache = (gx, gy, mag, detector)
        return gx, gy, mag, detector

    # apply the selected edge result 
    def _apply_edge(self):
        result = self._run_edge_detection()
        if result is None:
            return
        gx, gy, mag, detector = result

        choice = self.apply_var.get()
        if "Horizontal" in choice:
            img = gx
            kind = "Gx"
        elif "Vertical" in choice:
            img = gy
            kind = "Gy"
        else:
            img = mag
            kind = "Magnitude"

        desc = f"{detector} Edge — {kind}"
        self.pipeline.push(img, desc)
        self.on_image_updated(img)
        self.on_pipeline_updated()
        self.on_status(f"Applied: {desc}", "ok")

    def _show_all_edges(self):
        result = self._run_edge_detection()
        if result is None:
            return
        gx, gy, mag, detector = result
        EdgeResultsWindow(self.parent, gx, gy, mag, detector)
