"User interface for filter selection"
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from core.filters import box_smoothing_filter, gaussian_smoothing_filter, median_filter
from gui.theme import ACCENT_CYAN, ACCENT_PURPLE, BG_CARD, BG_DIVIDER, BORDER_CYAN, BG_SURFACE, FONT_TITLE, FONT_SMALL, TEXT_DIM, TEXT_MAIN


def parse_kernel_size(s: str) -> int:
	return int(s.split("x")[0])
"parses the kernel size from mXm -> m"

class FilterPanel:
	"""Smoothing filter controls and operations"""

	_FILTER_HINTS = {
		"Average": "Best for mild Gaussian, Poisson, and uniform noise.",
		"Gaussian": "Recommended for Gaussian and speckle noise; better edge preservation than average.",
		"Median": "Best for salt-and-pepper noise and other impulse noise.",
	}

	def __init__(self, parent, pipeline, on_image_updated, on_pipeline_updated, on_status):
		self.parent = parent
		self.pipeline = pipeline
		self.on_image_updated = on_image_updated
		self.on_pipeline_updated = on_pipeline_updated
		self.on_status = on_status
		self._build_ui()

	def _build_ui(self):
		"Drop down menu to allow user to select filter type and window size"
		self._card = ctk.CTkFrame(
			self.parent,
			fg_color=BG_CARD,
			corner_radius=10,
			border_width=2,
			border_color=BORDER_CYAN,
		)
		self._card.pack(fill="x", padx=10, pady=(0, 8))

		self._section_title(self._card, "SMOOTHING FILTERS")

		ctk.CTkLabel(self._card, text="Filter Type:", font=FONT_SMALL,
					 text_color=TEXT_DIM).pack(anchor="w", padx=12)
		self._filter_var = tk.StringVar(value="Average")
		ctk.CTkOptionMenu(self._card, variable=self._filter_var,
						  values=["Average", "Gaussian", "Median"],
						  command=self._on_filter_change,
						  width=226).pack(padx=12, pady=3)
		#Kernel size options: Smaller kernels (3x3, 5x5) are better for mild noise and better edge preservation, while larger kernels (7x7, 9x9)stronger smoothing but more blurring.
		ctk.CTkLabel(self._card, text="Kernel Size:", font=FONT_SMALL,
					 text_color=TEXT_DIM).pack(anchor="w", padx=12)
		self._kernel_var = tk.StringVar(value="3x3")
		ctk.CTkOptionMenu(self._card, variable=self._kernel_var,
						  values=["3x3", "5x5", "7x7", "9x9"],
						  width=226).pack(padx=12, pady=3)

		self._hint_label = ctk.CTkLabel(
			self._card,
			text=self._FILTER_HINTS["Average"],
			font=FONT_SMALL,
			text_color=TEXT_DIM,
			justify="left",
			wraplength=226,
		) 
		self._hint_label.pack(anchor="w", padx=12, pady=(0, 4))
		self._gaussian_frame = ctk.CTkFrame(self._card, fg_color="transparent")
		ctk.CTkLabel(self._gaussian_frame, text="Variance (σ²):",
					 font=FONT_SMALL, text_color=TEXT_DIM).pack(anchor="w")
		self._sigma_entry = ctk.CTkEntry(self._gaussian_frame,
										 placeholder_text="e.g. 2.25", width=226)
		self._sigma_entry.pack()

		self._apply_button = ctk.CTkButton(
			self._card,
			text="▶  Apply Smoothing",
			command=self._apply_smoothing,
			fg_color=ACCENT_CYAN,
			hover_color="#00869B",
		)
		self._apply_button.pack(padx=12, pady=6, fill="x")

		self._divider()

	def _section_title(self, parent, text):
		ctk.CTkLabel(parent, text=text, font=("Segoe UI", 14, "bold"), text_color=TEXT_MAIN).pack(
			anchor="w", padx=12, pady=(12, 2))

	def _divider(self):
		ctk.CTkFrame(self.parent, height=2, fg_color=BG_DIVIDER).pack(
			fill="x", padx=8, pady=8)

	def _on_filter_change(self, choice):
		self._hint_label.configure(text=self._FILTER_HINTS.get(choice, ""))
		if choice == "Gaussian":
			self._kernel_var.set("9x9")
			self._gaussian_frame.pack(padx=12, pady=2, fill="x", before=self._apply_button)
			self._sigma_entry.delete(0, "end")
			self._sigma_entry.insert(0, "32.0") # Default variance for 9x9 Gaussian kernel
		else:
			self._gaussian_frame.pack_forget()

	def _apply_smoothing(self):
		if self.pipeline.is_empty:
			messagebox.showwarning("No Image", "Load an image first.")
			return

		ftype = self._filter_var.get()
		ksize = parse_kernel_size(self._kernel_var.get())

		try:
            #Applying the specified filter to the image and documenting it in the pipeline.
			if ftype == "Average":
				desc = f"Average Filter {ksize}×{ksize}"
				result = self.pipeline.apply(lambda img: box_smoothing_filter(img, ksize), desc)

			elif ftype == "Gaussian":
				variance_txt = self._sigma_entry.get().strip() or "2.25"
				variance = float(variance_txt)
				if variance <= 0:
					raise ValueError("Variance must be positive.")
				desc = f"Gaussian Filter {ksize}×{ksize}, σ²={variance}"
				result = self.pipeline.apply(
					lambda img, k=ksize, v=variance: gaussian_smoothing_filter(img, k, v), desc
				)

			elif ftype == "Median":
				desc = f"Median Filter {ksize}×{ksize}"
				result = self.pipeline.apply(lambda img: median_filter(img, ksize), desc)

			else:
				return

		except ValueError as exc:
			messagebox.showerror("Input Error", str(exc))
			return
		except Exception as exc:
			messagebox.showerror("Filter Error", str(exc))
			return

		self.on_image_updated(result)
		self.on_pipeline_updated()
		self.on_status(f"Applied: {desc}", "ok")
