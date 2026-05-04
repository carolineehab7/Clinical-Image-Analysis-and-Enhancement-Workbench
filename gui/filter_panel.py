import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from core.filters import average_filter, gaussian_filter, median_filter


FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_SMALL = ("Segoe UI", 10)
TEXT_DIM = "#888888"


def parse_kernel_size(s: str) -> int:
	"""Convert a string like '3x3' to the integer kernel size 3."""
	return int(s.split("x")[0])


class FilterPanel:
	"""Smoothing filter controls and operations (Member 2 responsibilities)."""

	def __init__(self, parent, pipeline, on_image_updated, on_pipeline_updated, on_status):
		self.parent = parent
		self.pipeline = pipeline
		self.on_image_updated = on_image_updated
		self.on_pipeline_updated = on_pipeline_updated
		self.on_status = on_status
		self._build_ui()

	def _build_ui(self):
		self._section_title("SMOOTHING FILTERS")

		ctk.CTkLabel(self.parent, text="Filter Type:", font=FONT_SMALL,
					 text_color=TEXT_DIM).pack(anchor="w", padx=12)
		self._filter_var = tk.StringVar(value="Average")
		ctk.CTkOptionMenu(self.parent, variable=self._filter_var,
						  values=["Average", "Gaussian", "Median"],
						  command=self._on_filter_change,
						  width=226).pack(padx=12, pady=3)

		ctk.CTkLabel(self.parent, text="Kernel Size:", font=FONT_SMALL,
					 text_color=TEXT_DIM).pack(anchor="w", padx=12)
		self._kernel_var = tk.StringVar(value="3x3")
		ctk.CTkOptionMenu(self.parent, variable=self._kernel_var,
						  values=["3x3", "5x5", "7x7", "9x9"],
						  width=226).pack(padx=12, pady=3)

		self._sigma_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
		ctk.CTkLabel(self._sigma_frame, text="Sigma (σ):",
					 font=FONT_SMALL, text_color=TEXT_DIM).pack(anchor="w")
		self._sigma_entry = ctk.CTkEntry(self._sigma_frame,
										 placeholder_text="e.g. 1.5", width=226)
		self._sigma_entry.pack()

		ctk.CTkButton(self.parent, text="▶  Apply Smoothing",
					  command=self._apply_smoothing).pack(padx=12, pady=6, fill="x")

		self._divider()

	def _section_title(self, text):
		ctk.CTkLabel(self.parent, text=text, font=FONT_TITLE).pack(
			anchor="w", padx=12, pady=(12, 2))

	def _divider(self):
		ctk.CTkFrame(self.parent, height=2, fg_color="#0f3460").pack(
			fill="x", padx=8, pady=8)

	def _on_filter_change(self, choice):
		if choice == "Gaussian":
			self._sigma_frame.pack(padx=12, pady=2, fill="x")
		else:
			self._sigma_frame.pack_forget()

	def _apply_smoothing(self):
		if self.pipeline.is_empty:
			messagebox.showwarning("No Image", "Load an image first.")
			return

		ftype = self._filter_var.get()
		ksize = parse_kernel_size(self._kernel_var.get())

		try:
			if ftype == "Average":
				desc = f"Average Filter {ksize}×{ksize}"
				result = self.pipeline.apply(lambda img: average_filter(img, ksize), desc)

			elif ftype == "Gaussian":
				sigma_txt = self._sigma_entry.get().strip() or "1.5"
				sigma = float(sigma_txt)
				if sigma <= 0:
					raise ValueError("Sigma must be positive.")
				desc = f"Gaussian Filter {ksize}×{ksize}, σ={sigma}"
				result = self.pipeline.apply(
					lambda img, k=ksize, s=sigma: gaussian_filter(img, k, s), desc
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
