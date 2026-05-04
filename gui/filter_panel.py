import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from core.filters import average_filter, gaussian_filter
from core.Median import median_filter


FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_SMALL = ("Segoe UI", 10)
TEXT_DIM = "#888888"


def parse_kernel_size(s: str) -> int:
	"""Convert a string like '3x3' to the integer kernel size 3."""
	return int(s.split("x")[0])


class FilterPanel:
	"""Smoothing filter controls and operations (Member 2 responsibilities)."""

	_FILTER_HINTS = {
		"Average": "Best for mild Gaussian, Poisson, and uniform noise.",
		"Gaussian": "Best for Gaussian, speckle, Poisson, and uniform noise.",
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

		self._hint_label = ctk.CTkLabel(
			self.parent,
			text=self._FILTER_HINTS["Average"],
			font=FONT_SMALL,
			text_color=TEXT_DIM,
			justify="left",
			wraplength=226,
		)
		self._hint_label.pack(anchor="w", padx=12, pady=(0, 4))

		self._gaussian_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
		ctk.CTkLabel(self._gaussian_frame, text="Variance (σ²):",
					 font=FONT_SMALL, text_color=TEXT_DIM).pack(anchor="w")
		self._sigma_entry = ctk.CTkEntry(self._gaussian_frame,
										 placeholder_text="e.g. 2.25", width=226)
		self._sigma_entry.pack()

		self._apply_button = ctk.CTkButton(
			self.parent,
			text="▶  Apply Smoothing",
			command=self._apply_smoothing,
		)
		self._apply_button.pack(padx=12, pady=6, fill="x")

		self._divider()

	def _section_title(self, text):
		ctk.CTkLabel(self.parent, text=text, font=FONT_TITLE).pack(
			anchor="w", padx=12, pady=(12, 2))

	def _divider(self):
		ctk.CTkFrame(self.parent, height=2, fg_color="#0f3460").pack(
			fill="x", padx=8, pady=8)

	def _on_filter_change(self, choice):
		self._hint_label.configure(text=self._FILTER_HINTS.get(choice, ""))
		if choice == "Gaussian":
			self._gaussian_frame.pack(padx=12, pady=2, fill="x", before=self._apply_button)
		else:
			self._gaussian_frame.pack_forget()

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
				variance_txt = self._sigma_entry.get().strip() or "2.25"
				variance = float(variance_txt)
				if variance <= 0:
					raise ValueError("Variance must be positive.")
				desc = f"Gaussian Filter {ksize}×{ksize}, σ²={variance}"
				result = self.pipeline.apply(
					lambda img, k=ksize, v=variance: gaussian_filter(img, k, v), desc
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
