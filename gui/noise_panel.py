import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from core.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
    add_uniform_noise,
)


FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_SMALL = ("Segoe UI", 10)
TEXT_DIM = "#888888"


def parse_float(value: str, default: float) -> float:
    text = value.strip()
    return float(text) if text else default


class NoisePanel:
    def __init__(self, parent, pipeline, on_image_updated, on_pipeline_updated, on_status):
        self.parent = parent
        self.pipeline = pipeline
        self.on_image_updated = on_image_updated
        self.on_pipeline_updated = on_pipeline_updated
        self.on_status = on_status
        self._build_ui()

    def _build_ui(self):
        self._section_title("NOISE GENERATION")

        ctk.CTkLabel(self.parent, text="Noise Type:", font=FONT_SMALL,
                     text_color=TEXT_DIM).pack(anchor="w", padx=12)
        self._noise_var = tk.StringVar(value="Gaussian")
        ctk.CTkOptionMenu(
            self.parent,
            variable=self._noise_var,
            values=["Gaussian", "Salt & Pepper", "Speckle", "Uniform"],
            command=self._on_noise_change,
            width=226,
        ).pack(padx=12, pady=3)

        self._strength_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
        self._strength_label = ctk.CTkLabel(
            self._strength_frame,
            text="Strength / Sigma:",
            font=FONT_SMALL,
            text_color=TEXT_DIM,
        )
        self._strength_label.pack(anchor="w")
        self._strength_entry = ctk.CTkEntry(self._strength_frame,
                                         placeholder_text="e.g. 10 or 0.1",
                                         width=226)
        self._strength_entry.pack()

        self._pepper_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
        ctk.CTkLabel(self._pepper_frame, text="Salt Ratio:",
                     font=FONT_SMALL, text_color=TEXT_DIM).pack(anchor="w")
        self._salt_ratio_entry = ctk.CTkEntry(self._pepper_frame,
                                              placeholder_text="0.5",
                                              width=226)
        self._salt_ratio_entry.pack()

        ctk.CTkButton(
            self.parent,
            text="▶  Apply Noise",
            command=self._apply_noise,
        ).pack(padx=12, pady=6, fill="x")

        self._on_noise_change(self._noise_var.get())
        self._divider()

    def _section_title(self, text):
        ctk.CTkLabel(self.parent, text=text, font=FONT_TITLE).pack(
            anchor="w", padx=12, pady=(12, 2))

    def _divider(self):
        ctk.CTkFrame(self.parent, height=2, fg_color="#0f3460").pack(
            fill="x", padx=8, pady=8)

    def _on_noise_change(self, choice):
        self._strength_frame.pack_forget()
        self._pepper_frame.pack_forget()

        if choice == "Salt & Pepper":
            self._strength_label.configure(text="Noise Amount:")
            self._strength_entry.configure(placeholder_text="0.05")
            self._strength_frame.pack(padx=12, pady=2, fill="x")
            self._pepper_frame.pack(padx=12, pady=2, fill="x")
            self._strength_entry.delete(0, "end")
            self._strength_entry.insert(0, "0.05")
            self._salt_ratio_entry.delete(0, "end")
            self._salt_ratio_entry.insert(0, "0.5")
        elif choice == "Gaussian":
            self._strength_label.configure(text="Strength / Sigma:")
            self._strength_entry.configure(placeholder_text="10")
            self._strength_frame.pack(padx=12, pady=2, fill="x")
            self._strength_entry.delete(0, "end")
            self._strength_entry.insert(0, "10")
        elif choice == "Speckle":
            self._strength_label.configure(text="Strength / Sigma:")
            self._strength_entry.configure(placeholder_text="0.1")
            self._strength_frame.pack(padx=12, pady=2, fill="x")
            self._strength_entry.delete(0, "end")
            self._strength_entry.insert(0, "0.1")
        elif choice == "Uniform":
            self._strength_label.configure(text="Noise Strength:")
            self._strength_entry.configure(placeholder_text="20")
            self._strength_frame.pack(padx=12, pady=2, fill="x")
            self._strength_entry.delete(0, "end")
            self._strength_entry.insert(0, "20")

    def _apply_noise(self):
        if self.pipeline.is_empty:
            messagebox.showwarning("No Image", "Load an image first.")
            return

        choice = self._noise_var.get()
        current = self.pipeline.current_image

        try:
            if choice == "Gaussian":
                sigma = parse_float(self._strength_entry.get(), 10.0)
                if sigma <= 0:
                    raise ValueError("Sigma must be positive.")
                result = add_gaussian_noise(current, sigma=sigma)
                desc = f"Gaussian Noise σ={sigma}"

            elif choice == "Salt & Pepper":
                amount = parse_float(self._strength_entry.get(), 0.05)
                salt_ratio = parse_float(self._salt_ratio_entry.get(), 0.5)
                if not 0 <= amount <= 1:
                    raise ValueError("Noise amount must be between 0 and 1.")
                if not 0 <= salt_ratio <= 1:
                    raise ValueError("Salt ratio must be between 0 and 1.")
                result = add_salt_pepper_noise(current, amount=amount, salt_vs_pepper=salt_ratio)
                desc = f"Salt & Pepper Noise amount={amount}, salt={salt_ratio}"

            elif choice == "Speckle":
                sigma = parse_float(self._strength_entry.get(), 0.1)
                if sigma <= 0:
                    raise ValueError("Sigma must be positive.")
                result = add_speckle_noise(current, sigma=sigma)
                desc = f"Speckle Noise σ={sigma}"

            elif choice == "Uniform":
                strength = parse_float(self._strength_entry.get(), 20.0)
                if strength < 0:
                    raise ValueError("Noise strength must be non-negative.")
                result = add_uniform_noise(current, low=-strength, high=strength)
                desc = f"Uniform Noise ±{strength}"

            else:
                return

        except ValueError as exc:
            messagebox.showerror("Noise Input Error", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Noise Error", str(exc))
            return

        self.pipeline.push(result, desc)
        self.on_image_updated(result)
        self.on_pipeline_updated()
        self.on_status(f"Applied: {desc}", "ok")