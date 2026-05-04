from tkinter import messagebox
from core.interpolation import nearest_neighbor_zoom, bilinear_zoom

def do_zoom(self, factor: float):
    if self.pipeline.is_empty:
        messagebox.showwarning("No Image", "Load an image first.")
        return

    new_scale = self._zoom_level * factor
    if new_scale < 0.1 or new_scale > 8.0:
        self._set_status("Zoom limit reached (0.1× – 8.0×).", "warn")
        return

    current = self.pipeline.current_image
    interp = self._interp_var.get()
    step = "Zoom In" if factor > 1 else "Zoom Out"
    label = f"{step} ×{factor:.2f} ({interp}) → Scale {new_scale:.2f}×"

    if interp == "Nearest Neighbor":
        result = nearest_neighbor_zoom(current, new_scale)
    else:  
        result = bilinear_zoom(current, new_scale)

    self._zoom_level = new_scale
    self._zoom_lbl.configure(text=f"Scale: {new_scale:.2f}×")
    self.pipeline.push(result, label)
    self._display_image(result)
    self._update_pipeline_display()
    self._set_status(label, "ok")

def zoom_in(self):
    do_zoom(self, 2)

def zoom_out(self):
    do_zoom(self, 0.5)