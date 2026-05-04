"""
Pipeline Module — Sequential Enhancement State Manager.

Maintains the history of image processing steps so that:
  - Operations are applied cumulatively on top of the previous result.
  - "Undo" removes the last step and returns to the prior state.
  - "Reset" restores the original loaded image.
"""

import numpy as np
from typing import Callable, List, Optional, Tuple


class Pipeline:
    """
    Manages an ordered stack of (image_array, step_description) pairs.

    The stack always starts with the original image at index 0.
    Each subsequent call to apply() pushes a new entry onto the stack.
    """

    def __init__(self):
        self._stack: List[Tuple[np.ndarray, str]] = []

    # ─────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────

    def load(self, image: np.ndarray) -> None:
        """
        Set the original image and clear all prior history.
        Must be called every time a new image is loaded.
        """
        self._stack = [(image.copy(), "Original")]

    @property
    def is_empty(self) -> bool:
        return len(self._stack) == 0

    # ─────────────────────────────────────────────────────
    # Applying Operations
    # ─────────────────────────────────────────────────────

    def apply(self, operation: Callable[[np.ndarray], np.ndarray],
              description: str) -> Optional[np.ndarray]:
        """
        Apply an operation to the current (most recent) image.

        Args:
            operation:   A function that takes a numpy array and returns a numpy array.
            description: Human-readable label for this step (shown in the pipeline log).

        Returns:
            The processed image, or None if the pipeline has no image loaded.
        """
        if self.is_empty:
            return None

        current = self._stack[-1][0]
        result = operation(current)
        self._stack.append((result.copy(), description))
        return result

    def push(self, image: np.ndarray, description: str) -> np.ndarray:
        """
        Directly push a pre-computed image onto the stack.
        Used when the caller has already computed the result (e.g., zoom).
        """
        self._stack.append((image.copy(), description))
        return image

    # ─────────────────────────────────────────────────────
    # Navigation
    # ─────────────────────────────────────────────────────

    def undo(self) -> Optional[np.ndarray]:
        """
        Remove the last step. Cannot undo past the original image.

        Returns:
            The image that is now current after the undo, or None.
        """
        if len(self._stack) > 1:
            self._stack.pop()
        return self.current_image

    def reset(self) -> Optional[np.ndarray]:
        """
        Discard all steps and return to the original image.

        Returns:
            The original image, or None if nothing is loaded.
        """
        if len(self._stack) >= 1:
            original = self._stack[0]
            self._stack = [original]
        return self.current_image

    # ─────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────

    @property
    def current_image(self) -> Optional[np.ndarray]:
        """Return the most recent image in the pipeline."""
        if self._stack:
            return self._stack[-1][0]
        return None

    @property
    def original_image(self) -> Optional[np.ndarray]:
        """Return the original loaded image."""
        if self._stack:
            return self._stack[0][0]
        return None

    @property
    def step_count(self) -> int:
        return len(self._stack)

    @property
    def steps(self) -> List[str]:
        """Return list of step description strings."""
        return [desc for _, desc in self._stack]

    @property
    def history(self) -> List[Tuple[np.ndarray, str]]:
        """Return the full (image, description) history stack."""
        return self._stack
