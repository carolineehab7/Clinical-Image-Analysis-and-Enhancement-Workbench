import numpy as np
from typing import Callable, List, Optional, Tuple


class Pipeline:
    def __init__(self):
        self._stack: List[Tuple[np.ndarray, str]] = []

    def load(self, image: np.ndarray) -> None:
        self._stack = [(image.copy(), "Original")]

    def clear(self) -> None:
        """Remove all images from the pipeline."""
        self._stack = []

    @property
    def is_empty(self) -> bool:
        return len(self._stack) == 0

    def apply(self, operation: Callable[[np.ndarray], np.ndarray],
              description: str) -> Optional[np.ndarray]:
        if self.is_empty:
            return None

        current = self._stack[-1][0]
        result = operation(current)
        self._stack.append((result.copy(), description))
        return result

    def push(self, image: np.ndarray, description: str) -> np.ndarray:
        self._stack.append((image.copy(), description))
        return image

    def undo(self) -> Optional[np.ndarray]:
        if len(self._stack) > 1:
            self._stack.pop()
        return self.current_image

    def reset(self) -> Optional[np.ndarray]:
        if len(self._stack) >= 1:
            original = self._stack[0]
            self._stack = [original]
        return self.current_image

    @property
    def current_image(self) -> Optional[np.ndarray]:
        if self._stack:
            return self._stack[-1][0]
        return None

    @property
    def original_image(self) -> Optional[np.ndarray]:
        if self._stack:
            return self._stack[0][0]
        return None

    @property
    def step_count(self) -> int:
        return len(self._stack)

    @property
    def steps(self) -> List[str]:
        return [desc for _, desc in self._stack]

    @property
    def history(self) -> List[Tuple[np.ndarray, str]]:
        return self._stack
