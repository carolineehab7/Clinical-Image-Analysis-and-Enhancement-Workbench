import numpy as np
from typing import Callable, List, Optional, Tuple


class Pipeline:
    # the image as pixels and the description
    def __init__(self):
        self._stack: List[Tuple[np.ndarray, str]] = []
        self._redo_stack: List[Tuple[np.ndarray, str]] = []

    # Load an image into the pipeline and replacing any existing content
    # .copy is used to ensure that the original image data is not modified outside the pipeline
    def load(self, image: np.ndarray) -> None:
        self._stack = [(image.copy(), "Original")]
        self._redo_stack.clear()

    def clear(self) -> None:
        self._stack = []
        self._redo_stack.clear()

    #checks if the pipeline is empty and property decorator allows us to access it like an attribute
    @property 
    def is_empty(self) -> bool:
        return len(self._stack) == 0

    #applies a given operation to the current image and pushes the result onto the stack with a description
    def apply(self, operation: Callable[[np.ndarray], np.ndarray],
              description: str) -> Optional[np.ndarray]:
        if self.is_empty:
            return None
        current = self._stack[-1][0]
        result = operation(current)
        self._stack.append((result.copy(), description))
        self._redo_stack.clear()
        return result

    #pushes a new image and description onto the stack, allowing for manual addition of images without applying an operation
    def push(self, image: np.ndarray, description: str) -> np.ndarray:
        self._stack.append((image.copy(), description))
        self._redo_stack.clear()
        return image

    #removes the most recent image from the stack undoing the last operation, and returns the new current image
    def undo(self) -> Optional[np.ndarray]:
        if len(self._stack) > 1:
            self._redo_stack.append(self._stack.pop())
        return self.current_image

    #re-applies the last undone operation
    def redo(self) -> Optional[np.ndarray]:
        if self._redo_stack:
            self._stack.append(self._redo_stack.pop())
        return self.current_image

    #resets the pipeline to the original image by keeping only the first entry in the stack, and returns the current image after reset
    def reset(self) -> Optional[np.ndarray]:
        if len(self._stack) >= 1:
            original = self._stack[0]
            self._stack = [original]
        return self.current_image

    #returns the current image at the top of the stack, or None if the stack is empty
    @property
    def current_image(self) -> Optional[np.ndarray]:
        if self._stack:
            return self._stack[-1][0]

    # Public read-only views used by PipelinePanel
    @property
    def steps(self) -> list:
        return [desc for _, desc in self._stack]

    @property
    def step_count(self) -> int:
        return len(self._stack)

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0
