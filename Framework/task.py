from dataclasses import dataclass
from typing import Callable, Protocol

class Task(Protocol):
    def process(self, data_store) -> None:
        """Process the data"""

class AbstractTask:
    def __init__(self):   
        self.create_resource_path: Callable[[str], None] = None

    def set_resource_path_func(self, fn):
        self.create_resource_path = fn
