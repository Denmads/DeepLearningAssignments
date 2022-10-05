from dataclasses import dataclass
from typing import Callable
from Framework.task import AbstractTask

@dataclass
class CustomTask(AbstractTask):
    fn: Callable[[dict], None] # For params, use partial functions

    def process(self, data_store):
        self.fn(data_store, self.create_resource_path)