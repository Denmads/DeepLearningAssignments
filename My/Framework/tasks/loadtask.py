from dataclasses import dataclass
import numpy as np
from Framework.task import AbstractTask

@dataclass
class LoadNpzTask(AbstractTask):
    path_to_file: str
    output_name: str

    def process(self, data_store):
        """Load data from the file"""
        raw_data = np.load(self.path_to_file, allow_pickle=True)
        data_store[self.output_name] = raw_data

        print(f"Loaded data from file '{self.path_to_file}'")
        print(f"Number of samples: {raw_data['images'].shape[0]}")
        print()