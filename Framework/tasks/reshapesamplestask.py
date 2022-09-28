from dataclasses import dataclass
import numpy as np
from Framework.task import AbstractTask

@dataclass
class ConvertSamplesToRowsTask(AbstractTask):
    input_data_var: str
    output_data_var: str

    def process(self, data_store):
        """Reshape samples into one dimensional rows"""
        input_data = data_store[self.input_data_var]
        data_store[self.output_data_var] = np.reshape(input_data, (input_data.shape[0], -1))

        print(f"Reshaped {self.input_data_var}")
        