from dataclasses import dataclass
from Framework.task import AbstractTask

@dataclass
class SubSampleTask(AbstractTask):
    input_data_var: str
    output_data_var: str
    number_samples: int

    def process(self, data_store):
        """Take a subset of the samples"""
        num_samples = min(self.number_samples, data_store[self.input_data_var].shape[0])
        mask = list(range(num_samples))
        data_store[self.output_data_var] = data_store[self.input_data_var][mask]

        print(f"Subsampled {self.input_data_var}")
        