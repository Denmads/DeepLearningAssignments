from dataclasses import dataclass
from Framework.task import AbstractTask
import numpy as np

@dataclass
class CalculateMeanTask(AbstractTask):
    samples_to_calc_from: str
    mean_var_name: str

    def process(self, data_store):
        calc_samples = data_store[self.samples_to_calc_from]
        mean = np.mean(calc_samples, axis=0)
        data_store[self.mean_var_name] = mean

        print("Calculated Mean")


@dataclass
class SubtractMeanTask(AbstractTask):
    mean_var_name: str
    samples_to_subtract_from: list[str]

    def process(self, data_store):
        mean = data_store[self.mean_var_name]
        for samples in self.samples_to_subtract_from:
            data_store[samples] -= mean
        
        print("Subtracted Mean")

@dataclass
class AppendBiasTask(AbstractTask):
    samples_to_append_to: list[str]

    def process(self, data_store):
        for samples in self.samples_to_append_to:
            data_store[samples] = np.hstack([
                data_store[samples],
                np.ones(
                    (data_store[samples].shape[0], 1)
                )
            ])
        print("Appended Bias")