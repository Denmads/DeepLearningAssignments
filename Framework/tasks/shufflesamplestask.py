from dataclasses import dataclass
import random
import numpy as np
from Framework.task import AbstractTask

@dataclass
class ShuffleSamplesTask(AbstractTask):
    sample_var_name: str
    labels_var_name: str
    

    def process(self, data_store):
        """Shuffle the samples and store them back in the datastore"""
        samples = data_store[self.sample_var_name]
        labels = data_store[self.labels_var_name]
        sample_arr = list(zip(samples, labels))

        random.shuffle(sample_arr)

        samples, labels = zip(*sample_arr)
        data_store[self.sample_var_name] = np.array(samples)
        data_store[self.labels_var_name] = np.array(labels)

        print(f"Shuffled {self.sample_var_name} and {self.labels_var_name}")
        