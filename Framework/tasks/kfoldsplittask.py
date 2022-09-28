from dataclasses import dataclass
from Framework.task import AbstractTask
import numpy as np

@dataclass
class KfoldSplitTask(AbstractTask):
    input_data_var: str
    out_data_var: str
    num_folds: str = 3

    def process(self, data_store):
        data = data_store[self.input_data_var]
        folds = np.array_split(data, self.num_folds)

        data_store[self.out_data_var] = folds