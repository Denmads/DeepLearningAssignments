from dataclasses import dataclass
import numpy as np
from Framework.task import AbstractTask

@dataclass
class CalculateAccuracyTask(AbstractTask):
    prediction_var_name: str
    true_label_var_name: str
    var_name: str = None

    def process(self, data_store):
        """Take the predictions and the actual labels and calculate accuracy"""
        predictions = data_store[self.prediction_var_name]
        true_labels = data_store[self.true_label_var_name]

        num_correct = np.sum(predictions == true_labels)
        accuracy = float(num_correct) / float(predictions.shape[0])

        if self.var_name is not None:
            data_store[self.var_name] = accuracy

        print(f"Accuracy: {accuracy}")