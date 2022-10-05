from dataclasses import dataclass
from Framework.classifier import Classifier
from Framework.task import AbstractTask

@dataclass
class TrainModelTask(AbstractTask):
    model_instance: Classifier
    model_var_name: str
    sample_var_name: str
    label_var_name: str

    def process(self, data_store):
        """Trains a model and stores it"""
        samples = data_store[self.sample_var_name]
        labels = data_store[self.label_var_name]
        self.model_instance.train(samples, labels)
        data_store[self.model_var_name] = self.model_instance

        print(f"Training the {self.model_var_name} model.")