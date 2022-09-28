from dataclasses import dataclass, field
from Framework.task import AbstractTask

@dataclass
class ModelPredictTask(AbstractTask):
    model_var_name: str
    sample_var_name: str
    result_var_name: str
    predict_args: dict = field(default_factory=dict)

    def process(self, data_store):
        """Trains a model"""
        if "k" not in self.predict_args:
            self.predict_args["k"] = 1
        if "num_loops" not in self.predict_args:
            self.predict_args["num_loops"] = 0

        samples = data_store[self.sample_var_name]
        print(f"Predicting on the {self.sample_var_name} samples using the {self.model_var_name} model.")
        print(f"Using the following params: {self.predict_args}")
        result = data_store[self.model_var_name].predict(samples, **self.predict_args)
        data_store[self.result_var_name] = result
