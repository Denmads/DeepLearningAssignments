from dataclasses import dataclass
from Framework.task import AbstractTask

@dataclass
class SplitDataTask(AbstractTask):
    input_data_var: str
    sample_var_name: str
    label_var_name: str
    meta_var_name: str

    def process(self, data_store):
        """split full data into samples, labels, metadata"""
        data_store[self.sample_var_name] = data_store[self.input_data_var]["images"]
        data_store[self.label_var_name] = data_store[self.input_data_var]["labels"]
        data_store[self.meta_var_name] = data_store[self.input_data_var]["meta"]
        del data_store[self.input_data_var]

        print(f"split '{self.input_data_var}' into {self.sample_var_name}, {self.label_var_name}, {self.meta_var_name}")
        