from dataclasses import dataclass
from Framework.task import AbstractTask

@dataclass
class AddToOutputTask(AbstractTask):
    output_var_name: str
    data_store_var_name: str

    def process(self, data_store):
        """Set a single variable in the data store"""
        data_store["output"][self.output_var_name] = data_store[self.data_store_var_name]