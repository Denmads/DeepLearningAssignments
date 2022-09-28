from dataclasses import dataclass
from Framework.task import AbstractTask

@dataclass
class SetVarTask(AbstractTask):
    var_name: str
    value: any

    def process(self, data_store):
        """Set a single variable in the data store"""
        data_store[self.var_name] = self.value