from dataclasses import dataclass
import pickle
import os
from Framework.task import AbstractTask

@dataclass
class DataCheckpointTask(AbstractTask):
    checkpoint_file: str = "checkpoint.chp" # set by the pipeline if it is not a custom checkpoint
    checkpoint_num: int = 1 # set by the pipeline

    def checkpoint_file_path(self):
        return self.create_resource_path(self.checkpoint_file)

    def does_checkpoint_file_exist(self):
        isf = os.path.isfile(self.checkpoint_file_path())
        if isf:
            with open(self.checkpoint_file_path(), "rb") as f:
                data = pickle.load(f)
                return data["checkpoint_num"] == self.checkpoint_num
        else:
            return False

    def load_checkpoint(self) -> dict:
        print(f"Loaded checkpoint #{self.checkpoint_num}")

        with open(self.checkpoint_file_path(), "rb") as f:
            return pickle.load(f)["data"]

    def process(self, data_store):
        print("Saving checkpoint")
        with open(self.checkpoint_file_path(), "wb") as f:
            pickle.dump({"checkpoint_num": self.checkpoint_num, "data": data_store}, f)