from dataclasses import dataclass
import matplotlib.pyplot as plt
from Framework.task import AbstractTask

@dataclass
class VisualizeKnnDistancesTask(AbstractTask):
    model_var_name: str
    val_sample_var_name: str
    save_file: str = None

    def process(self, data_store):
        """Calculates distances and visualizes them"""
        val_samples = data_store[self.val_sample_var_name]
        dists = data_store[self.model_var_name].internal.compute_distances_two_loops(val_samples)

        plt.clf()
        plt.imshow(dists, interpolation="none")
        if self.save_file is None:
            plt.show()
        else:
            plt.savefig(self.create_resource_path(self.save_file), bbox_inches="tight")

