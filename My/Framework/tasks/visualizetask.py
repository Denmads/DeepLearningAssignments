from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from Framework.task import AbstractTask

@dataclass
class VisualizeSamplesTask(AbstractTask):
    meta_var_name: str
    sample_var_name: str
    classes: dict[str, str]
    num_samples_per_class: int
    save_file: str = None

    def process(self, data_store):
        """Visualize n samples frome each class"""
        num_classes = len(self.classes.keys())

        plt.clf()
        for y, cls in enumerate(self.classes.keys()):
            idxs = np.where(data_store[self.meta_var_name][:] == cls)[0]
            idxs = np.random.choice(idxs, self.num_samples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(self.num_samples_per_class, num_classes, plt_idx)
                plt.imshow(data_store[self.sample_var_name][idx].astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        if self.save_file == None:
            plt.show()
        else:
            plt.savefig(self.create_resource_path(self.save_file), bbox_inches='tight')