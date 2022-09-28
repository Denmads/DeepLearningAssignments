import json
from Framework.task import Task
from Framework.tasks.checkpointtask import DataCheckpointTask
import time
import os

class Pipeline:

    def __init__(self, *tasks: Task, **kwargs):
        self.tasks = tasks
        self.data = {
            "output": {}
        }

        self._process_kwargs(kwargs)

        for task in self.tasks:
            task.create_resource_path = self.create_resource_path

        self.checkpoints = filter(lambda t: type(t) == DataCheckpointTask, self.tasks)
        self._setup_checkpoints()
        if not self.from_start:
            self._check_checkpoints()

        self.set_resource_path_func(self.tasks)

    def _process_kwargs(self, kwargs):
        if "output_file" in kwargs:
            self.output_file = kwargs["output_file"]
        else:
            self.output_file = "output.json"

        if "pipeline_path" in kwargs:
            self.pipeline_path = kwargs["pipeline_path"]
        else:
            self.pipeline_path = ""
        
        if "from_start" in kwargs:
            self.from_start = kwargs["from_start"]
        else:
            self.from_start = False

        def join_path(file: str):
            return os.path.join(self.pipeline_path, file)

        self.create_resource_path = join_path

    def _setup_checkpoints(self):
        cnt = 1
        for chp in self.checkpoints:
            chp.checkpoint_num = cnt
            cnt += 1

    def _check_checkpoints(self):
        for i in range(len(self.tasks)-1, 0, -1):
            task = self.tasks[i]
            if type(task) == DataCheckpointTask and task.does_checkpoint_file_exist():
                self.data = task.load_checkpoint()
                self.tasks = self.tasks[i+1:]
                return

    def set_resource_path_func(self, tasks: list):
        for task in tasks:
            task.set_resource_path_func(self.create_resource_path)
            if hasattr(task, "tasks"):
                self.set_resource_path_func(task.tasks)

    def run(self):
        """Process tasks and checkpoints"""

        start = time.perf_counter()
        print(start)

        for task in self.tasks:
            task.process(self.data)
        self.write_output()
        
        diff = time.perf_counter() - start
        print(f"The pipeline took {diff:.2} second(s) to complete.")
    
    def write_output(self):
        if len(self.data["output"].keys()) > 0:
            with open(self.create_resource_path(self.output_file), "w") as f:
                f.write(json.dumps(self.data["output"]))