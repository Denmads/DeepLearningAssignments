from dataclasses import dataclass
from Framework.task import Task, AbstractTask

@dataclass
class ForLoopTask(AbstractTask):
    tasks: list[Task]
    loop_var_name: str
    initial_value: int = 0
    max_value: int = 5 # exclusive
    step_size: int = 1

    def process(self, data_store):
        for i in range(self.initial_value, self.max_value, self.step_size):
            data_store[self.loop_var_name] = i
            for task in self.tasks:
                task.process(data_store)
        del data_store[self.loop_var_name]
