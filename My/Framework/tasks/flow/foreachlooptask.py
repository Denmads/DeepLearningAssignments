from dataclasses import dataclass
from Framework.task import Task, AbstractTask

@dataclass
class ForEachLoopTask(AbstractTask):
    tasks: list[Task]
    loop_var_name: str
    loop_values: list

    def process(self, data_store):
        for val in self.loop_values:
            data_store[self.loop_var_name] = val
            for task in self.tasks:
                task.process(data_store)
        del data_store[self.loop_var_name]
