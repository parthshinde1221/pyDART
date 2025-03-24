from abc import ABC, abstractmethod

class MetricInterface(ABC):
    @abstractmethod
    def compute_task(self, task, taskset) -> float:
        """
        Compute overall cost for the task.
        """
        pass

    @abstractmethod
    def compute_layer(self, task, node, fx_node) -> float:
        """
        Compute cost for a single layer.
        """
        pass

class HPCMakespanMetric(MetricInterface):
    def __init__(self):
        self.transfer_penalty = 0.0

    def compute_task(self, task, taskset) -> float:
        return float(task.get_forward_pass_time(sum_across_compute=True))

    def compute_layer(self, task, node, fx_node) -> float:
        key = (node.node_id, fx_node.name)
        record = task.prof_records.get(key, None)
        if record is None:
            return 0.0
        return float(record.get("Total Execution Time (us)", 0.0))
