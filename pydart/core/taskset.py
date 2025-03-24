import time
import threading
from pydart.core.task import Task

class Taskset:
    def __init__(self, tasks, available_nodes, metric=None):
        self.tasks = tasks
        self.available_nodes = sorted(available_nodes, key=lambda n: 0 if n.gpu is not None else 1)
        self.metric = metric
        if self.tasks and self.available_nodes:
            sample_tensor = self.tasks[0].input_data
            from pydart.utils.helpers import measure_max_transfer_penalty
            penalty = measure_max_transfer_penalty(self.available_nodes, sample_tensor)
            print(f"Measured max transfer penalty: {penalty:.6f} s")
            if self.metric is not None:
                self.metric.transfer_penalty = penalty
        self.total_utilization = 0.0
        self.average_turnaround_time = 0.0
        self.throughput = 0.0
        self.makespan = 0.0
        self.task_completion_rate = 0.0
        self.loads = {}
        for t in self.tasks:
            if self.metric is not None:
                t.load_metric = self.metric
            self.loads[t.task_id] = t.load_metric.compute_task(t, self) if self.metric else 0.0
            t._available_nodes = self.available_nodes
            t.run_offline_partition_makespan()

    def execute_all(self):
        threads = []
        for t in self.tasks:
            thr = threading.Thread(target=self._execute_task, args=(t,))
            thr.start()
            threads.append(thr)
        for thr in threads:
            thr.join()
        self._calculate_metrics()

    def _execute_task(self, task: Task):
        print(f"[Taskset] Starting Task {task.task_id}")
        task.start_time = time.time()
        try:
            order = task.get_execution_order()
        except ValueError as e:
            print(f"[Taskset] {task.task_id} error: {e}")
            return
        node_outputs = {}
        for sid in order:
            stg = task.stages[sid]
            def run_stage(s=stg):
                s.run_stage(node_outputs)
            rq = stg.assigned_node.assign_task(run_stage)
            rq.get()
        task.finish_time = time.time()
        print(f"[Taskset] Completed Task {task.task_id} in {task.finish_time - task.start_time:.2f} s")

    def _calculate_metrics(self):
        total_busy_time = sum(stg.execution_time for t in self.tasks for stg in t.stages.values() if stg.execution_time)
        earliest_start = min((t.start_time for t in self.tasks if t.start_time), default=None)
        latest_finish = max((t.finish_time for t in self.tasks if t.finish_time), default=None)
        obs = (latest_finish - earliest_start) if earliest_start and latest_finish else 0.0
        used_nodes = {stg.assigned_node.node_id for t in self.tasks for stg in t.stages.values() if stg.assigned_node}
        total_available_time = obs * len(used_nodes)
        self.total_utilization = total_busy_time / total_available_time if total_available_time > 0 else 0.0
        ttimes = [t.get_total_execution_time() for t in self.tasks if t.start_time and t.finish_time]
        self.average_turnaround_time = sum(ttimes) / len(ttimes) if ttimes else 0.0
        self.makespan = obs
        self.throughput = len(self.tasks) / obs if obs > 0 else 0.0
        done = [t for t in self.tasks if t.output_data is not None]
        self.task_completion_rate = len(done) / len(self.tasks) if self.tasks else 0.0

    def __repr__(self):
        return (f"Taskset(num_tasks={len(self.tasks)}, makespan={self.makespan:.4f}s, "
                f"utilization={self.total_utilization:.2%}, throughput={self.throughput:.3f} tasks/s, "
                f"avg_turnaround={self.average_turnaround_time:.4f}s, completion_rate={self.task_completion_rate:.2%})")
