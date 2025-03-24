"""
User-friendly API for running inference using the pyDART framework.

Provides:
  - run_inference: Run inference on a single input.
  - run_batch_inference: Run inference on an entire dataset.
  - run_stream_inference: Process streaming data in mini-batches.
"""

from pydart.core.node import Node
from pydart.core.task import Task
from pydart.core.taskset import Taskset
from pydart.core.metrics import HPCMakespanMetric
from pydart.core.profiler import Profiler
from pydart.utils.helpers import create_dataloader
import torch

def run_inference(model, input_data=None, *, k=1, task_id="task", 
                  batch_size=10, num_samples=100, input_size=None):
    """
    Run inference on a single input tensor.
    """
    nodes = Node.discover_nodes(disjoint=True)[:k]
    profiler = Profiler(mode="init")
    if input_data is None:
        if input_size is None:
            raise ValueError("Either input_data or input_size must be provided.")
        dataloader = create_dataloader(batch_size=batch_size, num_samples=num_samples, input_size=input_size)
        input_data, _ = next(iter(dataloader))
    task = Task(
        task_id=task_id,
        model=model,
        input_data=input_data,
        model_name=model.__class__.__name__,
        profiler=profiler,
        load_metric=HPCMakespanMetric()
    )
    task._available_nodes = nodes
    task.run_offline_partition_makespan()
    task.populate_profile_records()
    taskset = Taskset([task], nodes, metric=HPCMakespanMetric())
    taskset.execute_all()
    for node in nodes:
        node.stop()
    return task.output_data

def run_batch_inference(model, dataloader, *, k=1, task_id_prefix="batch_task"):
    """
    Run inference on a dataset (DataLoader) by creating one Task per batch.
    """
    nodes = Node.discover_nodes(disjoint=True)[:k]
    profiler = Profiler(mode="init")
    tasks = []
    for i, (input_data, _) in enumerate(dataloader):
        task = Task(
            task_id=f"{task_id_prefix}_{i}",
            model=model,
            input_data=input_data,
            model_name=model.__class__.__name__,
            profiler=profiler,
            load_metric=HPCMakespanMetric()
        )
        task._available_nodes = nodes
        task.run_offline_partition_makespan()
        task.populate_profile_records()
        tasks.append(task)
    taskset = Taskset(tasks, nodes, metric=HPCMakespanMetric())
    taskset.execute_all()
    outputs = [t.output_data for t in tasks]
    for node in nodes:
        node.stop()
    return outputs

def run_stream_inference(model, data_generator, *, k=1, task_id_prefix="stream_task", stream_batch_size=1):
    """
    Run inference on streaming data in mini-batches.
    """
    nodes = Node.discover_nodes(disjoint=True)[:k]
    profiler = Profiler(mode="init")
    batch_tasks = []
    batch_index = 0
    for input_data in data_generator:
        task = Task(
            task_id=f"{task_id_prefix}_{batch_index}",
            model=model,
            input_data=input_data,
            model_name=model.__class__.__name__,
            profiler=profiler,
            load_metric=HPCMakespanMetric()
        )
        task._available_nodes = nodes
        task.run_offline_partition_makespan()
        task.populate_profile_records()
        batch_tasks.append(task)
        batch_index += 1
        if len(batch_tasks) == stream_batch_size:
            taskset = Taskset(batch_tasks, nodes, metric=HPCMakespanMetric())
            taskset.execute_all()
            for t in batch_tasks:
                yield t.output_data
            batch_tasks = []
    if batch_tasks:
        taskset = Taskset(batch_tasks, nodes, metric=HPCMakespanMetric())
        taskset.execute_all()
        for t in batch_tasks:
            yield t.output_data
    for node in nodes:
        node.stop()
