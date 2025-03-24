#!/usr/bin/env python
"""
Lower-level Experiment Functions for the pyDART Framework.

This module provides functions to:
  - run_experiment: Run a single experiment given a heavy/light ratio and number of tasks.
  - run_multiple_experiments: Run a series of experiments over a range of heavy/light ratios,
    generating CSV and plot outputs.
    
These functions set up the tasks, profile them, execute naive and partitioned runs,
and generate performance data.
"""

import copy
import time
import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from torchvision import models
from pydart.core.node import Node
from pydart.core.profiler import Profiler
from pydart.core.metrics import HPCMakespanMetric
from pydart.core.task import Task
from pydart.core.taskset import Taskset
from pydart.core.evaluator import Evaluator
from pydart.utils.helpers import create_dataloader

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

# Example test models
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 16, 5, padding=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 28 * 28, 10)
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x = torch.cat((x1, x2), dim=1)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class PretrainedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(PretrainedResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        n = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n, num_classes)
    def forward(self, x):
        return self.resnet18(x)

def run_experiment(k: int, heavy_light_ratio: tuple, num_tasks: int = 10):
    set_seed(42)
    all_nodes = Node.discover_nodes(disjoint=True)
    nodes = all_nodes[:k]
    print(f"Using {len(nodes)} nodes: {nodes}")

    profiler = Profiler(mode="init")
    heavy_models = [PretrainedResNet18]
    light_models = [SimpleCNN]
    heavy_count = (num_tasks * heavy_light_ratio[0]) // (heavy_light_ratio[0] + heavy_light_ratio[1])
    light_count = num_tasks - heavy_count
    print(f"Creating {heavy_count} heavy tasks and {light_count} light tasks.")
    tasks = []

    # Create heavy tasks
    for i in range(heavy_count):
        model = heavy_models[i % len(heavy_models)]()
        dl = create_dataloader(batch_size=10, num_samples=100, input_size=(3, 224, 224))
        inp, _ = next(iter(dl))
        task = Task(
            task_id=f"heavy_{i+1}",
            model=model,
            input_data=inp,
            model_name=model.__class__.__name__,
            profiler=profiler,
            load_metric=HPCMakespanMetric()
        )
        tasks.append(task)

    # Create light tasks
    for i in range(light_count):
        model = light_models[i % len(light_models)]()
        dl = create_dataloader(batch_size=10, num_samples=100, input_size=(3, 28, 28))
        inp, _ = next(iter(dl))
        task = Task(
            task_id=f"light_{i+1}",
            model=model,
            input_data=inp,
            model_name=model.__class__.__name__,
            profiler=profiler,
            load_metric=HPCMakespanMetric()
        )
        tasks.append(task)

    # Profile each task on each available node.
    for task in tasks:
        for node in nodes:
            inp_copy = copy.deepcopy(task.input_data)
            profiler.profile_model(model=task.model, input_data=inp_copy, node=node, task_id=task.task_id)
            time.sleep(0.05)
    for task in tasks:
        task.populate_profile_records()

    taskset = Taskset(tasks, nodes, metric=HPCMakespanMetric())
    for task in taskset.tasks:
        task.print_stage_allocations()

    evaluator = Evaluator(taskset, profiler)
    evaluator.run_naive_execution()
    evaluator.run_parallel_execution()
    evaluator.compare_outputs()
    evaluator.analyze_speedup_throughput()

    for node in nodes:
        node.stop()
    return evaluator

def run_multiple_experiments(k: int, num_tasks: int = 15):
    ratios = [round(i / 10, 1) for i in range(1, 10)]
    speedups = []
    parallel_throughputs = []
    naive_throughputs = []
    makespans = []
    config = defaultdict(dict)
    csv_filename = "experiment_results.csv"
    pdf_filename = "experiment_plots.pdf"
    profiling_file = "profiling_results.csv"

    for ratio in ratios:
        heavy_count = int(num_tasks * ratio)
        light_count = num_tasks - heavy_count
        print(f"Running experiment with {heavy_count} heavy and {light_count} light tasks.")
        if os.path.exists(profiling_file):
            os.remove(profiling_file)
            print("Removed existing profiling file.")
        evaluator = run_experiment(k=k, heavy_light_ratio=(heavy_count, light_count), num_tasks=num_tasks)
        sp = evaluator.speedup_makespan
        par_th = evaluator.throughput_makespan
        naive_thr = num_tasks / evaluator.naive_makespan if evaluator.naive_makespan > 0 else 0
        mk = evaluator.parallel_makespan
        speedups.append(sp)
        parallel_throughputs.append(par_th)
        naive_throughputs.append(naive_thr)
        makespans.append(mk)
        config[ratio] = {
            'speedup': sp,
            'parallel_throughput': par_th,
            'naive_throughput': naive_thr,
            'makespan': mk
        }
        print(f"Speedup for ratio {ratio}: {sp:.2f}")
        print(f"Parallel Throughput for ratio {ratio}: {par_th:.2f}")
        print(f"Naive Throughput for ratio {ratio}: {naive_thr:.2f}")
        print(f"Makespan for ratio {ratio}: {mk:.2f}\n")
    print("Final Results:")
    print("Speedups:", speedups)
    print("Parallel Throughputs:", parallel_throughputs)
    print("Naive Throughputs:", naive_throughputs)
    print("Makespans:", makespans)
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Heavy Task Ratio", "Speedup", "Parallel Throughput", "Naive Throughput", "Makespan"])
        for i, ratio in enumerate(ratios):
            writer.writerow([ratio, speedups[i], parallel_throughputs[i], naive_throughputs[i], makespans[i]])
    print(f"Experiment results saved to {csv_filename}")
    max_ratio = max(config, key=lambda r: config[r]['speedup'])
    best_config = config[max_ratio]
    print(f"Best configuration: {best_config}")
    print(f"Max Speedup Ratio: {max_ratio}\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(ratios, speedups, marker='o', linestyle='-', label='Speedup')
    axes[0].set_xlabel("Heavy Task Ratio")
    axes[0].set_ylabel("Speedup (makespan)")
    axes[0].set_title("Speedup vs Heavy Task Ratio")
    axes[0].legend()
    axes[1].plot(ratios, parallel_throughputs, marker='s', linestyle='-', label='Parallel Throughput')
    axes[1].plot(ratios, naive_throughputs, marker='^', linestyle='--', label='Naive Throughput', color='orange')
    axes[1].set_xlabel("Heavy Task Ratio")
    axes[1].set_ylabel("Throughput (tasks/s)")
    axes[1].set_title("Throughput Comparison")
    axes[1].legend()
    axes[2].plot(ratios, makespans, marker='d', linestyle='-', label='Makespan')
    axes[2].set_xlabel("Heavy Task Ratio")
    axes[2].set_ylabel("Makespan (s)")
    axes[2].set_title("Makespan vs Heavy Task Ratio")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(pdf_filename)
    print(f"Plots saved to {pdf_filename}")
    plt.show()
