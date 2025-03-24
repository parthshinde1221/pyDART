# File: pydart/core/experiment.py

import copy
import time
import os
import csv
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Type

from pydart.core.node import Node
from pydart.core.task import Task
from pydart.core.taskset import Taskset
from pydart.core.evaluator import Evaluator
from pydart.core.profiler import Profiler
from pydart.core.metrics import MetricInterface, HPCMakespanMetric
from pydart.utils.helpers import create_dataloader

# For reproducibility
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

# Single-run function with user-provided models, input sizes, etc.
def run_custom_experiment(
    heavy_models: List[Type[torch.nn.Module]],
    light_models: List[Type[torch.nn.Module]],
    heavy_input_size: Tuple[int, ...],
    light_input_size: Tuple[int, ...],
    k: int = 2,
    num_tasks: int = 20,
    heavy_light_ratio: Tuple[int, int] = (3, 2),
    batch_size: int = 10,
    metric: MetricInterface = None
) -> Evaluator:
    """
    Run a single custom experiment using user-defined models and input sizes,
    with VizTracer capturing naive and parallel traces.

    Args:
        heavy_models: List of user-provided heavy model classes.
        light_models: List of user-provided light model classes.
        heavy_input_size: E.g., (3, 224, 224) for heavy tasks.
        light_input_size: E.g., (3, 28, 28) for light tasks.
        k: Number of compute nodes to use.
        num_tasks: Total tasks.
        heavy_light_ratio: (heavy_ratio, light_ratio).
        batch_size: Synthetic data batch size.
        metric: Custom metric or HPCMakespanMetric by default.

    Returns:
        Evaluator object containing results (makespan, speedups, etc.).
    """
    set_seed(42)
    nodes = Node.discover_nodes(disjoint=True)[:k]
    metric = metric or HPCMakespanMetric()
    profiler = Profiler(mode="init")

    # Figure out how many tasks are heavy vs. light.
    heavy_count = (num_tasks * heavy_light_ratio[0]) // sum(heavy_light_ratio)
    light_count = num_tasks - heavy_count

    print(f"[Custom Experiment] Using {len(nodes)} node(s): {nodes}")
    print(f"[Custom Experiment] Creating {heavy_count} heavy tasks and {light_count} light tasks.")

    tasks = []

    # Create heavy tasks
    for i in range(heavy_count):
        model_cls = heavy_models[i % len(heavy_models)]
        # Generate synthetic data for heavy tasks
        dl = create_dataloader(batch_size=batch_size, num_samples=100, input_size=heavy_input_size)
        inp, _ = next(iter(dl))
        model = model_cls()
        task = Task(
            task_id=f"heavy_{i+1}",
            model=model,
            input_data=inp,
            model_name=model.__class__.__name__,
            profiler=profiler,
            load_metric=metric
        )
        tasks.append(task)

    # Create light tasks
    for i in range(light_count):
        model_cls = light_models[i % len(light_models)]
        # Generate synthetic data for light tasks
        dl = create_dataloader(batch_size=batch_size, num_samples=100, input_size=light_input_size)
        inp, _ = next(iter(dl))
        model = model_cls()
        task = Task(
            task_id=f"light_{i+1}",
            model=model,
            input_data=inp,
            model_name=model.__class__.__name__,
            profiler=profiler,
            load_metric=metric
        )
        tasks.append(task)

    # Profile each task on each node (collect per-layer performance data)
    for task in tasks:
        for node in nodes:
            inp_copy = copy.deepcopy(task.input_data)
            profiler.profile_model(model=task.model, input_data=inp_copy, node=node, task_id=task.task_id)
            time.sleep(0.05)

    # Populate the collected profile data
    for task in tasks:
        task.populate_profile_records()

    # Build the Taskset
    taskset = Taskset(tasks, nodes, metric=metric)
    evaluator = Evaluator(taskset, profiler)

    # ------------------ VizTracer Integration ------------------ #
    from viztracer import VizTracer

    # Trace naive execution
    tracer_naive = VizTracer(output_file="trace_naive.html")
    tracer_naive.start()
    print("[Custom Experiment] Running naive execution (traced)...")
    evaluator.run_naive_execution()
    tracer_naive.stop()
    tracer_naive.save()

    # Trace parallel execution
    tracer_parallel = VizTracer(output_file="trace_parallel.html")
    tracer_parallel.start()
    print("[Custom Experiment] Running parallel execution (traced)...")
    evaluator.run_parallel_execution()
    tracer_parallel.stop()
    tracer_parallel.save()

    # Compare outputs and print stats
    evaluator.compare_outputs()
    evaluator.analyze_speedup_throughput()

    # Stop node threads
    for node in nodes:
        node.stop()

    return evaluator

def run_multiple_custom_experiments(
    heavy_models: List[Type[torch.nn.Module]],
    light_models: List[Type[torch.nn.Module]],
    heavy_input_size: Tuple[int, ...],
    light_input_size: Tuple[int, ...],
    k: int = 2,
    num_tasks: int = 20,
    batch_size: int = 10,
    metric: MetricInterface = None,
    ratios=None
):
    """
    Run multiple custom experiments across various heavy/light ratios,
    each experiment captured with naive+parallel traces.

    Args:
        heavy_models: List of user-supplied heavy model classes.
        light_models: List of user-supplied light model classes.
        heavy_input_size: E.g., (3,224,224) for heavy tasks.
        light_input_size: E.g., (3,28,28) for light tasks.
        k: Number of compute nodes.
        num_tasks: Number of total tasks in each experiment.
        batch_size: Synthetic batch size for data generation.
        metric: Optional custom metric (HPCMakespanMetric default).
        ratios: Either numeric ratios (e.g., [0.1, 0.2, ...]) or list of pairs (e.g., [(3,2), (1,4)]) or None.

    Returns:
        None, but prints results + writes CSV. Each experiment has its own naive+parallel traces.
    """
    set_seed(42)
    metric = metric or HPCMakespanMetric()

    if ratios is None:
        ratios = [round(i/10, 1) for i in range(1, 10)]

    # Check if they're numeric or pair-based
    numeric_ratios = []
    pairs_list = []
    is_all_pairs = all(isinstance(r, tuple) and len(r) == 2 for r in ratios)
    is_all_numbers = all(isinstance(r, (float, int)) for r in ratios)

    if is_all_pairs:
        pairs_list = ratios
    elif is_all_numbers:
        numeric_ratios = ratios
    else:
        print("[Warning] The 'ratios' parameter is mixed or invalid. Using default numeric approach.")
        numeric_ratios = [round(i/10, 1) for i in range(1, 10)]

    results = []
    csv_filename = "custom_experiment_results.csv"

    speedups = []
    parallel_throughputs = []
    naive_throughputs = []
    makespans = []

    # Helper function to run each ratio & record data
    def run_one_experiment(heavy_count, light_count, ratio_label):
        # run_custom_experiment calls naive & parallel, capturing separate traces each time
        evaluator = run_custom_experiment(
            heavy_models=heavy_models,
            light_models=light_models,
            heavy_input_size=heavy_input_size,
            light_input_size=light_input_size,
            k=k,
            num_tasks=(heavy_count + light_count),
            heavy_light_ratio=(heavy_count, light_count),
            batch_size=batch_size,
            metric=metric
        )
        sp = evaluator.speedup_makespan
        pt = evaluator.throughput_makespan
        naive_thr = ((heavy_count + light_count) / evaluator.naive_makespan) if evaluator.naive_makespan > 0 else 0
        mk = evaluator.parallel_makespan

        speedups.append(sp)
        parallel_throughputs.append(pt)
        naive_throughputs.append(naive_thr)
        makespans.append(mk)
        results.append((ratio_label, sp, pt, naive_thr, mk))

    # Numeric approach
    if numeric_ratios:
        for ratio in numeric_ratios:
            heavy_count = int(num_tasks * ratio)
            light_count = num_tasks - heavy_count
            ratio_label = ratio
            print(f'[Multiple Custom Exp] ratio={ratio} => heavy_count={heavy_count}, light_count={light_count}')
            run_one_experiment(heavy_count, light_count, ratio_label)

    # Pair-based approach
    if pairs_list:
        for pair in pairs_list:
            total = pair[0] + pair[1]
            heavy_count = (num_tasks * pair[0]) // total
            light_count = num_tasks - heavy_count
            ratio_label = pair
            print(f'[Multiple Custom Exp] ratio pair={pair} => heavy_count={heavy_count}, light_count={light_count}')
            run_one_experiment(heavy_count, light_count, ratio_label)

    # Write to CSV
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Ratio", "Speedup", "Parallel Throughput", "Naive Throughput", "Makespan"])
        for row in results:
            writer.writerow(row)

    print(f'[Multiple Custom Exp] Results stored in {csv_filename}')
    print('[Multiple Custom Exp] Aggregated Results =>')
    print('Speedups:', speedups)
    print('Parallel Throughputs:', parallel_throughputs)
    print('Naive Throughputs:', naive_throughputs)
    print('Makespans:', makespans)

    # Quick plots if numeric
    if numeric_ratios and all(isinstance(x, float) for x in numeric_ratios):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].plot(numeric_ratios, speedups, marker='o', linestyle='-', label='Speedup')
        axes[0].set_xlabel("Heavy Task Ratio")
        axes[0].set_ylabel("Speedup (makespan)")
        axes[0].set_title("Speedup vs. Ratio")
        axes[0].legend()

        axes[1].plot(numeric_ratios, parallel_throughputs, marker='s', linestyle='-', label='Parallel Throughput')
        axes[1].plot(numeric_ratios, naive_throughputs, marker='^', linestyle='--', label='Naive Throughput', color='orange')
        axes[1].set_xlabel("Heavy Task Ratio")
        axes[1].set_ylabel("Throughput (tasks/s)")
        axes[1].set_title("Throughput Comparison")
        axes[1].legend()

        axes[2].plot(numeric_ratios, makespans, marker='d', linestyle='-', label='Makespan')
        axes[2].set_xlabel("Heavy Task Ratio")
        axes[2].set_ylabel("Makespan (s)")
        axes[2].set_title("Makespan vs. Ratio")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig('custom_experiment_plots.png')
        print('[Multiple Custom Exp] Plots saved to custom_experiment_plots.png')
        plt.show()
    else:
        print('[Multiple Custom Exp] Skipping plot generation because ratio approach is pairs or unknown.')
