#!/usr/bin/env python
"""
Command-line Experiment Runner for the pyDART Framework

This script provides a CLI interface to run evaluation experiments.
It accepts user parameters (such as the number of compute nodes, number of tasks,
and heavy/light ratio) and calls the experiment functions defined in experiment.py.
"""

import argparse
from scripts.experiment_static import run_experiment, run_multiple_experiments

def main():
    parser = argparse.ArgumentParser(description="Run experiments for the pyDART framework.")
    parser.add_argument("--mode", type=str, choices=["single", "multiple"], default="multiple",
                        help="Experiment mode: 'single' for one experiment, 'multiple' for a series.")
    parser.add_argument("--k", type=int, default=2, help="Number of compute nodes to use.")
    parser.add_argument("--num_tasks", type=int, default=40, help="Number of tasks per experiment.")
    parser.add_argument("--heavy_ratio", type=int, default=3,
                        help="Heavy tasks ratio component for single experiment.")
    parser.add_argument("--light_ratio", type=int, default=2,
                        help="Light tasks ratio component for single experiment.")
    parser.add_argument("--experiment_name", type=str, default="default_experiment",
                        help="Base name for experiment output files.")

    args = parser.parse_args()

    if args.mode == "single":
        print("Running a single experiment...")
        heavy_light_ratio = (args.heavy_ratio, args.light_ratio)
        run_experiment(k=args.k, heavy_light_ratio=heavy_light_ratio, num_tasks=args.num_tasks)
    else:
        print("Running multiple experiments...")
        run_multiple_experiments(k=args.k, num_tasks=args.num_tasks)

if __name__ == "__main__":
    main()
