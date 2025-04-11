# PyDART (Research Branch)

Welcome to **PyDART**, a framework for **dynamic programming–based partitioning** and **offline scheduling** of deep learning inference tasks. This **research** branch demonstrates the full workflow:

1. **Init Phase:** Users define tasks (heavy/light), custom models, and input sizes.
2. **Eval Phase:** Runs experiments comparing naive vs. DP-partitioned execution and generates detailed VizTracer traces.
3. **Runtime Phase:** Provides a high-level API for production inference (single, batch, or streaming).

> **Note:** This branch is intended for ongoing research and feature development. It may be unstable or subject to rapid changes. See the `main` branch for stable, production-ready code.

---

## Table of Contents
- [Features](#features)
- [Branch Overview](#branch-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Initialization (Init Phase)](#1-initialization-init-phase)
  - [2. Evaluation (Eval Phase)](#2-evaluation-eval-phase)
  - [3. Runtime (Production) Phase](#3-runtime-production-phase)
- [Examples](#examples)
- [License](#license)

---

## Features

- **DP-Based Partitioning:** Offline scheduling with dynamic programming to split your model’s FX graph into optimal stages.
- **Naive vs. Parallel Execution:** Compare baseline (naive) single-thread scheduling vs. multi-thread parallel execution using `Task`, `Taskset`, and `Evaluator`.
- **Profiling & Tracing:** Automatic instrumentation with PyTorch’s `torch.profiler` and optional VizTracer outputs (`trace_naive.html` & `trace_parallel.html`).
- **High-Level API (Runtime):** Functions like `run_inference`, `run_batch_inference`, and `run_stream_inference` for flexible production inference.

---

## Branch Overview

The **research** branch showcases advanced or experimental features, including:

- **Custom Experiments (`pydart/core/experiment.py`):** 
  - `run_custom_experiment` to set up heavy/light tasks, partition them, capture naive vs. parallel traces, and measure speedup.
  - `run_multiple_custom_experiments` to loop over various heavy/light ratios.
- **New Examples:** Demonstrating a full pipeline from user-defined tasks to final production inference.
- > **Note:** Checkout , Final_POC_Experiments folder for the above full-fledged sample notebook.
- **Scripts:** 
  - `experiment_static.py` for quick built-in model testing,
  - `experiment_runner.py` for an argparse CLI.

---

## Installation

1. **Clone and Check Out Research Branch**
   ```bash
   git clone https://github.com/parthshinde1221/pyDART.git
   cd pyDART
   git checkout research


2. **Check GPU (if any)**:
   ```bash
   nvidia-smi
   ```
   If you see NVIDIA driver info, you likely have a GPU. Otherwise, you’re on CPU.


3. **Install PyTorch**
    GPU Example
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
   ```

   CPU Example
   ```bash
   pip install torch torchvision torchaudio
   ```
