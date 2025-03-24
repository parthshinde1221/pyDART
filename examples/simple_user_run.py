#!/usr/bin/env python
"""
Full User Demo:
 - Demonstrates the entire pyDART workflow:
   (1) Init Phase: Define heavy/light tasks & models
   (2) Eval Phase: Run a custom experiment (DP partitioning, naive vs parallel scheduling)
   (3) Runtime Phase: Production inference with run_inference, run_batch_inference, run_stream_inference

Requires:
 - pydart/core/experiment.py for custom experiments
 - pydart/api.py for runtime inference
 - A user "heavy" model, a user "light" model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# 1) INIT PHASE
# ---------------------------
# We define two user-supplied models:
#   - MyHeavyModel for 'heavy' tasks
#   - MyLightModel for 'light' tasks

class MyHeavyModel(nn.Module):
    def __init__(self):
        super(MyHeavyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class MyLightModel(nn.Module):
    def __init__(self):
        super(MyLightModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        # e.g. input shape (B, 1, 28, 28) => flatten to (B, 784)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# We'll define their input shapes:
#  e.g., a 'heavy' input might be (3, 128, 128)
#  a 'light' input might be (1, 28, 28)
HEAVY_INPUT_SIZE = (3, 128, 128)
LIGHT_INPUT_SIZE = (1, 28, 28)

# ---------------------------
# 2) EVAL PHASE (Custom Experiment)
# ---------------------------
# We'll run a single custom experiment with naive vs. parallel scheduling,
# capturing VizTracer traces. We'll rely on the core library's 'run_custom_experiment'.

from pydart.core.experiment import run_custom_experiment

def evaluate_models():
    print("=== EVALUATION PHASE: Running Custom Experiment ===")

    # We'll run a single experiment with a ratio of 3:2 heavy-to-light tasks.
    evaluator = run_custom_experiment(
        heavy_models=[MyHeavyModel],
        light_models=[MyLightModel],
        heavy_input_size=HEAVY_INPUT_SIZE,
        light_input_size=LIGHT_INPUT_SIZE,
        k=2,                   # use 2 compute nodes
        num_tasks=10,          # total tasks
        heavy_light_ratio=(3, 2),
        batch_size=8           # synthetic batch size
        # metric=None => default HPCMakespanMetric
    )
    # This will produce 'trace_naive.html' and 'trace_parallel.html' for scheduling traces,
    # as well as console output comparing naive vs. parallel, speedup, throughput, etc.
    print("Experiment finished.\n")
    return evaluator  # if we want to inspect evaluator details further

# ---------------------------
# 3) RUNTIME PHASE
# ---------------------------
# We demonstrate how to do production inference (single input, batch, streaming) with the high-level API.

from pydart.api import run_inference, run_batch_inference, run_stream_inference

def runtime_inference_demo():
    print("=== RUNTIME PHASE: Demonstrating Production Inference ===")

    # (A) Single Inference
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 5)
        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    input_tensor = torch.randn(16, 10)   # 16 examples, 10 features each
    output_single = run_inference(
        model,
        input_data=input_tensor,
        k=1,
        task_id="single_inference"
    )
    print("Output (single inference):", output_single.shape)

    # (B) Batch Inference
    inputs = torch.randn(50, 10)  # 50 examples
    targets = torch.randint(0, 5, (50,))
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=10)  # 5 batches
    outputs_batch = run_batch_inference(
        model,
        dataloader=dataloader,
        k=1,
        task_id_prefix="batch_inference"
    )
    print("Batch Inference Outputs:")
    for i, out in enumerate(outputs_batch):
        print(f"Batch {i} => shape {out.shape}")

    # (C) Stream Inference
    # We'll create a generator that yields one sample at a time:
    def sample_generator(num_samples=8):
        for _ in range(num_samples):
            yield torch.randn(1, 10)  # shape (1,10)

    print("Stream Inference Outputs:")
    for i, output in enumerate(
        run_stream_inference(model, sample_generator(), k=1, task_id_prefix="stream_inference", stream_batch_size=2)
    ):
        print(f"Stream mini-batch {i}: shape {output.shape}")

def main():
    # Step 1 + 2 => Evaluate user-defined heavy/light tasks
    evaluate_models()

    # Step 3 => Show how to do runtime (production) inference with simpler models.
    runtime_inference_demo()

if __name__ == "__main__":
    main()
