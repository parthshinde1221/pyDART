import time
import torch

class Evaluator:
    def __init__(self, taskset, profiler):
        self.taskset = taskset
        self.profiler = profiler
        self.naive_outputs = {}
        self.parallel_outputs = {}
        self.naive_execution_times = {}
        self.parallel_execution_times = {}
        self.naive_completion_times = {}
        self.parallel_completion_times = {}
        self.naive_makespan = 0.0
        self.parallel_makespan = 0.0
        self.speedup_makespan = 0.0
        self.throughput_makespan = 0.0

    def run_naive_execution(self):
        print("[Evaluator] Starting Naive Execution.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print("Running on cuda")
        else:
            print("Running on cpu")
        start = time.time()
        for task in self.taskset.tasks:
            task.model.to(device)
            inp = task.input_data.to(device)
            t0 = time.time()
            with torch.no_grad():
                out = task.model(inp)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()
            et = t1 - t0
            self.naive_execution_times[task.task_id] = et
            self.naive_completion_times[task.task_id] = time.time() - start
            self.naive_outputs[task.task_id] = out.cpu()
            print(f"[Evaluator] Task {task.task_id}: Naive exec time: {et:.4f}s")
        self.naive_makespan = time.time() - start
        print(f"[Evaluator] Naive makespan: {self.naive_makespan:.4f}s\n")

    def run_parallel_execution(self):
        print("[Evaluator] Starting Parallel Execution.")
        self.parallel_outputs.clear()
        self.parallel_execution_times.clear()
        self.parallel_completion_times.clear()
        par_start = time.time()
        self.taskset.execute_all()
        par_end = time.time()
        self.parallel_makespan = par_end - par_start
        for task in self.taskset.tasks:
            self.parallel_outputs[task.task_id] = task.output_data.cpu() if task.output_data is not None else None
            self.parallel_execution_times[task.task_id] = task.get_total_execution_time()
            if task.finish_time:
                self.parallel_completion_times[task.task_id] = task.finish_time - par_start
            else:
                self.parallel_completion_times[task.task_id] = float('nan')
        print(f"[Evaluator] Parallel makespan: {self.parallel_makespan:.4f}s\n")

    def compare_outputs(self):
        print("[Evaluator] Comparing Outputs.")
        all_match = True
        for tid, naive_out in self.naive_outputs.items():
            par_out = self.parallel_outputs.get(tid)
            if naive_out is None or par_out is None:
                print(f"[Evaluator] Task {tid} missing output.")
                all_match = False
                continue
            if torch.equal(naive_out, par_out) or torch.allclose(naive_out, par_out, atol=1e-5):
                print(f"[Evaluator] Task {tid}: Outputs match.")
            else:
                print(f"[Evaluator] Task {tid}: Outputs do NOT match.")
                all_match = False
        if all_match:
            print("[Evaluator] All outputs match.\n")
        else:
            print("[Evaluator] Some outputs differ.\n")

    def analyze_speedup_throughput(self):
        print("[Evaluator] Analyzing Speedup and Throughput.\n")
        total_naive = sum(self.naive_execution_times.values())
        total_parallel = sum(self.parallel_execution_times.values())
        print("--- Sum-of-times ---")
        print(f"Naive total: {total_naive:.4f}s, Parallel total: {total_parallel:.4f}s")
        speedup_sum = total_naive / total_parallel if total_parallel > 0 else float('inf')
        n = len(self.taskset.tasks)
        print(f"Speedup (sum-of-times): {speedup_sum:.2f}x")
        print(f"Naive Throughput: {n / total_naive:.2f} tasks/s, Parallel Throughput: {n / total_parallel:.2f} tasks/s\n")
        print("--- Makespan ---")
        print(f"Naive makespan: {self.naive_makespan:.4f}s, Parallel makespan: {self.parallel_makespan:.4f}s")
        self.speedup_makespan = self.naive_makespan / self.parallel_makespan if self.parallel_makespan > 0 else float('inf')
        self.throughput_makespan = n / self.parallel_makespan if self.parallel_makespan > 0 else 0.0
        print(f"Speedup (makespan): {self.speedup_makespan:.2f}x")
        print(f"Naive Throughput (makespan): {n / self.naive_makespan:.2f} tasks/s, Parallel Throughput (makespan): {self.throughput_makespan:.2f} tasks/s\n")
        print("--- Task Completion Times ---")
        for tid in self.naive_completion_times:
            nf = self.naive_completion_times[tid]
            pf = self.parallel_completion_times.get(tid, float('nan'))
            print(f"Task {tid}: Naive finish: {nf:.4f}s, Parallel finish: {pf:.4f}s")
        print()

    def __repr__(self):
        return f"Evaluator(Taskset with {len(self.taskset.tasks)} tasks)"
