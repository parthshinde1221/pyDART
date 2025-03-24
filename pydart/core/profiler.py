import os
import io
import time
import copy
import torch
import torch.nn as nn
import torch.fx as fx
import pandas as pd
import logging
from pydart.core.metrics import HPCMakespanMetric

class Profiler:
    def __init__(self, mode: str, profile_db_path='profiling_results.csv', log_dir='logs'):
        assert mode in ['init', 'runtime'], "Mode must be 'init' or 'runtime'"
        self.mode = mode
        self.profile_db_path = profile_db_path
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.columns = ['Task_ID', 'Model', 'Layer', 'Compute',
                        'Self CPU (us)', 'CPU Total (us)', 'CUDA Total (us)',
                        'Self CPU Mem (bytes)', 'Self CUDA Mem (bytes)',
                        'Total Execution Time (us)', 'Total Memory Used (bytes)']
        if os.path.exists(self.profile_db_path):
            self.profile_db = pd.read_csv(self.profile_db_path)
        else:
            self.profile_db = pd.DataFrame(columns=self.columns)
        self.runtime_csv = os.path.join(self.log_dir, 'runtime_results.csv')
        if not os.path.exists(self.runtime_csv):
            pd.DataFrame(columns=['Task_ID', 'Model', 'Layer', 'Compute', 'Execution Time (us)']).to_csv(self.runtime_csv, index=False)
        self.observation_window = 0.0
        self.profile_cache = {}

    def profile_model(self, model: nn.Module, input_data, node, task_id: str, warmup_iters=3, profile_iters=5):
        cache_key = (model.__class__.__name__, node.node_id)
        if cache_key in self.profile_cache:
            cached_data = self.profile_cache[cache_key].copy()
            cached_data['Task_ID'] = task_id
            self.profile_db = pd.concat([self.profile_db, cached_data], ignore_index=True)
            print(f"[Profiler] Reused cached profiling data for {model.__class__.__name__} on {node.node_id}.")
            return
        old_affinity = os.sched_getaffinity(0)
        try:
            if node.cpus:
                os.sched_setaffinity(0, node.cpus)
            if node.gpu is not None and torch.cuda.is_available():
                torch.cuda.set_device(node.gpu)
                device = torch.device(f"cuda:{node.gpu}")
            else:
                device = torch.device("cpu")
            model_copy = self._clone_model_safely(model)
            instrumented_model = self._trace_and_instrument_model(model_copy)
            instrumented_model.to(device)
            instrumented_model.eval()
            with torch.no_grad():
                for _ in range(warmup_iters):
                    _ = instrumented_model(input_data.to(device))
            print(f"[Profiler] Starting profiling for Task '{task_id}' on {node.node_id} (device={device}).")
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=profile_iters),
                on_trace_ready=lambda prof: self._trace_handler(prof, task_id, model.__class__.__name__, node.node_id),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                for _ in range(profile_iters):
                    with torch.no_grad():
                        _ = instrumented_model(input_data.to(device))
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    prof.step()
            time.sleep(0.0001)
            new_rows = self.profile_db[self.profile_db['Task_ID'] == task_id].copy()
            self.profile_cache[cache_key] = new_rows
            self.profile_db.to_csv(self.profile_db_path, index=False)
            print(f"[Profiler] Profiling complete. Data saved to {self.profile_db_path}.")
        finally:
            os.sched_setaffinity(0, old_affinity)

    def _clone_model_safely(self, model: nn.Module) -> nn.Module:
        try:
            return copy.deepcopy(model)
        except Exception as e:
            print(f"[Profiler] deepcopy failed: {e}. Falling back to torch.save/load.")
            buffer = io.BytesIO()
            torch.save(model, buffer)
            buffer.seek(0)
            return torch.load(buffer)

    def _trace_and_instrument_model(self, model: nn.Module) -> fx.GraphModule:
        tracer = fx.Tracer()
        graph = tracer.trace(model)
        graph_module = fx.GraphModule(model, graph)
        profiler_attr_prefix = "_profiler_wrapped_"
        for node in list(graph.nodes):
            node_name = node.name
            if node.op == 'call_function':
                func = node.target
                wrapped_func_name = f"{profiler_attr_prefix}{node_name}_{id(func)}"
                def make_wrapped_func(original_func, profile_name):
                    def wrapped(*args, **kwargs):
                        with torch.profiler.record_function(profile_name):
                            return original_func(*args, **kwargs)
                    return wrapped
                wrapped_func = make_wrapped_func(func, node_name)
                setattr(model, wrapped_func_name, wrapped_func)
                node.target = getattr(model, wrapped_func_name)
            elif node.op == 'call_module':
                submodule = dict(model.named_modules())[node.target]
                func = submodule.forward
                wrapped_func_name = f"{profiler_attr_prefix}{node_name}_{id(func)}"
                def make_wrapped_forward(original_forward, profile_name):
                    def wrapped_forward(*args, **kwargs):
                        with torch.profiler.record_function(profile_name):
                            return original_forward(*args, **kwargs)
                    return wrapped_forward
                wrapped_forward = make_wrapped_forward(func, node_name)
                setattr(submodule, wrapped_func_name, wrapped_forward)
                submodule.forward = getattr(submodule, wrapped_func_name)
            elif node.op == 'call_method':
                method_name = node.target
                obj = node.args[0]
                original_method = getattr(obj, method_name, None)
                if original_method is None:
                    continue
                wrapped_method_name = f"{profiler_attr_prefix}{node_name}_{id(original_method)}"
                def make_wrapped_method(orig_meth, profile_name):
                    def wrapped_method(*args, **kwargs):
                        with torch.profiler.record_function(profile_name):
                            return orig_meth(*args, **kwargs)
                    return wrapped_method
                wrapped_method = make_wrapped_method(original_method, node_name)
                setattr(obj, wrapped_method_name, wrapped_method)
            # Placeholders and outputs remain unchanged.
        graph_module.recompile()
        return graph_module

    def _trace_handler(self, prof, task_id: str, model_name: str, node_id: str):
        self._process_profiler_data(prof, task_id, model_name, node_id)

    def _process_profiler_data(self, profiler, task_id: str, model_name: str, node_id: str):
        aggregated = {}
        forward_pass = {
            'Task_ID': task_id,
            'Model': model_name,
            'Layer': 'forward_pass',
            'Compute': node_id,
            'Self CPU (us)': 0.0,
            'CPU Total (us)': 0.0,
            'CUDA Total (us)': 0.0,
            'Self CPU Mem (bytes)': 0,
            'Self CUDA Mem (bytes)': 0,
            'Total Execution Time (us)': 0.0,
            'Total Memory Used (bytes)': 0
        }
        events = profiler.key_averages()
        for evt in events:
            layer_name = evt.key
            if layer_name.startswith("aten::"):
                continue
            forward_pass['Self CPU (us)'] += evt.self_cpu_time_total
            forward_pass['CPU Total (us)'] += evt.cpu_time_total
            forward_pass['CUDA Total (us)'] += getattr(evt, 'cuda_time_total', 0.0)
            forward_pass['Self CPU Mem (bytes)'] += getattr(evt, 'self_cpu_memory_usage', 0)
            forward_pass['Self CUDA Mem (bytes)'] += getattr(evt, 'self_cuda_memory_usage', 0)
            forward_pass['Total Execution Time (us)'] += evt.cpu_time_total + getattr(evt, 'cuda_time_total', 0.0)
            forward_pass['Total Memory Used (bytes)'] += getattr(evt, 'self_cpu_memory_usage', 0) + getattr(evt, 'self_cuda_memory_usage', 0)
            if layer_name not in aggregated:
                aggregated[layer_name] = {
                    'Task_ID': task_id,
                    'Model': model_name,
                    'Layer': layer_name,
                    'Compute': node_id,
                    'Self CPU (us)': 0.0,
                    'CPU Total (us)': 0.0,
                    'CUDA Total (us)': 0.0,
                    'Self CPU Mem (bytes)': 0,
                    'Self CUDA Mem (bytes)': 0,
                    'Total Execution Time (us)': 0.0,
                    'Total Memory Used (bytes)': 0
                }
            aggregated[layer_name]['Self CPU (us)'] += evt.self_cpu_time_total
            aggregated[layer_name]['CPU Total (us)'] += evt.cpu_time_total
            aggregated[layer_name]['CUDA Total (us)'] += getattr(evt, 'cuda_time_total', 0.0)
            aggregated[layer_name]['Self CPU Mem (bytes)'] += getattr(evt, 'self_cpu_memory_usage', 0)
            aggregated[layer_name]['Self CUDA Mem (bytes)'] += getattr(evt, 'self_cuda_memory_usage', 0)
            aggregated[layer_name]['Total Execution Time (us)'] += evt.cpu_time_total + getattr(evt, 'cuda_time_total', 0.0)
            aggregated[layer_name]['Total Memory Used (bytes)'] += getattr(evt, 'self_cpu_memory_usage', 0) + getattr(evt, 'self_cuda_memory_usage', 0)
        self.profile_db = self._upsert(self.profile_db, forward_pass)
        for data in aggregated.values():
            self.profile_db = self._upsert(self.profile_db, data)
        self.profile_db.to_csv(self.profile_db_path, index=False)

    def _upsert(self, df: pd.DataFrame, row: dict) -> pd.DataFrame:
        mask = (df['Task_ID'] == row['Task_ID']) & (df['Model'] == row['Model']) & (df['Layer'] == row['Layer']) & (df['Compute'] == row['Compute'])
        if mask.any():
            existing_time = df.loc[mask, 'Total Execution Time (us)'].max()
            if row['Total Execution Time (us)'] > existing_time:
                for key in self.columns:
                    df.loc[mask, key] = row[key]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        return df

    def get_profile_db(self):
        return self.profile_db

    def print_profile_db(self):
        if self.profile_db.empty:
            print("ProfileDB is empty.")
        else:
            print("ProfileDB:")
            print(self.profile_db.to_string(index=False))
