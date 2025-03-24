import time
import torch
import torch.fx as fx
import networkx as nx
from typing import List, Dict, Optional, Set, Tuple, Any
from pydart.core.metrics import HPCMakespanMetric
from pydart.utils.helpers import resolve_arg, move_tensor_to_device

class Task:
    def __init__(self, task_id: str, model, input_data: torch.Tensor, model_name: str, profiler, load_metric: Optional[HPCMakespanMetric] = None):
        self.task_id = task_id
        self.model = model
        self.input_data = input_data
        self.model_name = model_name
        self.profiler = profiler
        self.load_metric = load_metric if load_metric else HPCMakespanMetric()
        self.stages: Dict[str, Stage] = {}
        self.graph = nx.DiGraph()
        self.start_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        self.output_data: Optional[torch.Tensor] = None
        self.busy_time: float = 0.0
        self.computation_time: float = 0.0
        self.transfer_time: float = 0.0
        self.prof_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._available_nodes: List = []
        self.init_traced_graph: List[str] = []
        self.placeholder_names: Set[str] = set()
        self._initialize_dag()

    def _initialize_dag(self):
        tracer = fx.symbolic_trace(self.model)
        fx_nodes = list(tracer.graph.nodes)
        self.init_traced_graph = [node.name for node in fx_nodes]
        self.placeholder_names = set(n.name for n in fx_nodes if n.op == "placeholder")

    def get_forward_pass_time(self, sum_across_compute: bool = False) -> float:
        if not self.profiler:
            return 0.0
        df = self.profiler.get_profile_db()
        mask = (df['Task_ID'] == self.task_id) & (df['Model'] == self.model_name) & (df['Layer'] == 'forward_pass')
        matched = df.loc[mask]
        if matched.empty:
            return 0.0
        times = matched['Total Execution Time (us)']
        return float(times.sum()) if sum_across_compute else float(times.max())

    def populate_profile_records(self):
        if not self.profiler:
            return
        df = self.profiler.get_profile_db()
        mask_all = (df['Task_ID'] == self.task_id) & (df['Model'] == self.model_name)
        relevant = df.loc[mask_all]
        all_computes = relevant['Compute'].unique()
        for comp in all_computes:
            mask_comp = (relevant['Compute'] == comp)
            subdf = relevant.loc[mask_comp]
            for layer in self.init_traced_graph:
                row = subdf.loc[subdf['Layer'] == layer]
                if not row.empty:
                    self.prof_records[(comp, layer)] = row.iloc[0].to_dict()
                else:
                    self.prof_records[(comp, layer)] = None

    def run_offline_partition_makespan(self):
        tracer = fx.symbolic_trace(self.model)
        fx_nodes = list(tracer.graph.nodes)
        L = len(fx_nodes)
        K = len(self._available_nodes)
        if L == 0 or K == 0:
            return
        cost = [[0.0 for _ in range(L)] for _ in range(L)]
        for i in range(L):
            for j in range(i, L):
                block_costs = []
                for node in self._available_nodes:
                    s = 0.0
                    for l in range(i, j + 1):
                        s += self.load_metric.compute_layer(self, node, fx_nodes[l])
                    block_costs.append(s)
                cost[i][j] = min(block_costs)
        dp = [[float("inf")] * (K + 1) for _ in range(L)]
        split = [[-1] * (K + 1) for _ in range(L)]
        for i in range(L):
            dp[i][1] = cost[0][i]
        for k in range(2, K + 1):
            for i in range(L):
                for x in range(0, i):
                    candidate = max(dp[x][k - 1], cost[x + 1][i])
                    if candidate < dp[i][k]:
                        dp[i][k] = candidate
                        split[i][k] = x
        partitions = []
        k_val = K
        i_val = L - 1
        while k_val > 1:
            x = split[i_val][k_val]
            partitions.append((x + 1, i_val))
            i_val = x
            k_val -= 1
        partitions.append((0, i_val))
        partitions.reverse()
        self.stages.clear()
        self.graph.clear()
        for idx, (start, end) in enumerate(partitions):
            stage_id = f"{self.task_id}-stage-{idx + 1}"
            best_node = None
            best_cost = float("inf")
            for node in self._available_nodes:
                s = 0.0
                for l in range(start, end + 1):
                    s += self.load_metric.compute_layer(self, node, fx_nodes[l])
                if s < best_cost:
                    best_cost = s
                    best_node = node
            stg = Stage(stage_id, fx_nodes[start:end + 1], best_node, self)
            stg.execution_time = best_cost
            self.stages[stage_id] = stg
            self.graph.add_node(stage_id, stage=stg)
        stage_ids = sorted(self.stages.keys(), key=lambda sid: int(sid.split("-")[-1]))
        for i in range(len(stage_ids) - 1):
            self.stages[stage_ids[i + 1]].add_dependency(stage_ids[i])
            self.stages[stage_ids[i]].add_dependent(stage_ids[i + 1])
            self.graph.add_edge(stage_ids[i], stage_ids[i + 1])

    def get_execution_order(self) -> List[str]:
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Cycle in stage DAG")

    def update_busy_time(self, exec_time: float, transfer_time: float = 0.0):
        self.busy_time += exec_time
        self.transfer_time += transfer_time
        self.computation_time += (exec_time - transfer_time)

    def set_output_data(self, output: torch.Tensor):
        self.output_data = output.cpu() if output is not None else None
        self.finish_time = time.time()

    def get_total_execution_time(self) -> float:
        if self.start_time and self.finish_time:
            return self.finish_time - self.start_time
        return 0.0

    def print_stage_allocations(self):
        print(f"=== Stage Allocations for Task {self.task_id} ===")
        for sid, stg in self.stages.items():
            layer_names = [fxn.name for fxn in stg.nodes]
            node_id = stg.assigned_node.node_id if stg.assigned_node else "Unassigned"
            print(f"{sid}: Node={node_id}, Layers={layer_names}, Deps={stg.dependencies}")

    def __repr__(self):
        return f"Task({self.task_id}, model={self.model_name})"

class Stage:
    def __init__(self, stage_id: str, nodes: List[fx.Node], assigned_node, task: Task):
        self.stage_id = stage_id
        self.nodes = nodes
        self.assigned_node = assigned_node
        self.dependencies = []
        self.dependents = []
        self.execution_time: Optional[float] = None
        self.transfer_time: float = 0.0
        self.output_data: Optional[torch.Tensor] = None
        self.task = task
        self.stage_device: str = "cpu"

    def add_dependency(self, stage_id: str):
        self.dependencies.append(stage_id)

    def add_dependent(self, stage_id: str):
        self.dependents.append(stage_id)

    def run_stage(self, node_outputs: Dict[str, torch.Tensor]):
        start_time = time.time()
        transfer_time = 0.0
        if self.assigned_node and self.assigned_node.gpu is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.assigned_node.gpu}")
            self.stage_device = str(device)
        else:
            device = torch.device("cpu")
            self.stage_device = "cpu"
        try:
            with torch.no_grad():
                for fx_node in self.nodes:
                    resolved_args = resolve_arg(fx_node.args, node_outputs)
                    resolved_kwargs = resolve_arg(fx_node.kwargs, node_outputs)
                    t_start = time.time()
                    resolved_args = move_tensor_to_device(resolved_args, device)
                    resolved_kwargs = move_tensor_to_device(resolved_kwargs, device)
                    t_end = time.time()
                    transfer_time += (t_end - t_start)
                    if fx_node.op == "placeholder":
                        out = self.task.input_data.to(device)
                    elif fx_node.op == "get_attr":
                        out = getattr(self.task.model, fx_node.target)
                    elif fx_node.op == "call_module":
                        submodule = self.task.model.get_submodule(fx_node.target)
                        submodule.to(device)
                        out = submodule(*resolved_args, **resolved_kwargs)
                    elif fx_node.op == "call_function":
                        out = fx_node.target(*resolved_args, **resolved_kwargs)
                    elif fx_node.op == "call_method":
                        method = getattr(resolved_args[0], fx_node.target)
                        out = method(*resolved_args[1:], **resolved_kwargs)
                    elif fx_node.op == "output":
                        out = resolved_args[0]
                    else:
                        raise NotImplementedError(f"Operation '{fx_node.op}' is not supported.")
                    node_outputs[fx_node.name] = out
        except Exception as e:
            print(f"[Stage] {self.stage_id} error: {e}")
            self.execution_time = float('inf')
            self.transfer_time = float('inf')
            node_outputs[self.stage_id] = None
            return
        finally:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            end_time = time.time()
            self.execution_time = end_time - start_time
            self.transfer_time = transfer_time
        self.task.update_busy_time(self.execution_time, self.transfer_time)
        if not self.dependents:
            final_output_node = next((n for n in self.nodes if n.op == 'output'), None)
            if final_output_node:
                arg = final_output_node.args[0]
                if isinstance(arg, torch.Tensor):
                    final_res = arg.cpu()
                elif isinstance(arg, fx.Node):
                    final_res = node_outputs.get(arg.name, None)
                else:
                    final_res = None
                self.task.set_output_data(final_res)
            else:
                self.task.set_output_data(None)
            print(f"[Stage] {self.stage_id}: Executed on {self.assigned_node.node_id} in {self.execution_time:.4f} s, Transfer: {self.transfer_time:.4f} s.")

    def __repr__(self):
        return (f"Stage({self.stage_id}, device={self.stage_device}, node={self.assigned_node.node_id if self.assigned_node else 'None'}, "
                f"deps={self.dependencies}, exec_time={self.execution_time}, transfer_time={self.transfer_time})")
