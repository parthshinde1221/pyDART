import os
import time
import queue
import threading
import logging
import torch
from viztracer import get_tracer

class Node:
    def __init__(self, node_id: str, cpus=None, gpu=None):
        self._node_id = node_id
        self._cpus = tuple(cpus or [])
        self._gpu = gpu
        self._original_affinity = os.sched_getaffinity(0)
        self._task_queue = queue.Queue()
        self._stop_signal = False
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"Worker-{node_id}"
        )
        self._worker_thread.start()
        self.current_load = 0.0
        self.assigned_stages = []

    @property
    def node_id(self):
        return self._node_id

    @property
    def cpus(self):
        return self._cpus

    @property
    def gpu(self):
        return self._gpu

    def assign_task(self, func, *args, **kwargs):
        result_queue = queue.Queue(maxsize=1)
        self._task_queue.put((func, args, kwargs, result_queue))
        return result_queue

    def stop(self):
        self._stop_signal = True
        self._task_queue.put(None)
        self._worker_thread.join()

    def _worker_loop(self):
        while not self._stop_signal:
            item = self._task_queue.get()
            if item is None:
                break
            func, args, kwargs, result_queue = item
            with get_tracer().log_event(f"Task_{func.__name__}"):
                logging.debug(f"Task START: {func.__name__}")
                start_time = time.time()
                try:
                    self._set_context()
                    result = func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Task ERROR: {func.__name__} with error {e}")
                    result = None
                finally:
                    self._reset_context()
                end_time = time.time()
                logging.debug(f"Task FINISH: {func.__name__} Duration: {end_time - start_time:.4f}s")
                result_queue.put(result)

    def _set_context(self):
        if self._cpus:
            os.sched_setaffinity(0, self._cpus)
        if self._gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(self._gpu)
            torch.cuda.synchronize(self._gpu)

    def _reset_context(self):
        os.sched_setaffinity(0, self._original_affinity)
        if self._gpu is not None and torch.cuda.is_available():
            torch.cuda.synchronize(self._gpu)

    @staticmethod
    def discover_nodes(disjoint=True):
        nodes = []
        num_cpus = os.cpu_count() or 1
        ngpus = torch.cuda.device_count()
        if not disjoint:
            for core_id in range(num_cpus):
                nodes.append(Node(node_id=f"CPU-{core_id}", cpus=[core_id]))
            for g in range(ngpus):
                for core_id in range(num_cpus):
                    nodes.append(Node(node_id=f"GPU-{g}-CPU-{core_id}", cpus=[core_id], gpu=g))
        else:
            cpu_nodes = [Node(node_id=f"CPU-{i}", cpus=[i]) for i in range(num_cpus)]
            gpu_nodes = []
            for g in range(ngpus):
                if cpu_nodes:
                    cpu_node = cpu_nodes.pop()
                    gpu_nodes.append(Node(node_id=f"GPU-{g}-CPU-{cpu_node.cpus[0]}", cpus=[cpu_node.cpus[0]], gpu=g))
                else:
                    gpu_nodes.append(Node(node_id=f"GPU-{g}", cpus=[], gpu=g))
            nodes = gpu_nodes + cpu_nodes
        return nodes

    def __repr__(self):
        return f"Node({self._node_id}, cpus={self._cpus}, gpu={self._gpu})"
