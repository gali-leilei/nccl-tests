#!/usr/bin/env python3
"""
PyTorch NCCL Benchmark Suite

A Python-based NCCL benchmark that emulates nccl-tests using PyTorch's distributed
primitives. This leverages the call chain: Python -> torch.distributed -> libtorch.so -> libnccl.so

Usage:
    # Single node, 8 GPUs
    torchrun --nproc_per_node 8 pytorch_nccl_tests.py --op all_reduce -b 8 -e 128M -f 2

    # Multi-node (2 nodes, 8 GPUs each)
    torchrun --nnodes $WORLD_SIZE --nproc_per_node 8 --master-addr $MASTER_ADDR --master-port $MASTER_PORT --node-rank $NODE_RANK \
        pytorch_nccl_tests.py --op all_reduce -b 8 -e 128M -f 2
"""

import argparse
import os
import re
import socket
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist


# =============================================================================
# Constants and Type Mappings
# =============================================================================

DTYPE_MAP = {
    "float": torch.float32,
    "float32": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "double": torch.float64,
    "float64": torch.float64,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "uint8": torch.uint8,
    "int8": torch.int8,
}

REDOP_MAP = {
    "sum": dist.ReduceOp.SUM,
    "prod": dist.ReduceOp.PRODUCT,
    "min": dist.ReduceOp.MIN,
    "max": dist.ReduceOp.MAX,
}

# Operations that support reduction
REDUCTION_OPS = {"all_reduce", "reduce", "reduce_scatter"}

# All supported operations
SUPPORTED_OPS = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "reduce",
    "all_to_all",
    "send_recv",
    "scatter",
    "gather",
]


# =============================================================================
# Utility Functions
# =============================================================================

def parse_size(size_str: str) -> int:
    """Parse size string like '8', '128M', '1G' into bytes."""
    size_str = size_str.strip().upper()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT]?)I?B?$", size_str)
    if not match:
        # Try parsing as plain integer
        try:
            return int(size_str)
        except ValueError:
            raise ValueError(f"Cannot parse size: {size_str}")

    value = float(match.group(1))
    suffix = match.group(2)

    multipliers = {
        "": 1,
        "K": 1024,
        "M": 1024 ** 2,
        "G": 1024 ** 3,
        "T": 1024 ** 4,
    }
    return int(value * multipliers.get(suffix, 1))


def get_hostname() -> str:
    """Get short hostname."""
    hostname = socket.gethostname()
    return hostname.split(".")[0]


def dtype_size(dtype: torch.dtype) -> int:
    """Get size in bytes for a dtype."""
    return torch.tensor([], dtype=dtype).element_size()


def dtype_name(dtype: torch.dtype) -> str:
    """Get short name for dtype."""
    names = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float64: "float64",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.uint8: "uint8",
        torch.int8: "int8",
    }
    return names.get(dtype, str(dtype))


def redop_name(op) -> str:
    """Get short name for reduce op."""
    if op == dist.ReduceOp.SUM:
        return "sum"
    elif op == dist.ReduceOp.PRODUCT:
        return "prod"
    elif op == dist.ReduceOp.MIN:
        return "min"
    elif op == dist.ReduceOp.MAX:
        return "max"
    return str(op)


# =============================================================================
# Bandwidth Calculation
# =============================================================================

@dataclass
class BandwidthResult:
    """Result of a bandwidth measurement."""
    size_bytes: int
    count: int
    time_ms: float
    alg_bw: float  # GB/s
    bus_bw: float  # GB/s


def calculate_bandwidth(
    op_name: str,
    size_bytes: int,
    time_sec: float,
    nranks: int,
) -> tuple[float, float]:
    """
    Calculate algorithm bandwidth and bus bandwidth.

    Bus bandwidth correction factors (from PERFORMANCE.md):
    - AllReduce: 2*(n-1)/n
    - AllGather/ReduceScatter/AllToAll: (n-1)/n
    - Broadcast/Reduce: 1
    """
    if time_sec <= 0:
        return 0.0, 0.0

    # Algorithm bandwidth: size / time (in GB/s)
    alg_bw = (size_bytes / 1e9) / time_sec

    # Bus bandwidth correction factor
    if op_name == "all_reduce":
        factor = 2.0 * (nranks - 1) / nranks
    elif op_name in ("all_gather", "reduce_scatter", "all_to_all"):
        factor = (nranks - 1) / nranks
    elif op_name in ("broadcast", "reduce", "send_recv", "scatter", "gather"):
        factor = 1.0
    else:
        factor = 1.0

    bus_bw = alg_bw * factor
    return alg_bw, bus_bw


# =============================================================================
# Collective Operations
# =============================================================================

class CollectiveRunner:
    """Runs collective operations and measures performance."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        local_rank: int,
        device: torch.device,
    ):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device

    def _time_operation(
        self,
        op_fn: Callable[[], None],
        warmup_iters: int,
        timed_iters: int,
    ) -> float:
        """
        Time an operation using CUDA events.

        Returns average time in seconds.
        """
        # Warmup
        for _ in range(warmup_iters):
            op_fn()

        # Synchronize before timing
        torch.cuda.synchronize(self.device)
        dist.barrier()

        # Timed iterations
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record(stream=torch.cuda.current_stream(self.device))
        for _ in range(timed_iters):
            op_fn()
        end_event.record(stream=torch.cuda.current_stream(self.device))

        torch.cuda.synchronize(self.device)

        # Time in milliseconds -> seconds
        elapsed_ms = start_event.elapsed_time(end_event)
        return (elapsed_ms / 1000.0) / timed_iters

    def run_all_reduce(
        self,
        size_bytes: int,
        dtype: torch.dtype,
        op,  # dist.ReduceOp
        warmup_iters: int,
        timed_iters: int,
        check: bool = False,
    ) -> BandwidthResult:
        """Run AllReduce benchmark."""
        elem_size = dtype_size(dtype)
        count = max(1, size_bytes // elem_size)
        actual_bytes = count * elem_size

        # Allocate buffers
        sendbuf = torch.ones(count, dtype=dtype, device=self.device) * (self.rank + 1)

        def op_fn():
            buf = sendbuf.clone()
            dist.all_reduce(buf, op=op)

        time_sec = self._time_operation(op_fn, warmup_iters, timed_iters)

        # Correctness check
        if check:
            buf = sendbuf.clone()
            dist.all_reduce(buf, op=op)
            if op == dist.ReduceOp.SUM:
                expected = sum(range(1, self.world_size + 1))
                if not torch.allclose(buf, torch.full_like(buf, expected)):
                    print(f"[Rank {self.rank}] AllReduce correctness check FAILED")

        alg_bw, bus_bw = calculate_bandwidth("all_reduce", actual_bytes, time_sec, self.world_size)
        return BandwidthResult(actual_bytes, count, time_sec * 1000, alg_bw, bus_bw)

    def run_all_gather(
        self,
        size_bytes: int,
        dtype: torch.dtype,
        warmup_iters: int,
        timed_iters: int,
        check: bool = False,
    ) -> BandwidthResult:
        """Run AllGather benchmark."""
        elem_size = dtype_size(dtype)
        # size_bytes is per-rank input size
        count_per_rank = max(1, size_bytes // elem_size)
        total_count = count_per_rank * self.world_size
        actual_bytes = total_count * elem_size

        # Allocate buffers
        sendbuf = torch.ones(count_per_rank, dtype=dtype, device=self.device) * (self.rank + 1)
        recvbuf = torch.zeros(total_count, dtype=dtype, device=self.device)

        def op_fn():
            dist.all_gather_into_tensor(recvbuf, sendbuf)

        time_sec = self._time_operation(op_fn, warmup_iters, timed_iters)

        # Correctness check
        if check:
            dist.all_gather_into_tensor(recvbuf, sendbuf)
            for r in range(self.world_size):
                start = r * count_per_rank
                end = start + count_per_rank
                expected = r + 1
                if not torch.allclose(recvbuf[start:end], torch.full((count_per_rank,), expected, dtype=dtype, device=self.device)):
                    print(f"[Rank {self.rank}] AllGather correctness check FAILED at rank {r}")

        alg_bw, bus_bw = calculate_bandwidth("all_gather", actual_bytes, time_sec, self.world_size)
        return BandwidthResult(actual_bytes, total_count, time_sec * 1000, alg_bw, bus_bw)

    def run_reduce_scatter(
        self,
        size_bytes: int,
        dtype: torch.dtype,
        op,  # dist.ReduceOp
        warmup_iters: int,
        timed_iters: int,
        check: bool = False,
    ) -> BandwidthResult:
        """Run ReduceScatter benchmark."""
        elem_size = dtype_size(dtype)
        # size_bytes is total size, output is size/nranks
        total_count = max(self.world_size, size_bytes // elem_size)
        # Make sure total_count is divisible by world_size
        total_count = (total_count // self.world_size) * self.world_size
        count_per_rank = total_count // self.world_size
        actual_bytes = total_count * elem_size

        # Allocate buffers
        sendbuf = torch.ones(total_count, dtype=dtype, device=self.device) * (self.rank + 1)
        recvbuf = torch.zeros(count_per_rank, dtype=dtype, device=self.device)

        def op_fn():
            sendbuf_copy = sendbuf.clone()
            dist.reduce_scatter_tensor(recvbuf, sendbuf_copy, op=op)

        time_sec = self._time_operation(op_fn, warmup_iters, timed_iters)

        # Correctness check
        if check and op == dist.ReduceOp.SUM:
            sendbuf_copy = sendbuf.clone()
            dist.reduce_scatter_tensor(recvbuf, sendbuf_copy, op=op)
            expected = sum(range(1, self.world_size + 1))
            if not torch.allclose(recvbuf, torch.full_like(recvbuf, expected)):
                print(f"[Rank {self.rank}] ReduceScatter correctness check FAILED")

        alg_bw, bus_bw = calculate_bandwidth("reduce_scatter", actual_bytes, time_sec, self.world_size)
        return BandwidthResult(actual_bytes, total_count, time_sec * 1000, alg_bw, bus_bw)

    def run_broadcast(
        self,
        size_bytes: int,
        dtype: torch.dtype,
        root: int,
        warmup_iters: int,
        timed_iters: int,
        check: bool = False,
    ) -> BandwidthResult:
        """Run Broadcast benchmark."""
        elem_size = dtype_size(dtype)
        count = max(1, size_bytes // elem_size)
        actual_bytes = count * elem_size

        # Allocate buffer
        if self.rank == root:
            buf = torch.ones(count, dtype=dtype, device=self.device) * 42.0
        else:
            buf = torch.zeros(count, dtype=dtype, device=self.device)

        def op_fn():
            nonlocal buf
            if self.rank == root:
                buf = torch.ones(count, dtype=dtype, device=self.device) * 42.0
            else:
                buf = torch.zeros(count, dtype=dtype, device=self.device)
            dist.broadcast(buf, src=root)

        time_sec = self._time_operation(op_fn, warmup_iters, timed_iters)

        # Correctness check
        if check:
            if self.rank == root:
                buf = torch.ones(count, dtype=dtype, device=self.device) * 42.0
            else:
                buf = torch.zeros(count, dtype=dtype, device=self.device)
            dist.broadcast(buf, src=root)
            if not torch.allclose(buf, torch.full_like(buf, 42.0)):
                print(f"[Rank {self.rank}] Broadcast correctness check FAILED")

        alg_bw, bus_bw = calculate_bandwidth("broadcast", actual_bytes, time_sec, self.world_size)
        return BandwidthResult(actual_bytes, count, time_sec * 1000, alg_bw, bus_bw)

    def run_reduce(
        self,
        size_bytes: int,
        dtype: torch.dtype,
        op,  # dist.ReduceOp
        root: int,
        warmup_iters: int,
        timed_iters: int,
        check: bool = False,
    ) -> BandwidthResult:
        """Run Reduce benchmark."""
        elem_size = dtype_size(dtype)
        count = max(1, size_bytes // elem_size)
        actual_bytes = count * elem_size

        # Allocate buffer
        sendbuf = torch.ones(count, dtype=dtype, device=self.device) * (self.rank + 1)

        def op_fn():
            buf = sendbuf.clone()
            dist.reduce(buf, dst=root, op=op)

        time_sec = self._time_operation(op_fn, warmup_iters, timed_iters)

        # Correctness check
        if check and op == dist.ReduceOp.SUM:
            buf = sendbuf.clone()
            dist.reduce(buf, dst=root, op=op)
            if self.rank == root:
                expected = sum(range(1, self.world_size + 1))
                if not torch.allclose(buf, torch.full_like(buf, expected)):
                    print(f"[Rank {self.rank}] Reduce correctness check FAILED")

        alg_bw, bus_bw = calculate_bandwidth("reduce", actual_bytes, time_sec, self.world_size)
        return BandwidthResult(actual_bytes, count, time_sec * 1000, alg_bw, bus_bw)

    def run_all_to_all(
        self,
        size_bytes: int,
        dtype: torch.dtype,
        warmup_iters: int,
        timed_iters: int,
        check: bool = False,
    ) -> BandwidthResult:
        """Run AllToAll benchmark."""
        elem_size = dtype_size(dtype)
        # size_bytes is total size per rank
        total_count = max(self.world_size, size_bytes // elem_size)
        # Make sure total_count is divisible by world_size
        total_count = (total_count // self.world_size) * self.world_size
        actual_bytes = total_count * elem_size

        # Allocate buffers
        sendbuf = torch.zeros(total_count, dtype=dtype, device=self.device)
        recvbuf = torch.zeros(total_count, dtype=dtype, device=self.device)

        # Fill send buffer: each chunk going to rank r has value (self.rank * world_size + r)
        chunk_size = total_count // self.world_size
        for r in range(self.world_size):
            start = r * chunk_size
            end = start + chunk_size
            sendbuf[start:end] = self.rank * self.world_size + r

        def op_fn():
            dist.all_to_all_single(recvbuf, sendbuf)

        time_sec = self._time_operation(op_fn, warmup_iters, timed_iters)

        # Correctness check
        if check:
            dist.all_to_all_single(recvbuf, sendbuf)
            for r in range(self.world_size):
                start = r * chunk_size
                end = start + chunk_size
                expected = r * self.world_size + self.rank
                if not torch.allclose(recvbuf[start:end], torch.full((chunk_size,), expected, dtype=dtype, device=self.device)):
                    print(f"[Rank {self.rank}] AllToAll correctness check FAILED from rank {r}")

        alg_bw, bus_bw = calculate_bandwidth("all_to_all", actual_bytes, time_sec, self.world_size)
        return BandwidthResult(actual_bytes, total_count, time_sec * 1000, alg_bw, bus_bw)

    def run_send_recv(
        self,
        size_bytes: int,
        dtype: torch.dtype,
        warmup_iters: int,
        timed_iters: int,
        check: bool = False,
    ) -> BandwidthResult:
        """
        Run SendRecv benchmark.
        Each rank sends to (rank+1) % world_size and receives from (rank-1) % world_size.
        """
        elem_size = dtype_size(dtype)
        count = max(1, size_bytes // elem_size)
        actual_bytes = count * elem_size

        send_to = (self.rank + 1) % self.world_size
        recv_from = (self.rank - 1 + self.world_size) % self.world_size

        sendbuf = torch.ones(count, dtype=dtype, device=self.device) * (self.rank + 1)
        recvbuf = torch.zeros(count, dtype=dtype, device=self.device)

        def op_fn():
            nonlocal recvbuf
            recvbuf = torch.zeros(count, dtype=dtype, device=self.device)
            # Use sendrecv for bidirectional communication
            if self.rank % 2 == 0:
                dist.send(sendbuf, dst=send_to)
                dist.recv(recvbuf, src=recv_from)
            else:
                dist.recv(recvbuf, src=recv_from)
                dist.send(sendbuf, dst=send_to)

        time_sec = self._time_operation(op_fn, warmup_iters, timed_iters)

        # Correctness check
        if check:
            recvbuf = torch.zeros(count, dtype=dtype, device=self.device)
            if self.rank % 2 == 0:
                dist.send(sendbuf, dst=send_to)
                dist.recv(recvbuf, src=recv_from)
            else:
                dist.recv(recvbuf, src=recv_from)
                dist.send(sendbuf, dst=send_to)
            expected = recv_from + 1
            if not torch.allclose(recvbuf, torch.full_like(recvbuf, expected)):
                print(f"[Rank {self.rank}] SendRecv correctness check FAILED")

        alg_bw, bus_bw = calculate_bandwidth("send_recv", actual_bytes, time_sec, self.world_size)
        return BandwidthResult(actual_bytes, count, time_sec * 1000, alg_bw, bus_bw)

    def run_scatter(
        self,
        size_bytes: int,
        dtype: torch.dtype,
        root: int,
        warmup_iters: int,
        timed_iters: int,
        check: bool = False,
    ) -> BandwidthResult:
        """
        Run Scatter benchmark.
        Root rank distributes different chunks of data to each rank.
        """
        elem_size = dtype_size(dtype)
        count = max(1, size_bytes // elem_size)
        actual_bytes = count * elem_size

        # Output buffer (all ranks receive into this)
        recvbuf = torch.zeros(count, dtype=dtype, device=self.device)

        # Input list (only needed on root rank)
        if self.rank == root:
            scatter_list = [
                torch.ones(count, dtype=dtype, device=self.device) * (r + 1)
                for r in range(self.world_size)
            ]
        else:
            scatter_list = None

        def op_fn():
            nonlocal recvbuf
            recvbuf = torch.zeros(count, dtype=dtype, device=self.device)
            dist.scatter(recvbuf, scatter_list, src=root)

        time_sec = self._time_operation(op_fn, warmup_iters, timed_iters)

        # Correctness check
        if check:
            recvbuf = torch.zeros(count, dtype=dtype, device=self.device)
            dist.scatter(recvbuf, scatter_list, src=root)
            expected = self.rank + 1
            if not torch.allclose(recvbuf, torch.full_like(recvbuf, expected)):
                print(f"[Rank {self.rank}] Scatter correctness check FAILED")

        alg_bw, bus_bw = calculate_bandwidth("scatter", actual_bytes, time_sec, self.world_size)
        return BandwidthResult(actual_bytes, count, time_sec * 1000, alg_bw, bus_bw)

    def run_gather(
        self,
        size_bytes: int,
        dtype: torch.dtype,
        root: int,
        warmup_iters: int,
        timed_iters: int,
        check: bool = False,
    ) -> BandwidthResult:
        """
        Run Gather benchmark.
        All ranks send data to root rank, which collects into a list.
        """
        elem_size = dtype_size(dtype)
        count = max(1, size_bytes // elem_size)
        actual_bytes = count * elem_size

        # Each rank has its own send buffer
        sendbuf = torch.ones(count, dtype=dtype, device=self.device) * (self.rank + 1)

        # Gather list (only needed on root rank)
        if self.rank == root:
            gather_list = [
                torch.zeros(count, dtype=dtype, device=self.device)
                for _ in range(self.world_size)
            ]
        else:
            gather_list = None

        def op_fn():
            nonlocal gather_list
            if self.rank == root:
                gather_list = [
                    torch.zeros(count, dtype=dtype, device=self.device)
                    for _ in range(self.world_size)
                ]
            dist.gather(sendbuf, gather_list, dst=root)

        time_sec = self._time_operation(op_fn, warmup_iters, timed_iters)

        # Correctness check
        if check:
            if self.rank == root:
                gather_list = [
                    torch.zeros(count, dtype=dtype, device=self.device)
                    for _ in range(self.world_size)
                ]
            dist.gather(sendbuf, gather_list, dst=root)
            if self.rank == root:
                for r in range(self.world_size):
                    expected = r + 1
                    if not torch.allclose(gather_list[r], torch.full_like(gather_list[r], expected)):
                        print(f"[Rank {self.rank}] Gather correctness check FAILED for rank {r}")

        alg_bw, bus_bw = calculate_bandwidth("gather", actual_bytes, time_sec, self.world_size)
        return BandwidthResult(actual_bytes, count, time_sec * 1000, alg_bw, bus_bw)


# =============================================================================
# Main Benchmark Logic
# =============================================================================

def generate_sizes(min_bytes: int, max_bytes: int, step_factor: Optional[float], step_bytes: Optional[int]) -> list[int]:
    """Generate list of sizes to benchmark."""
    sizes = []
    current = min_bytes

    while current <= max_bytes:
        sizes.append(current)
        if step_factor is not None and step_factor > 1:
            current = int(current * step_factor)
        elif step_bytes is not None and step_bytes > 0:
            current += step_bytes
        else:
            break

        # Avoid infinite loop
        if current <= sizes[-1]:
            break

    # Make sure max_bytes is included
    if sizes and sizes[-1] < max_bytes:
        sizes.append(max_bytes)

    return sizes


def print_header(op_name: str, dtype: torch.dtype, nranks: int, is_reduction: bool):
    """Print benchmark header."""
    print(f"#")
    print(f"# PyTorch NCCL Tests - {op_name}")
    print(f"#")
    print(f"# nranks: {nranks}")
    print(f"# dtype: {dtype_name(dtype)}")
    print(f"#")

    if is_reduction:
        print(f"#{'size':>12} {'count':>12} {'type':>10} {'redop':>8} {'root':>6} {'time(ms)':>12} {'algbw(GB/s)':>14} {'busbw(GB/s)':>14}")
    else:
        print(f"#{'size':>12} {'count':>12} {'type':>10} {'root':>6} {'time(ms)':>12} {'algbw(GB/s)':>14} {'busbw(GB/s)':>14}")

    print("#" + "-" * 100)


def print_result(result: BandwidthResult, dtype: torch.dtype, redop, root: Optional[int], is_reduction: bool):
    """Print a single benchmark result."""
    type_str = dtype_name(dtype)
    root_str = str(root) if root is not None else "-"

    if is_reduction:
        op_str = redop_name(redop) if redop else "-"
        print(f" {result.size_bytes:>12} {result.count:>12} {type_str:>10} {op_str:>8} {root_str:>6} {result.time_ms:>12.4f} {result.alg_bw:>14.2f} {result.bus_bw:>14.2f}")
    else:
        print(f" {result.size_bytes:>12} {result.count:>12} {type_str:>10} {root_str:>6} {result.time_ms:>12.4f} {result.alg_bw:>14.2f} {result.bus_bw:>14.2f}")


def print_summary(results: list[BandwidthResult]):
    """Print summary statistics."""
    if not results:
        return

    bus_bws = [r.bus_bw for r in results]
    avg_bus_bw = sum(bus_bws) / len(bus_bws)
    max_bus_bw = max(bus_bws)
    min_bus_bw = min(bus_bws)

    print("#" + "-" * 100)
    print(f"# Avg bus bandwidth    : {avg_bus_bw:.2f} GB/s")
    print(f"# Max bus bandwidth    : {max_bus_bw:.2f} GB/s")
    print(f"# Min bus bandwidth    : {min_bus_bw:.2f} GB/s")
    print(f"# Number of data points: {len(results)}")


def run_benchmark(args):
    """Main benchmark runner."""
    # Initialize distributed
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Parse parameters
    dtype = DTYPE_MAP.get(args.datatype.lower(), torch.float32)
    redop = REDOP_MAP.get(args.op_type.lower(), dist.ReduceOp.SUM) if args.op_type else dist.ReduceOp.SUM
    root = args.root

    # Generate sizes
    sizes = generate_sizes(args.minbytes, args.maxbytes, args.stepfactor, args.stepbytes)

    # Create runner
    runner = CollectiveRunner(rank, world_size, local_rank, device)

    # Print header (rank 0 only)
    is_reduction = args.op in REDUCTION_OPS
    if rank == 0:
        print_header(args.op, dtype, world_size, is_reduction)

    # Run benchmarks
    results = []
    for size in sizes:
        dist.barrier()

        if args.op == "all_reduce":
            result = runner.run_all_reduce(size, dtype, redop, args.warmup_iters, args.iters, args.check)
        elif args.op == "all_gather":
            result = runner.run_all_gather(size, dtype, args.warmup_iters, args.iters, args.check)
        elif args.op == "reduce_scatter":
            result = runner.run_reduce_scatter(size, dtype, redop, args.warmup_iters, args.iters, args.check)
        elif args.op == "broadcast":
            result = runner.run_broadcast(size, dtype, root, args.warmup_iters, args.iters, args.check)
        elif args.op == "reduce":
            result = runner.run_reduce(size, dtype, redop, root, args.warmup_iters, args.iters, args.check)
        elif args.op == "all_to_all":
            result = runner.run_all_to_all(size, dtype, args.warmup_iters, args.iters, args.check)
        elif args.op == "send_recv":
            result = runner.run_send_recv(size, dtype, args.warmup_iters, args.iters, args.check)
        elif args.op == "scatter":
            result = runner.run_scatter(size, dtype, root, args.warmup_iters, args.iters, args.check)
        elif args.op == "gather":
            result = runner.run_gather(size, dtype, root, args.warmup_iters, args.iters, args.check)
        else:
            raise ValueError(f"Unknown operation: {args.op}")

        results.append(result)

        if rank == 0:
            print_result(result, dtype, redop if is_reduction else None, root if args.op in ("broadcast", "reduce", "scatter", "gather") else None, is_reduction)

    # Print summary (rank 0 only)
    if rank == 0:
        print_summary(results)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch NCCL Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single node, 8 GPUs, AllReduce
    torchrun --nproc_per_node=8 pytorch_nccl_tests.py --op all_reduce -b 8 -e 128M -f 2

    # All operations with default settings
    torchrun --nproc_per_node=8 pytorch_nccl_tests.py --op all_gather -b 1K -e 1G -f 2
        """,
    )

    # Operation selection
    parser.add_argument(
        "--op",
        type=str,
        choices=SUPPORTED_OPS,
        default="all_reduce",
        help=f"Collective operation to benchmark (default: all_reduce)",
    )

    # Size parameters (match nccl-tests)
    parser.add_argument(
        "-b", "--minbytes",
        type=str,
        default="32M",
        help="Minimum message size (default: 32M)",
    )
    parser.add_argument(
        "-e", "--maxbytes",
        type=str,
        default="32M",
        help="Maximum message size (default: 32M)",
    )
    parser.add_argument(
        "-i", "--stepbytes",
        type=str,
        default=None,
        help="Fixed increment between sizes (default: 1M)",
    )
    parser.add_argument(
        "-f", "--stepfactor",
        type=float,
        default=None,
        help="Multiplication factor between sizes (e.g., 2 for doubling)",
    )

    # NCCL operation arguments
    parser.add_argument(
        "-o", "--op_type",
        type=str,
        choices=["sum", "prod", "min", "max"],
        default="sum",
        help="Reduction operation (default: sum)",
    )
    parser.add_argument(
        "-d", "--datatype",
        type=str,
        default="float",
        help=f"Data type (default: float). Choices: {', '.join(DTYPE_MAP.keys())}",
    )
    parser.add_argument(
        "-r", "--root",
        type=int,
        default=0,
        help="Root rank for broadcast/reduce (default: 0)",
    )

    # Performance parameters
    parser.add_argument(
        "-n", "--iters",
        type=int,
        default=20,
        help="Number of timed iterations (default: 20)",
    )
    parser.add_argument(
        "-w", "--warmup_iters",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )

    # Test options
    parser.add_argument(
        "-c", "--check",
        action="store_true",
        help="Enable correctness checking",
    )

    args = parser.parse_args()

    # Parse size strings
    args.minbytes = parse_size(args.minbytes)
    args.maxbytes = parse_size(args.maxbytes)
    if args.stepbytes:
        args.stepbytes = parse_size(args.stepbytes)
    else:
        args.stepbytes = parse_size("1M")

    # Default step factor if neither is specified
    if args.stepfactor is None and args.stepbytes is None:
        args.stepbytes = parse_size("1M")

    run_benchmark(args)


if __name__ == "__main__":
    main()
