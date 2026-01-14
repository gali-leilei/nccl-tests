#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

import argparse
import os
import re
import sqlite3
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from string import Template
from typing import TypedDict

DB_PATH = Path(__file__).parent / "pytorch_benchmark.db"


def is_master_node() -> bool:
    """Check if this is the master node (node rank 0).

    In SPMD execution, only node rank 0 should write to the database.
    Returns True if RANK is 0 or not set (single-node case).
    """
    rank = os.environ.get("RANK", "0")
    return rank == "0"


# Supported operations matching pytorch_nccl_tests.py
SUPPORTED_OPS = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "reduce",
    "all_to_all",
    "send_recv",
    "scatter",
    "gather",
)

CMD_TEMPLATE = Template(
    "NCCL_DEBUG=INFO torchrun --nnodes $$WORLD_SIZE --nproc_per_node 8 "
    "--master-addr $$MASTER_ADDR --master-port $$MASTER_PORT --node-rank $$RANK "
    "$script_path --op $op $args_str"
)

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id TEXT PRIMARY KEY,
            command TEXT NOT NULL,
            command_kind TEXT,
            log_output TEXT NOT NULL,
            nccl_version TEXT,
            avg_bandwidth TEXT,
            exit_status INTEGER,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()

    return conn


def extract_command_kind(command: str) -> str:
    """Extract the operation name from a pytorch_nccl_tests.py command.

    e.g. "torchrun ... --op all_reduce -b 8" -> "all_reduce"
    e.g. "torchrun ... --op=all_reduce -b 8" -> "all_reduce"

    """
    # Try to extract --op argument value (supports both --op value and --op=value)
    op_match = re.search(r"--op[=\s]+(\w+)", command)
    if op_match:
        return op_match.group(1)
    return "unknown"


class LogResult(TypedDict):
    nccl_version: str
    avg_bandwidth: str


def analyze_log(log_output: str) -> LogResult:
    """Extract metrics from log output.

    Returns dict with nccl_version, avg_bandwidth.
    """
    # Extract NCCL version
    version_match = re.search(r"NCCL INFO NCCL version ([^\s,]+)", log_output)
    nccl_version = version_match.group(1) if version_match else "unknown"

    # Extract average bus bandwidth
    bandwidth_match = re.search(r"# Avg bus bandwidth\s*:\s*([\d.]+)", log_output)
    avg_bandwidth = bandwidth_match.group(1) if bandwidth_match else "unknown"

    return LogResult(nccl_version=nccl_version, avg_bandwidth=avg_bandwidth)


class SingleTestResult(TypedDict):
    run_id: str
    command_kind: str
    nccl_version: str
    avg_bandwidth: str
    exit_status: int


def cmd_single_test(conn: sqlite3.Connection, command: str) -> SingleTestResult:
    """Run a benchmark command and store results.

    Args:
        conn: Database connection
        command: The command to run

    Returns:
        SingleTestResult with run info. Only writes to DB on master node (RANK=0).
    """
    run_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    # Run the command and capture output
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
    )
    log_output = result.stdout + result.stderr

    # Extract command kind and analyze log
    command_kind = extract_command_kind(command)
    analysis = analyze_log(log_output)

    # Only store in database on master node (RANK=0)
    if is_master_node():
        conn.execute(
            """INSERT INTO benchmark_runs
               (id, command, command_kind, log_output, nccl_version, avg_bandwidth, exit_status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                command,
                command_kind,
                log_output,
                analysis["nccl_version"],
                analysis["avg_bandwidth"],
                result.returncode,
                created_at,
            ),
        )
        conn.commit()

    return SingleTestResult(
        run_id=run_id,
        command_kind=command_kind,
        nccl_version=analysis["nccl_version"],
        avg_bandwidth=analysis["avg_bandwidth"],
        exit_status=result.returncode,
    )


def cmd_list_runs(conn: sqlite3.Connection) -> list[tuple]:
    """List all stored benchmark runs.

    Returns list of (id, command_kind, nccl_version, avg_bandwidth, created_at) tuples.
    """
    cursor = conn.execute(
        """SELECT id, command_kind, nccl_version, avg_bandwidth, created_at
           FROM benchmark_runs ORDER BY created_at DESC"""
    )
    return cursor.fetchall()


class AverageResult(TypedDict):
    command: str
    nccl_version: str
    avg_bandwidth: float
    num_runs: int


def compute_average_results(
    conn: sqlite3.Connection,
) -> list[AverageResult]:
    """Compute average bandwidth from DB grouped by (command, nccl_version).

    Args:
        conn: Database connection

    Returns:
        List of averaged results, only including successful runs (exit_status = 0).
    """
    cursor = conn.execute(
        """SELECT command, nccl_version, AVG(CAST(avg_bandwidth AS REAL)), COUNT(*)
            FROM benchmark_runs
            WHERE avg_bandwidth != 'unknown'
                AND exit_status = 0
            GROUP BY command, nccl_version
            ORDER BY command, nccl_version"""
    )

    return [
        AverageResult(
            command=cmd,
            nccl_version=nccl_version,
            avg_bandwidth=avg_bw,
            num_runs=num_runs,
        )
        for cmd, nccl_version, avg_bw, num_runs in cursor.fetchall()
        if avg_bw is not None
    ]


def cmd_all_tests(
    conn: sqlite3.Connection,
    args_str: str,
    repeat: int,
    dry_run: bool = False,
) -> list[AverageResult]:
    """Run all pytorch_nccl_tests.py operations with given args, repeated N times.

    Returns averaged results grouped by (command, nccl_version) from ALL runs in DB.
    """
    script_path = Path(__file__).parent / "pytorch_nccl_tests.py"

    if not script_path.exists():
        print(f"Error: pytorch_nccl_tests.py not found at {script_path}")
        return []

    print(f"Running {len(SUPPORTED_OPS)} operation(s) with 8 GPU(s):")
    for op in SUPPORTED_OPS:
        print(f"  - {op}")
    print()

    # Build commands for each operation
    commands = []
    for op in SUPPORTED_OPS:
        cmd = CMD_TEMPLATE.substitute(script_path=script_path, op=op, args_str=args_str).strip()
        commands.append(cmd)

    # Dry run: just list the commands
    if dry_run:
        print(
            f"Commands to run ({len(commands)} commands x {repeat} repeats = {len(commands) * repeat} total):"
        )
        for cmd in commands:
            print(f"  {cmd}")
        return []

    total_runs = len(commands) * repeat
    current_run = 0

    for iteration in range(1, repeat + 1):
        print(f"=== Iteration {iteration}/{repeat} ===")
        for cmd in commands:
            current_run += 1
            print(f"[{current_run}/{total_runs}] Running: {cmd}")

            result = cmd_single_test(conn, cmd)

            status_str = (
                "OK" if result["exit_status"] == 0 else f"FAILED (exit={result['exit_status']})"
            )
            print(
                f"    -> [{status_str}] NCCL {result['nccl_version']}, Bandwidth: {result['avg_bandwidth']}"
            )
            print()

    # Compute averages from ALL runs in DB grouped by (command, nccl_version)
    # Only meaningful on master node (where data was written)
    if is_master_node():
        return compute_average_results(conn)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch NCCL Benchmark CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # single-test subcommand
    single_test_parser = subparsers.add_parser(
        "single-test", help="Run a benchmark command and store results"
    )
    single_test_parser.add_argument("command", help="The benchmark command to run")

    # list-runs subcommand
    subparsers.add_parser("list-runs", help="List all stored runs")

    # all-tests subcommand
    all_tests_parser = subparsers.add_parser(
        "all-tests",
        help="Run all pytorch_nccl_tests.py operations with given args, repeated N times",
    )
    all_tests_parser.add_argument(
        "--args",
        type=str,
        default="-b 8 -e 128M -f 2",
        help="Arguments to pass to pytorch_nccl_tests.py (e.g., '-b 8 -e 128M -f 2')",
    )
    all_tests_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat each test (default: 1)",
    )
    all_tests_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List commands without running them",
    )

    # compute-average subcommand
    subparsers.add_parser(
        "compute-average",
        help="Compute average bandwidth from all stored runs (grouped by command, nccl_version)",
    )

    args = parser.parse_args()

    # Connect to DB - writes are guarded by is_master_node() check
    # For read-only commands (list-runs, compute-average), any node can read
    conn = get_connection()
    try:
        if args.subcommand == "single-test":
            result = cmd_single_test(conn, args.command)
            print(f"Run ID: {result['run_id']}")
            print(f"NCCL Version: {result['nccl_version']}")
            print(f"Avg Bandwidth: {result['avg_bandwidth']}")
            print(f"Exit Status: {result['exit_status']}")

        elif args.subcommand == "list-runs":
            rows = cmd_list_runs(conn)
            if not rows:
                print("No runs found.")
            else:
                print(
                    f"{'UUID':<36}  {'Command Kind':<20}  {'NCCL Version':<14}  {'Avg BW':<10}  {'Timestamp'}"
                )
                print("-" * 110)
                for (
                    run_id,
                    command_kind,
                    nccl_version,
                    avg_bandwidth,
                    created_at,
                ) in rows:
                    cmd_kind_display = command_kind or "N/A"
                    version_display = nccl_version or "N/A"
                    bw_display = avg_bandwidth or "N/A"
                    print(
                        f"{run_id}  {cmd_kind_display:<20}  {version_display:<14}  {bw_display:<10}  {created_at}"
                    )

        elif args.subcommand == "all-tests":
            results = cmd_all_tests(conn, args.args, args.repeat, args.dry_run)
            if results:
                print("\n" + "=" * 80)
                print("SUMMARY: Averaged Results (grouped by command, nccl_version)")
                print("=" * 80)
                print(
                    f"{'Command':<50}  {'NCCL Version':<14}  {'Avg BW':<12}  {'Runs'}"
                )
                print("-" * 90)
                for r in results:
                    cmd_display = r["command"]
                    if len(cmd_display) > 48:
                        cmd_display = "..." + cmd_display[-45:]
                    print(
                        f"{cmd_display:<50}  {r['nccl_version']:<14}  {r['avg_bandwidth']:<12.4f}  {r['num_runs']}"
                    )
            else:
                print("No results collected.")

        elif args.subcommand == "compute-average":
            results = compute_average_results(conn)
            if results:
                print("Averaged Results (grouped by command, nccl_version)")
                print("=" * 90)
                print(
                    f"{'Command':<50}  {'NCCL Version':<14}  {'Avg BW':<12}  {'Runs'}"
                )
                print("-" * 90)
                for r in results:
                    cmd_display = r["command"]
                    if len(cmd_display) > 48:
                        cmd_display = "..." + cmd_display[-45:]
                    print(
                        f"{cmd_display:<50}  {r['nccl_version']:<14}  {r['avg_bandwidth']:<12.4f}  {r['num_runs']}"
                    )
            else:
                print("No results found.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
