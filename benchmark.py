#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

import argparse
import os
import re
import shlex
import sqlite3
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import TypedDict

DB_PATH = Path(__file__).parent / "benchmark.db"


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
    # Migration: add exit_status column if it doesn't exist
    cursor = conn.execute("PRAGMA table_info(benchmark_runs)")
    columns = [row[1] for row in cursor.fetchall()]
    if "exit_status" not in columns:
        conn.execute("ALTER TABLE benchmark_runs ADD COLUMN exit_status INTEGER")
    conn.commit()

    return conn


def extract_command_kind(command: str) -> str:
    """Extract the executable name from a shell command (args[0] basename).

    e.g. "/path/to/all_reduce_perf -b 8" -> "all_reduce_perf"
    """
    try:
        parts = shlex.split(command)
        if parts:
            return Path(parts[0]).name
    except ValueError:
        # shlex.split can raise on malformed input
        pass
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


def cmd_single_test(conn: sqlite3.Connection, command: str) -> str:
    """Run a benchmark command and store results.

    Returns the run_id.
    """
    run_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    # Run the command and capture output
    # Create environment with NCCL_DEBUG enabled to capture debug messages
    env = os.environ.copy()
    env["NCCL_DEBUG"] = "INFO"

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        env=env,
    )
    log_output = result.stdout + result.stderr

    # Extract command kind and analyze log
    command_kind = extract_command_kind(command)
    analysis = analyze_log(log_output)

    # Store in database
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

    return run_id


def cmd_list_runs(conn: sqlite3.Connection) -> list[tuple]:
    """List all stored benchmark runs.

    Returns list of (id, command_kind, nccl_version, avg_bandwidth, created_at) tuples.
    """
    cursor = conn.execute(
        """SELECT id, command_kind, nccl_version, avg_bandwidth, created_at
           FROM benchmark_runs ORDER BY created_at DESC"""
    )
    return cursor.fetchall()


def find_executables(build_dir: Path) -> list[Path]:
    """Find all executable files under the build directory.

    Returns a list of Path objects for each executable.
    """
    executables = []
    if not build_dir.exists():
        return executables

    for item in build_dir.iterdir():
        if item.is_file() and os.access(item, os.X_OK):
            executables.append(item)

    return sorted(executables)


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
    conn: sqlite3.Connection, args_str: str, repeat: int, dry_run: bool = False
) -> list[AverageResult]:
    """Run all executables in ./build with given args, repeated N times.

    Returns averaged results grouped by (command, nccl_version) from ALL runs in DB.
    """
    build_dir = Path(__file__).parent / "build"
    binaries = find_executables(build_dir)

    if not binaries:
        print(f"No executables found in {build_dir}")
        return []

    print(f"Found {len(binaries)} executable(s) in {build_dir}:")
    for b in binaries:
        print(f"  - {b.name}")
    print()

    # Build commands from binaries and args
    commands = []
    for binary in binaries:
        cmd = f"{binary} {args_str}".strip()
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

            run_id = cmd_single_test(conn, cmd)

            # Fetch the result we just stored
            cursor = conn.execute(
                "SELECT nccl_version, avg_bandwidth, exit_status FROM benchmark_runs WHERE id = ?",
                (run_id,),
            )
            row = cursor.fetchone()
            if row:
                nccl_version, avg_bandwidth, exit_status = row
                status_str = (
                    "OK" if exit_status == 0 else f"FAILED (exit={exit_status})"
                )
                print(
                    f"    -> [{status_str}] NCCL {nccl_version}, Bandwidth: {avg_bandwidth}"
                )
            print()

    # Compute averages from ALL runs in DB grouped by (command, nccl_version)
    return compute_average_results(conn)


def main() -> None:
    parser = argparse.ArgumentParser(description="NCCL Benchmark CLI")
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
        help="Run all executables in ./build with given args, repeated N times",
    )
    all_tests_parser.add_argument(
        "--args",
        type=str,
        default="-b 8 -e 128M -f 2 -g 8",
        help="Arguments to pass to each executable (e.g., '-b 8 -e 128M')",
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

    conn = get_connection()
    try:
        if args.subcommand == "single-test":
            run_id = cmd_single_test(conn, args.command)
            print(f"Run ID: {run_id}")

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
