#!/usr/bin/env python3
"""bench-suite: pre-commit regression benchmark for TIRx kernels.

See README.md in this directory for setup, baseline workflow, and flags.

Quick start:
    python -m tirx_kernels.bench_suite
    python -m tirx_kernels.bench_suite --rounds 5
    python tirx_kernels/bench_suite/promote_baseline.py .bench-suite/runs/<id>.json --merge

Exit codes:
    0  no regressions (or no baseline yet)
    2  config error (no workloads / bad YAML)
    3  one or more regressions exceeded the threshold
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import ClassVar

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def _kernels_repo_root() -> Path:
    """Git root of the tirx-kernels repo (parent of the tirx_kernels package)."""
    return SCRIPT_DIR.parent.parent


DEFAULT_OUT_DIR = _kernels_repo_root() / ".bench-suite"
DEFAULT_WORKLOADS = SCRIPT_DIR / "workloads.yaml"
# Single pinned baseline: every run benches our kernel + all reference impls,
# so one JSON holds both. Promote a run over it via promote_baseline.py.
DEFAULT_BASELINE = SCRIPT_DIR / "baseline.json"
DEFAULT_REGRESSION_THRESHOLD = 1.0
POLL_INTERVAL = 5.0  # seconds between GPU re-checks when none is free
MONITOR_INTERVAL = 0.5  # seconds between nvidia-smi polls during a workload
# 0 means auto: one worker per probe-OK GPU (see main()).
DEFAULT_CPU_WORKERS = 0
DEFAULT_ROUND_COOLDOWN_S = 1.0
DEFAULT_UTIL_THRESHOLD = 0.0  # % GPU util above which a card counts as busy.
DEFAULT_MEM_THRESHOLD = 0.0  # % compute-app memory above which a card counts as busy.

# Tiny real workload used to decide whether a GPU is actually usable.
# Catches: driver hangs, ECC errors when touching memory, cuBLAS init
# failures, MIG/cgroup restrictions, fragmentation surprises — issues that
# nvidia-smi "free" status alone won't surface.
PROBE_SCRIPT = r"""
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("PROBE_FAIL: torch.cuda.is_available()=False", file=sys.stderr)
        sys.exit(1)
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    del a, b, c
    torch.cuda.empty_cache()
except Exception as e:
    print(f"PROBE_FAIL: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
print("PROBE_OK")
"""


# ── Workload loading ─────────────────────────────────────────────────────────


def load_workloads(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text()) or {}
    defaults = data.get("defaults") or {}
    out: list[dict] = []
    for entry in data.get("workloads") or []:
        if "kernel" not in entry or "config" not in entry:
            raise ValueError(f"workload missing kernel/config: {entry}")
        workload = {**defaults, **entry}
        num_gpus = workload.get("num_gpus", 1)
        if type(num_gpus) is not int or num_gpus < 1:
            raise ValueError(f"workload num_gpus must be a positive integer: {workload}")
        workload["num_gpus"] = num_gpus
        if workload.get("timer") == "megamoe" and (
            workload.get("warmup") is not None or workload.get("repeat") is not None
        ):
            raise ValueError(
                "timer='megamoe' uses a fixed DeepGEMM protocol and cannot override "
                f"warmup/repeat: {workload}"
            )
        out.append(workload)
    return out


# ── GPU pool ─────────────────────────────────────────────────────────────────


class GpuPool:
    """Exclusive GPU resource pool for bench worker threads.

    A worker atomically acquires all GPUs required by its workload, holds them
    in `_owned` for the whole subprocess, then releases them in a finally block.
    At most one orchestrator job per card at a time, including multi-GPU jobs.

    acquire() also re-queries utilization each loop: a card counts as taken if
    it is in `_owned`, its GPU util is above `util_threshold`, or its memory
    use is above `mem_threshold`. Startup probe filters broken cards into `allowed`.
    """

    def __init__(
        self,
        allowed: set[str] | None = None,
        util_threshold: float = DEFAULT_UTIL_THRESHOLD,
        mem_threshold: float = DEFAULT_MEM_THRESHOLD,
    ):
        self._owned: set[str] = set()
        self._waiters: list[tuple[object, int]] = []
        self._lock = threading.Lock()
        self._allowed = allowed
        self.util_threshold = util_threshold
        self.mem_threshold = mem_threshold

    @staticmethod
    def _nvidia_smi(args: list[str]) -> list[str]:
        try:
            out = subprocess.run(
                ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            # Transient nvidia-smi stall under cluster load: degrade to an empty
            # reading instead of killing the whole sweep. Callers treat an empty
            # utilization map as "no occupancy info this tick"; a real co-run
            # conflict is still caught by the per-PID interference check.
            return []
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]

    def _all_gpus(self) -> list[tuple[str, str]]:
        rows = self._nvidia_smi(["--query-gpu=index,uuid"])
        result = []
        for line in rows:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                result.append((parts[0], parts[1]))
        return result

    def _busy_indices(self) -> set[str]:
        """GPU indices with at least one compute-app PID (anyone's). Kept for
        the informational startup banner only — selection uses _occupied_indices
        (utilization), since a PID may just be parking idle VRAM."""
        rows = self._nvidia_smi(["--query-compute-apps=gpu_uuid"])
        busy_uuids = {row for row in rows if row}
        return {idx for idx, uuid in self._all_gpus() if uuid in busy_uuids}

    def _utils(self) -> dict[str, float]:
        """Map GPU index -> current utilization.gpu (percent)."""
        rows = self._nvidia_smi(["--query-gpu=index,utilization.gpu"])
        out: dict[str, float] = {}
        for line in rows:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        return out

    def _mem_used_pct(self) -> dict[str, float]:
        """Map GPU index -> compute-app used_memory / memory.total (percent)."""
        gpus = self._nvidia_smi(["--query-gpu=index,uuid,memory.total"])
        uuid_to_idx: dict[str, str] = {}
        total_by_idx: dict[str, float] = {}
        for line in gpus:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    uuid_to_idx[parts[1]] = parts[0]
                    total_by_idx[parts[0]] = float(parts[2])
                except ValueError:
                    pass
        used_by_idx: dict[str, float] = {idx: 0.0 for idx in total_by_idx}
        rows = self._nvidia_smi(["--query-compute-apps=gpu_uuid,used_memory"])
        for line in rows:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                idx = uuid_to_idx.get(parts[0])
                if idx is None:
                    continue
                try:
                    used_by_idx[idx] += float(parts[1])
                except ValueError:
                    pass
        out: dict[str, float] = {}
        for idx, used in used_by_idx.items():
            total = total_by_idx.get(idx, 0.0)
            out[idx] = 100.0 * used / total if total > 0 else 0.0
        return out

    def _occupied_indices(self) -> set[str]:
        """GPU indices over the configured SM or memory threshold."""
        util_busy = {idx for idx, u in self._utils().items() if u > self.util_threshold}
        mem_busy = {idx for idx, m in self._mem_used_pct().items() if m > self.mem_threshold}
        return util_busy | mem_busy

    def total_visible(self) -> int:
        gpus = self._all_gpus()
        if self._allowed is not None:
            gpus = [g for g in gpus if g[0] in self._allowed]
        return len(gpus)

    def acquire_many(self, count: int) -> tuple[str, ...]:
        """Block until ``count`` GPUs are free and claim them atomically.

        Selection re-queries nvidia-smi on every loop iteration so busy cards
        can become free. No partial claim is retained while waiting, avoiding
        deadlocks between concurrent multi-GPU workloads. Larger waiting claims
        are served first so a stream of single-GPU jobs cannot starve them.
        """
        if type(count) is not int or count < 1:
            raise ValueError(f"GPU count must be a positive integer, got {count!r}")
        token = object()
        with self._lock:
            self._waiters.append((token, count))
        try:
            while True:
                with self._lock:
                    largest_count = max(waiting_count for _, waiting_count in self._waiters)
                    next_token = next(
                        waiting_token
                        for waiting_token, waiting_count in self._waiters
                        if waiting_count == largest_count
                    )
                    if count != largest_count or token is not next_token:
                        pass
                    else:
                        occupied = self._occupied_indices()
                        free: list[str] = []
                        for idx, _uuid in self._all_gpus():
                            if self._allowed is not None and idx not in self._allowed:
                                continue
                            if idx in self._owned or idx in occupied:
                                continue
                            free.append(idx)
                        if len(free) >= count:
                            selected = tuple(sorted(random.sample(free, count), key=int))
                            self._owned.update(selected)
                            self._waiters.remove((token, count))
                            return selected
                time.sleep(POLL_INTERVAL)
        finally:
            with self._lock:
                if (token, count) in self._waiters:
                    self._waiters.remove((token, count))

    def acquire(self) -> str:
        """Backward-compatible single-GPU acquisition."""
        return self.acquire_many(1)[0]

    def release_many(self, indices: tuple[str, ...] | list[str]) -> None:
        with self._lock:
            self._owned.difference_update(indices)

    def release(self, idx: str) -> None:
        self.release_many((idx,))


# ── Tee stdout → run log ─────────────────────────────────────────────────────


class _Tee:
    """Write to multiple streams; flush on every write so the log is live.

    Locks per write so two threads' simultaneous writes don't interleave
    bytes. For atomic *lines*, callers should still hold _log_lock around
    the full print+flush sequence — see log() below.
    """

    def __init__(self, *streams):
        self._streams = streams
        self._lock = threading.Lock()

    def write(self, s):
        with self._lock:
            for st in self._streams:
                st.write(s)
                st.flush()
        return len(s)

    def flush(self):
        with self._lock:
            for st in self._streams:
                st.flush()


# Thread-safe one-liner emitter. `print()` calls file.write() multiple times
# (once for the message, once for the trailing newline), so without this
# lock concurrent prints from worker threads can interleave halfway through
# a line. Use log() for any [bench-suite] status print from a worker thread.
_log_lock = threading.Lock()


def log(msg: str) -> None:
    with _log_lock:
        print(msg, flush=True)


# ── GPU probe ────────────────────────────────────────────────────────────────


def probe_gpu(idx: str, timeout: float = 60.0) -> tuple[bool, str]:
    """Run PROBE_SCRIPT on a single GPU. Returns (ok, error_message)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = idx
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PROBE_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {timeout:.0f}s"
    except Exception as e:
        return False, repr(e)
    if proc.returncode == 0 and "PROBE_OK" in proc.stdout:
        return True, ""
    msg = (proc.stderr or proc.stdout).strip().splitlines()
    return False, msg[-1] if msg else f"exit {proc.returncode}"


def detect_usable_gpus(
    candidates: list[str], probe_timeout: float
) -> tuple[set[str], dict[str, str]]:
    """Probe candidates in parallel. Returns (usable_set, failures)."""
    usable: set[str] = set()
    failures: dict[str, str] = {}
    if not candidates:
        return usable, failures
    with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
        futs = {ex.submit(probe_gpu, idx, probe_timeout): idx for idx in candidates}
        for fut in as_completed(futs):
            idx = futs[fut]
            ok, err = fut.result()
            if ok:
                usable.add(idx)
                log(f"[bench-suite]   gpu {idx}: ok")
            else:
                failures[idx] = err
                log(f"[bench-suite]   gpu {idx}: FAIL — {err}")
    return usable, failures


# ── Workload execution ───────────────────────────────────────────────────────


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gpu_uuid_of(idx: str) -> str | None:
    """Look up the UUID for a GPU index via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == idx:
            return parts[1]
    return None


def _pids_on_gpu(uuid: str) -> set[int]:
    """Set of PIDs currently using the given GPU UUID."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        return set()
    pids: set[int] = set()
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1] == uuid:
            try:
                pids.add(int(parts[0]))
            except ValueError:
                pass
    return pids


def _pid_sm_on_gpu(gpu_index: str) -> dict[int, float]:
    """Map PID -> sm-utilization (%) for every compute process on the given
    physical GPU, via `nvidia-smi pmon`.

    This is the signal that separates a neighbor *actively burning the GPU*
    from one merely *parking resident VRAM* at 0% sm — and, crucially, it is
    per-process, so it stays meaningful while our own kernel pegs the
    device-level utilization. A single `pmon -c 1` snapshot is ~0.15s here.

    pmon `-s u` columns: gpu  pid  type  sm  mem  enc  dec  jpg  ofa  command.
    Inactive rows show "-" for pid/sm; those are skipped.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "pmon", "-i", str(gpu_index), "-c", "1", "-s", "u"],
            capture_output=True,
            text=True,
            timeout=8,
        ).stdout
    except Exception:
        return {}
    result: dict[int, float] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            pid = int(fields[1])
            sm = float(fields[3])
        except ValueError:
            continue  # pid or sm is "-" (no active process this sample)
        result[pid] = sm
    return result


def _active_strangers(gpu_index: str, our_pids: set[int], sm_threshold: float) -> dict[int, float]:
    """PIDs on `gpu_index` that are NOT ours and whose sm-util exceeds threshold.

    Empty result == no neighbor is actively computing right now, so an
    idle-but-resident squatter (sm 0) does not count as interference and we
    are free to share the card."""
    return {
        pid: sm
        for pid, sm in _pid_sm_on_gpu(gpu_index).items()
        if pid not in our_pids and sm > sm_threshold
    }


def _active_strangers_on_gpus(
    gpu_indices: tuple[str, ...], our_pids: set[int], sm_threshold: float
) -> dict[int, float]:
    """Merge active intruders observed on any GPU assigned to one workload."""
    active: dict[int, float] = {}
    for gpu_index in gpu_indices:
        for pid, sm in _active_strangers(gpu_index, our_pids, sm_threshold).items():
            active[pid] = max(active.get(pid, 0.0), sm)
    return active


class _BenchPidRegistry:
    """Bench subprocess PIDs spawned by this orchestrator (register at Popen)."""

    _lock = threading.Lock()
    _roots: ClassVar[set[int]] = set()
    _our_pids_cache: ClassVar[tuple[float, set[int]] | None] = None

    @classmethod
    def register(cls, pid: int) -> None:
        with cls._lock:
            cls._roots.add(pid)
            cls._our_pids_cache = None

    @classmethod
    def unregister(cls, pid: int) -> None:
        with cls._lock:
            cls._roots.discard(pid)
            cls._our_pids_cache = None


def _proc_children_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            pid = int(entry)
            with open(f"/proc/{entry}/stat") as f:
                data = f.read()
            rparen = data.rfind(")")
            fields = data[rparen + 2 :].split()
            ppid = int(fields[1])
            children.setdefault(ppid, []).append(pid)
        except (OSError, ValueError, IndexError):
            continue
    return children


def _descendants_of(roots: set[int]) -> set[int]:
    if not roots:
        return set()
    children = _proc_children_map()
    out: set[int] = set()
    stack = list(roots)
    while stack:
        p = stack.pop()
        for c in children.get(p, ()):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


_OUR_PIDS_TTL = 0.25  # seconds; amortize /proc walks across pmon polls


def _our_pids() -> set[int]:
    """Orchestrator + every registered bench subprocess and its descendants."""
    now = time.time()
    with _BenchPidRegistry._lock:
        hit = _BenchPidRegistry._our_pids_cache
        if hit is not None and now - hit[0] < _OUR_PIDS_TTL:
            return hit[1]
        roots = set(_BenchPidRegistry._roots)
    pids = {os.getpid()} | roots | _descendants_of(roots)
    with _BenchPidRegistry._lock:
        _BenchPidRegistry._our_pids_cache = (now, pids)
    return pids


def _reap_subprocess(proc: subprocess.Popen) -> None:
    """Ensure the child is reaped so it cannot linger as a zombie holding VRAM."""
    try:
        proc.wait(timeout=0)
    except subprocess.TimeoutExpired:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    except ChildProcessError:
        pass


def _run_subprocess_monitored(
    cmd: list[str],
    env: dict[str, str],
    cwd: str,
    log_path: Path,
    gpu_indices: tuple[str, ...],
    monitor_interval: float,
    sm_threshold: float,
) -> tuple[int, bool, list[int]]:
    """Spawn ``cmd`` on assigned GPUs and watch all of them for active intruders.

    Returns (returncode, interfered, intruder_pids).

    Interference == another tenant is actually computing on an assigned card, i.e. a
    PID that is not registered as ours has sm-utilization above `sm_threshold`.

    Two-stage protection, both using per-PID sm-util (`nvidia-smi pmon`):

    1. **Pre-spawn check**: if any stranger is already actively computing,
       someone grabbed an assigned card between pool acquisition and now (or
       an idle-looking card just woke up). Don't launch — return INTERFERED so
       the dispatcher requeues this workload.

    2. **Per-poll check**: at every `monitor_interval`, take the per-PID sm
       map, drop registered bench PIDs (and their descendants, e.g. nvcc),
       and if any remaining PID is above the sm threshold, SIGTERM the
       subprocess. This catches a brand-new intruder *and* a resident neighbor
       that bursts its own sm mid-run — per-PID sm stays meaningful even while
       our own kernel pegs the device-level utilization.
    """
    proc: subprocess.Popen | None = None
    registered_pid: int | None = None
    intruders: list[int] = []
    if gpu_indices:
        pre = _active_strangers_on_gpus(gpu_indices, _our_pids(), sm_threshold)
        if pre:
            with open(log_path, "w") as lf:
                lf.write(f"RACE_LOST: pre-spawn check — active strangers {pre}\n")
            return -1, True, sorted(pre)
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    registered_pid = proc.pid
    _BenchPidRegistry.register(registered_pid)
    try:
        while True:
            try:
                proc.wait(timeout=monitor_interval)
                break  # subprocess exited normally
            except subprocess.TimeoutExpired:
                pass
            if not gpu_indices:
                continue
            active = _active_strangers_on_gpus(gpu_indices, _our_pids(), sm_threshold)
            if active:
                intruders = sorted(active)
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    finally:
        if registered_pid is not None:
            _BenchPidRegistry.unregister(registered_pid)
        if proc is not None:
            _reap_subprocess(proc)
    return proc.returncode, bool(intruders), intruders


def run_one(
    workload: dict,
    pool: GpuPool,
    log_dir: Path,
    *,
    attempt: int = 1,
    rounds: int = 1,
    round_cooldown: float = DEFAULT_ROUND_COOLDOWN_S,
    bench_aggregate: str = "mean",
) -> dict:
    kernel = workload["kernel"]
    config = workload["config"]
    warmup = workload.get("warmup")
    repeat = workload.get("repeat")
    timer = workload.get("timer")
    num_gpus = workload.get("num_gpus", 1)

    gpus = pool.acquire_many(num_gpus)
    gpu_csv = ",".join(gpus)
    json_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    json_tmp.close()
    log_path = log_dir / f"{kernel}__{config}__a{attempt}.log"

    cmd = [
        sys.executable,
        "-m",
        "tirx_kernels.bench",
        "--kernel",
        kernel,
        "--config",
        config,
        "--json-file",
        json_tmp.name,
        "--rounds",
        str(rounds),
        "--round-cooldown",
        str(round_cooldown),
    ]
    if warmup is not None:
        cmd += ["--warmup", str(warmup)]
    if repeat is not None:
        cmd += ["--repeat", str(repeat)]
    if timer is not None:
        cmd += ["--timer", timer]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_csv

    # Each workload gets its own scratch cwd so concurrent runs don't race on
    # proton's <proton_name>.hatchet file.
    workdir = tempfile.mkdtemp(prefix=f"bench-suite-{kernel}-{config}-")

    label = f"{kernel}/{config}"
    worker = threading.current_thread().name
    started = now_iso()
    record: dict = {
        "kernel": kernel,
        "config": config,
        "gpu": gpu_csv,
        "gpus": list(gpus),
        "num_gpus": num_gpus,
        "started_at": started,
    }
    interfered = False
    intruder_pids: list[int] = []
    try:
        log(f"[bench-suite] {started} {worker} gpus={gpu_csv} START {label} (attempt {attempt})")
        # Pass every physical GPU index; the monitor uses per-PID sm-util (pmon).
        returncode, interfered, intruder_pids = _run_subprocess_monitored(
            cmd, env, workdir, log_path, gpus, MONITOR_INTERVAL, pool.util_threshold
        )
        if interfered:
            record["status"] = "INTERFERED"
            record["intruder_pids"] = intruder_pids
            record["error"] = f"gpus {gpu_csv}: intruder PIDs {intruder_pids}"
        elif returncode != 0:
            tail = "\n".join(log_path.read_text().splitlines()[-30:])
            record["status"] = "FAIL"
            record["error"] = f"exit {returncode}\n{tail}"
        else:
            payload = json.loads(Path(json_tmp.name).read_text())
            rows = payload.get("results") or []
            match = next(
                (r for r in rows if r.get("kernel") == kernel and r.get("label") == config), None
            )
            if match is None:
                record["status"] = "FAIL"
                record["error"] = f"no matching row in bench JSON ({len(rows)} rows)"
            else:
                st = match.get("status")
                if st == "SKIP":
                    record.update(match)
                elif st == "FAIL":
                    record.update(match)
                    record.setdefault("status", "FAIL")
                else:
                    _finalize_bench_record(match, rounds=rounds, bench_aggregate=bench_aggregate)
                    record.update(match)
                    record.setdefault("label", config)
                    if record.get("status") != "ok":
                        record["error"] = match.get("error", "bench finalize failed")
    except Exception as e:
        record["status"] = "FAIL"
        record["error"] = repr(e)
    finally:
        try:
            os.unlink(json_tmp.name)
        except FileNotFoundError:
            pass
        shutil.rmtree(workdir, ignore_errors=True)
        pool.release_many(gpus)

    record["finished_at"] = now_iso()
    status = record.get("status", "ok")
    impls = record.get("impls") or {}
    impl_str = ", ".join(f"{k}={v:.3f}µs" for k, v in impls.items())
    if interfered:
        # Make INTERFERED stand out — easy to spot when scrolling.
        log("[bench-suite] " + "*" * 70)
        log(f"[bench-suite] *** INTERFERED *** {worker} gpus={gpu_csv} {label} attempt {attempt}")
        log(f"[bench-suite] ***   intruder PIDs on gpus {gpu_csv}: {intruder_pids}")
        log("[bench-suite] ***   subprocess killed, will retry until ok")
        log("[bench-suite] " + "*" * 70)
    else:
        log(
            f"[bench-suite] {record['finished_at']} {worker} gpus={gpu_csv} "
            f"{status:4s} {label} {impl_str}"
        )
    return record


# ── Output ───────────────────────────────────────────────────────────────────


def git_label(repo: Path) -> str | None:
    if not repo.exists():
        return None
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if not sha:
            return None
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def _tir_repo_root() -> Path | None:
    """TVM git root: TVM_PATH env, else installed tvm package checkout."""
    env = os.environ.get("TVM_PATH")
    if env:
        p = Path(env).resolve()
        if (p / "python" / "tvm").is_dir():
            return p
    return _module_repo_root("tvm")


def _module_repo_root(import_name: str) -> Path | None:
    """Git root of an importable package, if it's a local checkout."""
    try:
        mod = __import__(import_name)
    except Exception:
        return None
    pkg_file = getattr(mod, "__file__", None)
    if not pkg_file:
        try:
            paths = list(getattr(mod, "__path__", []) or [])
            if paths:
                pkg_file = str(Path(paths[0]) / "__init__.py")
        except Exception:
            pass
    if not pkg_file:
        return None
    for p in [Path(pkg_file).resolve().parent, *Path(pkg_file).resolve().parents]:
        if (p / ".git").exists():
            return p
    return None


def collect_repo_git() -> dict[str, str | None]:
    """SHAs for the three repos involved: tvm, tirx-kernels, tirx-bench-ci."""
    tir_root = _tir_repo_root()
    tirx_root = _module_repo_root("tirx_kernels") or _kernels_repo_root()
    bench_ci_root: Path | None = None
    for base in (tirx_root, tir_root):
        if base is None:
            continue
        candidate = base.parent / "tirx-bench-ci"
        if (candidate / ".git").exists():
            bench_ci_root = candidate
            break
    return {
        "tir": git_label(tir_root) if tir_root else None,
        "tirx-kernels": git_label(tirx_root) if tirx_root else None,
        "tirx-bench-ci": git_label(bench_ci_root) if bench_ci_root else None,
    }


def collect_kernel_fingerprint() -> dict[str, str | None]:
    """Merge-stable content fingerprints (git *tree* SHAs) of the source that
    determines kernel codegen + perf.

    The commit SHAs in ``collect_repo_git`` are rewritten by a squash/rebase
    merge, so a baseline that records only commit SHAs can't be mapped back to a
    mainline commit afterwards. A git tree SHA is content-addressed (Merkle): it
    is identical before and after a merge as long as the directory's content is
    unchanged. Confirm a checkout matches a recorded baseline with
    ``git rev-parse HEAD:<path>``.
    """
    tir_root = _tir_repo_root()
    tirx_root = _module_repo_root("tirx_kernels") or _kernels_repo_root()

    def _tree(root: Path | None, path: str) -> str | None:
        if root is None:
            return None
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "rev-parse", f"HEAD:{path}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        return out.stdout.strip() or None

    return {
        "tir:python/tvm/tirx": _tree(tir_root, "python/tvm/tirx"),
        "tirx-kernels:tirx_kernels": _tree(tirx_root, "tirx_kernels"),
    }


# Packages used as baselines in workloads.yaml — anything our regression
# numbers compare against, so the recorded version pins the comparison.
BASELINE_PACKAGES = ["torch", "deep_gemm", "flashinfer", "flash_attn", "sglang", "cutlass"]


def package_provenance(import_name: str) -> dict | None:
    """Probe a Python package: version + (if editable git install) repo + SHA.

    Returns None when neither the package nor distribution metadata exists.
    """

    def _record_git(path: Path, info: dict) -> None:
        try:
            root = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            if not root:
                return
            sha = subprocess.run(
                ["git", "-C", root, "rev-parse", "--short=8", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            if not sha:
                return
            dirty = subprocess.run(
                ["git", "-C", root, "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            info["git_dir"] = root
            info["git_sha"] = sha + ("-dirty" if dirty else "")
        except Exception:
            pass

    dists: list[str] = []
    try:
        from importlib.metadata import distribution as _probe_dist

        _probe_dist(import_name)
        dists.append(import_name)
    except Exception:
        pass
    try:
        from importlib.metadata import packages_distributions

        for dist_name in packages_distributions().get(import_name) or []:
            if dist_name not in dists:
                dists.append(dist_name)
    except Exception:
        pass
    if not dists:
        dists = [import_name]

    mod = None
    try:
        mod = __import__(import_name)
    except Exception:
        pass
    info: dict = {"importable": mod is not None}
    # Heavy optional baselines can fail during their top-level import even when
    # their source checkout is discoverable on PYTHONPATH. Resolve the module
    # spec without executing it so provenance still records that checkout.
    try:
        spec = find_spec(import_name)
    except Exception:
        spec = None
    if spec is not None:
        source_dir = None
        if spec.origin and spec.origin not in ("built-in", "frozen"):
            source_dir = Path(spec.origin).resolve().parent
        elif spec.submodule_search_locations:
            source_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
        if source_dir is not None:
            info.setdefault("source_dir", str(source_dir))
            _record_git(source_dir, info)
    # Version: prefer __version__, else importlib.metadata. Top-level import
    # name and the distribution name often disagree (e.g. flash_attn ↔
    # flash-attn-4) — use packages_distributions() to bridge.
    version = getattr(mod, "__version__", None) if mod is not None else None
    if version is None:
        try:
            from importlib.metadata import version as _meta_version

            for d in dists:
                try:
                    version = _meta_version(d)
                    if version is not None:
                        info["dist"] = d
                        break
                except Exception:
                    continue
        except Exception:
            pass
    if version is not None:
        info["version"] = str(version)
    if import_name == "torch":
        cuda = getattr(getattr(mod, "version", None), "cuda", None)
        git_v = getattr(getattr(mod, "version", None), "git_version", None)
        if cuda:
            info["cuda"] = str(cuda)
        if git_v:
            info["torch_git_version"] = str(git_v)
    # PEP 610 direct_url.json: when a package was `pip install -e <path>` or
    # `pip install <path>`, pip writes the source path/URL into the dist-info.
    # This catches the editable case (the package lives outside the repo it
    # was built from, so the __file__ walk below misses it). dist resolution:
    # prefer `info["dist"]` if we set it above, else default to import_name.
    try:
        from importlib.metadata import distribution as _meta_dist

        dist = None
        for dist_name in [info.get("dist"), *dists, import_name]:
            if not dist_name:
                continue
            try:
                dist = _meta_dist(dist_name)
                info.setdefault("dist", dist.metadata["Name"])
                break
            except Exception:
                continue
        if dist is not None:
            direct_url_text = dist.read_text("direct_url.json")
            if direct_url_text:
                direct = json.loads(direct_url_text)
                url = direct.get("url") or ""
                if url.startswith("file://"):
                    src_path = Path(url[len("file://") :]).resolve()
                    info["source_dir"] = str(src_path)
                    if direct.get("dir_info", {}).get("editable"):
                        info["editable"] = True
                    _record_git(src_path, info)
    except Exception:
        pass
    if mod is None:
        return info if "version" in info or "source_dir" in info else None
    # Resolve a directory we can git-probe. Namespace packages and some
    # __init__.py-less namespaces set mod.__file__ to None — fall back to
    # __path__[0] then to a known submodule's file.
    pkg_file = getattr(mod, "__file__", None)
    if not pkg_file:
        try:
            paths = list(getattr(mod, "__path__", []) or [])
            if paths:
                pkg_file = str(Path(paths[0]) / "__init__.py")
        except Exception:
            pass
    if not pkg_file:
        # Last resort: try to import a likely submodule with a real file.
        for sub in (".cute", ".csrc", ".jit_kernels", ".jit"):
            try:
                submod = __import__(import_name + sub, fromlist=["__file__"])
                if getattr(submod, "__file__", None):
                    pkg_file = submod.__file__
                    break
            except Exception:
                continue
    if pkg_file:
        pkg_dir = Path(pkg_file).resolve().parent
        # Walk up looking for a git repo. .git can be a dir (regular clone)
        # or a file (worktree); both are fine for `git rev-parse`.
        _record_git(pkg_dir, info)
    return info


def collect_baseline_provenance() -> dict:
    return {name: package_provenance(name) or {"installed": False} for name in BASELINE_PACKAGES}


def write_run(
    out_dir: Path, stamp: str, results: list[dict], label: str | None, probe: dict | None = None
) -> Path:
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": stamp,
        "label": label,
        "git": collect_repo_git(),
        "kernel_tree": collect_kernel_fingerprint(),
        "baselines": collect_baseline_provenance(),
        "probe": probe or {},
        "results": results,
    }
    path = runs_dir / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


BASELINE_IMPL_BY_KERNEL = {
    "fp16_bf16_gemm": "torch-cublas",
    "fp8_blockwise_gemm": "deepgemm",
    "nvfp4_gemm": "flashinfer",
    "flash_attention4": "flashattn_sm100",
    "deepgemm_sm100_fp8_mqa_logits": "deepgemm",
    "deepgemm_sm100_fp4_mqa_logits": "deepgemm",
    "deepgemm_sm100_fp4_paged_mqa_logits": "deepgemm",
    "deepgemm_sm100_fp8_paged_mqa_logits": "deepgemm",
    "deepgemm_sm100_tf32_hc_prenorm_gemm": "deepgemm",
    "flash_mla_sparse_fwd": "flashmla",
    "sparse_flashmla_prefill_head64_phase1": "flashmla",
    "sparse_flashmla_prefill_head128_phase1": "flashmla",
    "sparse_flashmla_prefill_head128_small_topk_phase1": "flashmla",
}


def _our_impl(row_impls: dict) -> str | None:
    """Pick our impl ('tir' or 'tirx') from a row's impls dict."""
    for name in ("tir", "tirx"):
        if name in row_impls:
            return name
    return None


def write_summary(out_dir: Path, current: dict) -> Path:
    """Human-readable per-run report, grouped by kernel.

    Times are in µs to match the existing bench-suite doc convention. Per row:
    config, one column per impl present in that kernel, baseline/ours ratio
    (against the kernel's reference impl from BASELINE_IMPL_BY_KERNEL),
    then attempt + gpu.
    """
    stamp = current["timestamp"]
    reports_dir = out_dir / "reports" / stamp
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# bench-suite run {stamp}")
    lines.append("")
    label = current.get("label") or "-"
    git = current.get("git") or {}
    lines.append(f"- label: `{label}`")
    lines.append(
        f"- git: tir=`{git.get('tir') or '-'}`  "
        f"tirx-kernels=`{git.get('tirx-kernels') or '-'}`  "
        f"tirx-bench-ci=`{git.get('tirx-bench-ci') or '-'}`"
    )
    statuses: dict[str, int] = {}
    for r in current.get("results") or []:
        s = r.get("status") or "?"
        statuses[s] = statuses.get(s, 0) + 1
    status_line = ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
    lines.append(f"- status: {status_line} (over {sum(statuses.values())} workloads)")
    lines.append("")

    baselines = current.get("baselines") or {}
    if baselines:
        lines.append("## Baseline impl provenance")
        lines.append("")
        for name, info in sorted(baselines.items()):
            if not info or info.get("installed") is False:
                lines.append(f"- `{name}`: not installed")
                continue
            bits = []
            if "version" in info:
                bits.append(f"v{info['version']}")
            if "cuda" in info:
                bits.append(f"cuda={info['cuda']}")
            if "torch_git_version" in info:
                bits.append(f"torch_git={info['torch_git_version'][:12]}")
            if "git_sha" in info:
                bits.append(f"@`{info['git_sha']}`")
            if "git_dir" in info:
                bits.append(f"({info['git_dir']})")
            lines.append(f"- `{name}`: {' '.join(bits) if bits else '?'}")
        lines.append("")

    # Group by kernel
    by_kernel: dict[str, list[dict]] = {}
    for r in current.get("results") or []:
        by_kernel.setdefault(r["kernel"], []).append(r)

    for kernel in sorted(by_kernel):
        rows = sorted(by_kernel[kernel], key=lambda r: r.get("label") or r.get("config") or "")
        # Discover all impl names that appear in this kernel
        impl_names: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for impl in r.get("impls") or {}:
                if impl not in seen:
                    seen.add(impl)
                    impl_names.append(impl)
        impl_names.sort()
        baseline_impl = BASELINE_IMPL_BY_KERNEL.get(kernel)
        # Determine "ours" impl name once for the whole kernel (constant per kernel)
        ours_impl = None
        for r in rows:
            ours_impl = _our_impl(r.get("impls") or {})
            if ours_impl:
                break
        ratio_label = f"{baseline_impl}/{ours_impl}" if baseline_impl and ours_impl else "ratio"
        lines.append(f"## `{kernel}`")
        if baseline_impl and ours_impl:
            lines.append("")
            lines.append(
                f"_baseline impl_: `{baseline_impl}` · _ours_: `{ours_impl}` · "
                f"_ratio_ = baseline/ours · `>1` means ours is faster"
            )
        lines.append("")
        # Table header
        header = ["config", *impl_names, ratio_label, "attempt", "gpus"]
        align = ["---"] + ["---:"] * len(impl_names) + ["---:", "---:", "---:"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(align) + "|")
        for r in rows:
            cfg = r.get("label") or r.get("config") or "?"
            status = r.get("status", "ok")
            impls = r.get("impls") or {}
            row = [cfg]
            for impl in impl_names:
                us = impls.get(impl)
                row.append(f"{us:.2f}" if us is not None else "—")
            # Ratio column
            ratio_cell = "—"
            if baseline_impl and ours_impl:
                base_us = impls.get(baseline_impl)
                ours_us = impls.get(ours_impl)
                if base_us is not None and ours_us is not None and ours_us > 0:
                    ratio = base_us / ours_us
                    # Bold values that flag a regression risk (we're slower)
                    ratio_cell = f"**{ratio:.3f}**" if ratio < 1.0 else f"{ratio:.3f}"
            row.append(ratio_cell)
            if status != "ok":
                row[0] = f"{cfg} **[{status}]**"
            row.append(str(r.get("attempt", 1)))
            row.append(str(r.get("gpu", "-")))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    path = reports_dir / "summary.md"
    path.write_text("\n".join(lines))
    return path


def load_baseline(path=None):
    """Load the pinned baseline.json, or None if no baseline exists yet.

    ``path`` (optional) overrides the default baseline location."""
    p = Path(path) if path is not None else DEFAULT_BASELINE
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ── Main ─────────────────────────────────────────────────────────────────────


def aggregate_impl_times(values: list[float], method: str) -> float:
    """Combine per-round impl times for one workload."""
    import statistics

    if method == "mean":
        return statistics.mean(values)
    if method == "median":
        return statistics.median(values)
    if method == "trimmed_mean":
        if len(values) < 3:
            return statistics.mean(values)
        ranked = sorted(values)
        return statistics.mean(ranked[1:-1])
    raise ValueError(f"aggregate must be mean, median, or trimmed_mean, got {method!r}")


def _finalize_bench_record(row: dict, *, rounds: int, bench_aggregate: str) -> None:
    """Validate in-bench round samples and write aggregated impl times (microseconds)."""
    samples = row.get("round_samples")
    if not samples:
        impls = row.get("impls") or {}
        samples = {k: [v] for k, v in impls.items() if v is not None and v > 0}
    if not samples:
        row["status"] = "FAIL"
        row["error"] = "no round samples in bench JSON"
        return
    bad = {impl: len(vals) for impl, vals in samples.items() if len(vals) != rounds}
    if bad:
        row["status"] = "FAIL"
        row["error"] = f"expected {rounds} round(s) per impl, got {bad}"
        return
    row["impls"] = {
        impl: aggregate_impl_times(vals, bench_aggregate) for impl, vals in samples.items()
    }
    row["aggregated"] = {"rounds": rounds, "method": bench_aggregate}
    row["status"] = "ok"
    row.pop("round_samples", None)


def run_scheduled_jobs(
    workloads: list[dict],
    pool: GpuPool,
    log_dir: Path,
    *,
    rounds: int,
    round_cooldown: float,
    bench_aggregate: str,
    cpu_workers: int,
) -> tuple[list[dict], list[tuple[str, str, int, str]]]:
    """Run one subprocess job per workload; failed jobs go back on the queue."""
    n_jobs = len(workloads)
    if not n_jobs:
        return [], []

    pending: queue.Queue[tuple[dict, int] | None] = queue.Queue()
    for w in workloads:
        pending.put((w, 1))

    records: list[dict] = []
    retry_log: list[tuple[str, str, int, str]] = []
    state_lock = threading.Lock()
    done_cv = threading.Condition(state_lock)
    n_done = 0

    def worker() -> None:
        nonlocal n_done
        while True:
            with state_lock:
                if n_done >= n_jobs:
                    return
            try:
                item = pending.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                pending.task_done()
                return

            workload, attempt = item
            kernel = workload["kernel"]
            config = workload["config"]
            record = run_one(
                workload,
                pool,
                log_dir,
                attempt=attempt,
                rounds=rounds,
                round_cooldown=round_cooldown,
                bench_aggregate=bench_aggregate,
            )
            if record.get("status") in ("ok", "SKIP"):
                record["attempt"] = attempt
                with state_lock:
                    records.append(record)
                    n_done += 1
                    done_cv.notify_all()
            else:
                reason = record.get("status", "FAIL")
                detail = record.get("error") or ""
                if record.get("intruder_pids"):
                    detail = f"intruders {record['intruder_pids']}"
                with state_lock:
                    retry_log.append((kernel, config, attempt, detail[:240]))
                log(
                    f"[bench-suite] >>> REQUEUE {kernel}/{config} "
                    f"attempt {attempt} ({reason}): {detail[:160]} <<<"
                )
                pending.put((workload, attempt + 1))
            pending.task_done()

    n_workers = min(cpu_workers, n_jobs)
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="bench") as ex:
        futs = [ex.submit(worker) for _ in range(n_workers)]
        with state_lock:
            while n_done < n_jobs:
                done_cv.wait(timeout=1.0)
        for _ in range(n_workers):
            pending.put(None)
        for fut in as_completed(futs):
            fut.result()
    return records, retry_log


def main() -> None:
    ap = argparse.ArgumentParser(description="bench-suite: pre-commit regression benchmark")
    ap.add_argument(
        "--workloads",
        type=Path,
        default=DEFAULT_WORKLOADS,
        help="YAML file listing kernels/configs to bench",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where to store runs/, logs/, reports/, latest.json "
        "(default: <tirx-kernels>/.bench-suite)",
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline JSON to diff against instead of the pinned baseline.json",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_REGRESSION_THRESHOLD,
        help=f"Regression threshold in percent slowdown (default {DEFAULT_REGRESSION_THRESHOLD:g})",
    )
    ap.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only keep workloads whose kernel contains this substring",
    )
    # NOTE: there is intentionally no --gpus flag. GPU selection is automatic
    # (util-gated probe + per-acquire utilization scan); a human pinning cards
    # defeats that and can land work on a busy card. See acquire()/_occupied_indices.
    ap.add_argument(
        "--label",
        type=str,
        default=None,
        help="Free-form label for this run (default: git short sha)",
    )
    ap.add_argument("--no-report", action="store_true", help="Skip regression report generation")
    ap.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the per-GPU probe (use nvidia-smi free-status only)",
    )
    ap.add_argument(
        "--probe-timeout",
        type=float,
        default=60.0,
        help="Per-GPU probe timeout in seconds (default 60)",
    )
    ap.add_argument(
        "--util-threshold",
        type=float,
        default=DEFAULT_UTIL_THRESHOLD,
        help="%% GPU/sm utilization above which a card counts as "
        "actively in use: selection skips such cards and the "
        "monitor requeues if a neighbor crosses it mid-run "
        f"(default {DEFAULT_UTIL_THRESHOLD:g})",
    )
    ap.add_argument(
        "--mem-threshold",
        type=float,
        default=DEFAULT_MEM_THRESHOLD,
        help="%% GPU memory used by compute apps above which a card counts as occupied "
        f"(default {DEFAULT_MEM_THRESHOLD:g})",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="In-bench rounds per workload subprocess (default 1). Compile once, "
        "then warmup+repeat each round; failed jobs are requeued until ok.",
    )
    ap.add_argument(
        "--round-cooldown",
        type=float,
        default=DEFAULT_ROUND_COOLDOWN_S,
        help=f"Seconds to sleep between in-bench rounds (default {DEFAULT_ROUND_COOLDOWN_S:g}).",
    )
    ap.add_argument(
        "--bench-aggregate",
        choices=("mean", "median", "trimmed_mean"),
        default="mean",
        help="How to combine in-bench round samples per impl (default mean). "
        "trimmed_mean drops the fastest and slowest round.",
    )
    ap.add_argument(
        "--cpu-workers",
        type=int,
        default=DEFAULT_CPU_WORKERS,
        help="Concurrent bench workers (default 0 = probe-OK GPU count). "
        "Each worker atomically holds all GPUs requested by its workload.",
    )
    ap.add_argument(
        "--check-imports",
        action="store_true",
        help="Import every unique kernel in --workloads and exit (for CI import gates)",
    )
    args = ap.parse_args()
    if args.rounds < 1:
        print("[bench-suite] --rounds must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.round_cooldown < 0:
        print("[bench-suite] --round-cooldown must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.util_threshold < 0 or args.mem_threshold < 0:
        print("[bench-suite] --util-threshold/--mem-threshold must be >= 0", file=sys.stderr)
        sys.exit(2)

    workloads = load_workloads(args.workloads)
    if args.filter:
        workloads = [w for w in workloads if args.filter in w["kernel"]]
    if not workloads:
        print("[bench-suite] no workloads to run.", file=sys.stderr)
        sys.exit(2)

    if args.check_imports:
        from tirx_kernels.registry import check_workload_imports

        names = check_workload_imports(workloads, strict=True)
        print(f"[bench-suite] import check ok ({len(names)} kernels from {args.workloads})")
        return

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Run id: incrementing integer — one more than the highest existing numeric
    # run in runs/ (runs/7.json, reports/7/, latest -> 7).
    _existing = [int(p.stem) for p in runs_dir.glob("*.json") if p.stem.isdigit()]
    stamp = str(max(_existing, default=0) + 1)
    run_log_path = runs_dir / f"{stamp}.log"
    run_log_fh = open(run_log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, run_log_fh)
    sys.stderr = _Tee(sys.stderr, run_log_fh)
    # Repoint `latest.log` symlink immediately so `tail -f .bench-suite/latest.log`
    # picks up this run before any output happens.
    latest_log = out_dir / "latest.log"
    if latest_log.exists() or latest_log.is_symlink():
        latest_log.unlink()
    latest_log.symlink_to(run_log_path.relative_to(out_dir))

    print(f"[bench-suite] live log: {run_log_path}")
    print(f"[bench-suite]   tail : tail -f {latest_log}")
    print(f"[bench-suite] run id : {stamp}")

    # ── Automatic GPU selection (no manual override on purpose) ──
    # 1. Startup probe: run a tiny fp16 matmul on every visible card
    #    (including busy ones — the probe is light, finishes fine on a
    #    contended card; this catches broken drivers / ECC). Probe failures
    #    are banned for the rest of the run.
    # 2. Per-workload acquire: re-scan utilization/memory every time we need a card.
    listing_pool = GpuPool(util_threshold=args.util_threshold, mem_threshold=args.mem_threshold)
    in_filter = [idx for idx, _ in listing_pool._all_gpus()]
    if not in_filter:
        print("[bench-suite] no visible GPUs.", file=sys.stderr)
        sys.exit(1)
    utils_now = listing_pool._utils()
    mem_now = listing_pool._mem_used_pct()
    occupied_now = sorted(listing_pool._occupied_indices() & set(in_filter), key=int)
    resident = sorted(listing_pool._busy_indices() & set(in_filter), key=int)
    util_str = " ".join(f"{i}:{utils_now.get(i, 0):.0f}%" for i in sorted(in_filter, key=int))
    mem_str = " ".join(f"{i}:{mem_now.get(i, 0):.1f}%" for i in sorted(in_filter, key=int))
    print(
        f"[bench-suite] visible: {len(in_filter)} {sorted(in_filter, key=int)}; "
        f"util now [{util_str}]; mem now [{mem_str}]",
        flush=True,
    )
    print(
        f"[bench-suite] gate: util-threshold={args.util_threshold:g}%, "
        f"mem-threshold={args.mem_threshold:g}% — "
        f"occupied (skip): {occupied_now if occupied_now else 'none'}; "
        f"shareable: "
        f"{sorted((set(in_filter) - set(occupied_now)), key=int)} "
        f"(resident-VRAM cards: {resident if resident else 'none'})",
        flush=True,
    )

    if args.no_probe:
        usable = set(in_filter)
        probe_failures: dict[str, str] = {}
    else:
        print(
            f"[bench-suite] probing {len(in_filter)} GPU(s) with fp16 512x512 matmul ...",
            flush=True,
        )
        usable, probe_failures = detect_usable_gpus(in_filter, args.probe_timeout)

    if not usable:
        print("[bench-suite] no usable GPUs (all probes failed).", file=sys.stderr)
        for idx, err in probe_failures.items():
            print(f"[bench-suite]   gpu {idx}: {err}", file=sys.stderr)
        sys.exit(1)

    max_required_gpus = max(workload.get("num_gpus", 1) for workload in workloads)
    if max_required_gpus > len(usable):
        print(
            f"[bench-suite] workload requires {max_required_gpus} GPU(s), but only "
            f"{len(usable)} passed the startup probe.",
            file=sys.stderr,
        )
        sys.exit(2)

    pool = GpuPool(
        allowed=usable, util_threshold=args.util_threshold, mem_threshold=args.mem_threshold
    )
    n_gpus = len(usable)
    cpu_workers = args.cpu_workers if args.cpu_workers > 0 else n_gpus
    cpu_workers = min(cpu_workers, n_gpus)

    _repo_git = collect_repo_git()
    label = args.label or _repo_git.get("tirx-kernels") or _repo_git.get("tir") or "local"
    agg_note = (
        f", {args.rounds} in-bench round(s), aggregate={args.bench_aggregate}, "
        f"round_cooldown={args.round_cooldown:g}s"
        if args.rounds > 1 or args.round_cooldown > 0
        else ""
    )
    print(
        f"[bench-suite] {len(workloads)} workloads, {n_gpus} probe-OK GPU(s) in pool, "
        f"{cpu_workers} worker(s), label={label}{agg_note}",
        flush=True,
    )

    results, retry_log = run_scheduled_jobs(
        workloads,
        pool,
        log_dir,
        rounds=args.rounds,
        round_cooldown=args.round_cooldown,
        bench_aggregate=args.bench_aggregate,
        cpu_workers=cpu_workers,
    )

    if retry_log:
        log(f"[bench-suite] requeue summary: {len(retry_log)} failed attempt(s) before success")
        for k, c, att, detail in retry_log:
            log(f"[bench-suite]   - {k}/{c}: attempt {att} → {detail}")
    else:
        log("[bench-suite] requeue summary: none (every job succeeded on first try)")

    results.sort(key=lambda r: (r["kernel"], r.get("label") or r.get("config")))
    probe_meta = {"enabled": not args.no_probe, "usable": sorted(usable), "failed": probe_failures}
    run_path = write_run(out_dir, stamp, results, label, probe=probe_meta)
    current = json.loads(run_path.read_text())

    latest = out_dir / "latest.json"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_path.relative_to(out_dir))

    summary_path = write_summary(out_dir, current)
    print(f"[bench-suite] wrote {run_path}")
    print(f"[bench-suite] wrote {summary_path}")

    if args.no_report:
        return

    # Single pinned baseline (baseline.json). Promote a fresh run over it via
    # promote_baseline.py.
    baseline = load_baseline(args.baseline)
    if baseline is None:
        print("[bench-suite] no baseline (baseline.json) — skipping regression report")
        print(f"[bench-suite]   set baseline: promote_baseline.py {run_path} --merge")
        return

    reports_dir = out_dir / "reports" / current["timestamp"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    # keep reports/latest pointing at the most recent run's folder
    reports_latest = out_dir / "reports" / "latest"
    if reports_latest.exists() or reports_latest.is_symlink():
        reports_latest.unlink()
    reports_latest.symlink_to(current["timestamp"])

    sys.path.insert(0, str(SCRIPT_DIR))
    from ratio_diff import build_report as _build_bench_report

    try:
        bench_md, n_regress = _build_bench_report(baseline, current, threshold_pct=args.threshold)
    except Exception as e:
        print(f"[bench-suite] bench report failed: {e}", file=sys.stderr)
        sys.exit(3)

    bench_path = reports_dir / "bench.md"
    bench_path.write_text(bench_md)
    print(f"[bench-suite] wrote {bench_path}\n")
    print(bench_md)

    if n_regress > 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
