#!/usr/bin/env python3
"""tir-bench: pre-commit regression benchmark for TIRx kernels.

See README.md in this directory for setup, baseline workflow, and flags.

Quick start:
    python run.py --impls baseline --rounds 5 --restable-reps 0
    python reaggregate_from_logs.py --ref
    python run.py --impls ours
    python promote_baseline.py .tir-bench/runs/<id>.json --tir

See README.md for the full strategy.

Exit codes:
    0  no regressions (or no baseline yet)
    2  config error (no workloads / bad YAML)
    3  one or more regressions exceeded the threshold
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def _kernels_repo_root() -> Path:
    """Git root of the tirx-kernels repo (parent of the tirx_kernels package)."""
    return SCRIPT_DIR.parent.parent


DEFAULT_OUT_DIR = _kernels_repo_root() / ".tir-bench"
DEFAULT_WORKLOADS = SCRIPT_DIR / "workloads.yaml"
# Two pinned baselines with independent update cadences: our kernels (refreshed
# by `--impls ours`) and external references (refreshed by `--impls baseline`).
# The diff joins them at read time — there is no combined baseline file.
DEFAULT_TIR_BASELINE = SCRIPT_DIR / "tir.json"
DEFAULT_REF_BASELINE = SCRIPT_DIR / "ref.json"
DEFAULT_RATIO_BASELINE = SCRIPT_DIR / "ratio.json"
DEFAULT_REGRESSION_THRESHOLD = 1.0
DEFAULT_RESTABLE_THRESHOLD = (
    3.0  # only re-bench movers past this; sub-3% drifts are usually contention noise
)
OURS_IMPLS = frozenset(
    {"tir", "tirx"}
)  # our own kernel labels; keep in sync with ratio_diff.OUR_IMPLS
POLL_INTERVAL = 5.0  # seconds between GPU re-checks when none is free
MONITOR_INTERVAL = 0.5  # seconds between nvidia-smi polls during a workload
MAX_INTERFERED_RETRIES = 5  # workloads that hit INTERFERED get requeued up to this many times
DEFAULT_UTIL_THRESHOLD = 10.0  # % GPU util at/above which a card counts as "actively computing"
# Why util, not PID-presence: on shared boxes other tenants routinely *park*
# processes that hold tens-to-hundreds of GiB of VRAM at 0% utilization. They
# aren't competing for SMs, so co-running our bench on such a card is fine.
# Gating on "any compute-app PID present" would reject every such card and
# starve the sweep; gating on utilization lets us share idle-but-resident cards
# while still avoiding cards where a neighbor is actually burning the GPU.

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
        out.append({**defaults, **entry})
    return out


# ── GPU pool ─────────────────────────────────────────────────────────────────


class GpuPool:
    """Hand out free GPU indices to worker threads.

    Every acquire() re-queries nvidia-smi utilization to decide who is free
    right now: a card counts as taken only if its GPU utilization is at/above
    `util_threshold` (someone is actively computing) — a card merely *holding*
    VRAM at 0% util is fair game to co-run on. So a GPU that was pegged at
    sweep start and went idle later is reusable the moment its util drops. The
    broken-card probe is a separate startup step; by the time the pool is
    built, `allowed` already excludes broken cards.
    """

    def __init__(
        self,
        allowed: set[str] | None = None,
        util_threshold: float = DEFAULT_UTIL_THRESHOLD,
    ):
        self._owned: set[str] = set()
        self._lock = threading.Lock()
        self._allowed = allowed
        self.util_threshold = util_threshold
        self._rr = 0  # round-robin cursor for next_gpu()

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

    def _occupied_indices(self) -> set[str]:
        """GPU indices actively computing (util >= threshold) — i.e. a real
        tenant is burning the GPU, so we should not co-run there. Idle cards
        holding only resident VRAM read ~0% util and are NOT occupied."""
        return {idx for idx, u in self._utils().items() if u >= self.util_threshold}

    def total_visible(self) -> int:
        gpus = self._all_gpus()
        if self._allowed is not None:
            gpus = [g for g in gpus if g[0] in self._allowed]
        return len(gpus)

    def acquire(self) -> str:
        """Block until a free GPU is found; return its index string.

        Re-queries nvidia-smi utilization on every loop iteration so that a
        GPU which was pegged when the previous workload acquired now counts as
        free once the other tenant's util drops below the threshold. A card
        that only holds resident VRAM (0% util) counts as free.
        """
        while True:
            with self._lock:
                occupied = self._occupied_indices()
                for idx, _uuid in self._all_gpus():
                    if self._allowed is not None and idx not in self._allowed:
                        continue
                    if idx in self._owned or idx in occupied:
                        continue
                    self._owned.add(idx)
                    return idx
            time.sleep(POLL_INTERVAL)

    def release(self, idx: str) -> None:
        with self._lock:
            self._owned.discard(idx)

    def next_gpu(self) -> str:
        """Round-robin a card to a workload (non-blocking).

        Under CPU overcommit the per-GPU flock in the subprocess serializes the
        measurement phase, so we don't need exclusive ownership here — just
        spread workloads across cards so each card's flock queue stays balanced.

        We prefer cards with no *foreign* tenant actively computing. A sibling of
        ours measuring on a card (under its flock) does NOT make the card
        off-limits — that's the whole point of overcommit — so we exclude the
        orchestrator's process tree when judging "busy". If every card has a
        foreign tenant we fall back to all cards (the pre-spawn interference
        check + requeue then handle it). Uses the cached pmon, so the per-card
        check is cheap across many concurrent assignments."""
        # Candidate cards come from the probe-OK set fixed at startup, NOT a live
        # nvidia-smi query. _all_gpus() returns [] when nvidia-smi transiently
        # stalls under cluster load (see _nvidia_smi) — and turning that empty
        # reading into "no usable GPUs" would abort the whole sweep, the exact
        # outcome _nvidia_smi degrades gracefully to avoid. GPU indices are stable
        # for the life of the process, so self._allowed is the authoritative pool
        # (guaranteed non-empty — main() exits earlier if every probe failed).
        # Only fall back to a live enumeration when the pool is unrestricted.
        if self._allowed is not None:
            cards = sorted(self._allowed, key=int)
        else:
            cards = sorted((g[0] for g in self._all_gpus()), key=int)
        if not cards:
            raise RuntimeError("no usable GPUs in pool")
        ours = _our_process_tree(os.getpid())
        free = [c for c in cards if not _active_strangers(c, ours, self.util_threshold)]
        pick_from = free or cards
        with self._lock:
            gpu = pick_from[self._rr % len(pick_from)]
            self._rr += 1
            return gpu


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
# a line. Use log() for any [tir-bench] status print from a worker thread.
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
                log(f"[tir-bench]   gpu {idx}: ok")
            else:
                failures[idx] = err
                log(f"[tir-bench]   gpu {idx}: FAIL — {err}")
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


_PMON_CACHE: dict[str, tuple[float, dict[int, float]]] = {}
_PMON_CACHE_LOCK = threading.Lock()
_PMON_TTL = 1.0  # seconds; dedupes pmon across many concurrent monitors under overcommit


def _pid_sm_on_gpu_cached(gpu_index: str) -> dict[int, float]:
    """`_pid_sm_on_gpu` with a short per-GPU TTL cache. Under CPU overcommit many
    monitor threads watch the same handful of cards; without this each would
    fork its own `nvidia-smi pmon` and flood the driver."""
    now = time.time()
    with _PMON_CACHE_LOCK:
        hit = _PMON_CACHE.get(gpu_index)
        if hit is not None and now - hit[0] < _PMON_TTL:
            return hit[1]
    val = _pid_sm_on_gpu(gpu_index)
    with _PMON_CACHE_LOCK:
        _PMON_CACHE[gpu_index] = (now, val)
    return val


def _active_strangers(gpu_index: str, our_pids: set[int], sm_threshold: float) -> dict[int, float]:
    """PIDs on `gpu_index` that are NOT ours and whose sm-util >= threshold.

    Empty result == no neighbor is actively computing right now, so an
    idle-but-resident squatter (sm 0) does not count as interference and we
    are free to share the card."""
    return {
        pid: sm
        for pid, sm in _pid_sm_on_gpu(gpu_index).items()
        if pid not in our_pids and sm >= sm_threshold
    }


def _our_process_tree(root_pid: int) -> set[int]:
    """Set of PIDs in the process tree rooted at root_pid (inclusive).

    Replaces grace-period "ours" accumulation: rather than guessing that
    every PID seen on the GPU within the first N seconds is ours, we
    actually walk the PPID chain via /proc and only call a PID "ours" if
    it's a descendant of our subprocess. Anything else on our GPU = intruder.
    """
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                # /proc/PID/stat: pid (comm) state ppid ...
                # comm can contain spaces & parens, so split from the last ')'
                data = f.read()
            rparen = data.rfind(")")
            fields = data[rparen + 2 :].split()
            ppid = int(fields[1])
            children.setdefault(ppid, []).append(int(entry))
        except (OSError, ValueError, IndexError):
            continue
    ours = {root_pid}
    stack = [root_pid]
    while stack:
        p = stack.pop()
        for c in children.get(p, ()):
            if c not in ours:
                ours.add(c)
                stack.append(c)
    return ours


def _run_subprocess_monitored(
    cmd: list[str],
    env: dict[str, str],
    cwd: str,
    log_path: Path,
    gpu_index: str,
    monitor_interval: float,
    sm_threshold: float,
) -> tuple[int, bool, list[int]]:
    """Spawn `cmd` on the assigned GPU and watch for *active* intruders.

    Returns (returncode, interfered, intruder_pids).

    Interference == another tenant is actually computing on our card, i.e. a
    PID that is not in our process tree has sm-utilization >= `sm_threshold`.
    A neighbor that only parks resident VRAM at 0% sm is NOT interference — we
    deliberately co-run with those (that is the whole point of the util gate).

    Two-stage protection, both using per-PID sm-util (`nvidia-smi pmon`):

    1. **Pre-spawn check**: if any stranger is already actively computing,
       someone grabbed the card between pool.acquire() and now (or an
       idle-looking card just woke up). Don't launch — return INTERFERED so
       the dispatcher requeues this workload.

    2. **Per-poll check**: at every `monitor_interval`, take the per-PID sm
       map, drop our process tree (walked via /proc PPID chain), and if any
       remaining PID is at/above the sm threshold, SIGTERM the subprocess.
       This catches a brand-new intruder *and* a resident neighbor that
       bursts its own sm mid-run — per-PID sm stays meaningful even while our
       own kernel pegs the device-level utilization.
    """
    # "Ours" = the whole orchestrator process tree, not just this one
    # subprocess. Under CPU overcommit many of our bench subprocesses share a
    # physical GPU (serialized by the per-GPU flock), so a *sibling* that is
    # actively measuring would otherwise look like a foreign intruder. Excluding
    # the orchestrator's full descendant set keeps only genuinely-foreign PIDs.
    if gpu_index:
        pre = _active_strangers(gpu_index, _our_process_tree(os.getpid()), sm_threshold)
        if pre:
            with open(log_path, "w") as lf:
                lf.write(f"RACE_LOST: pre-spawn check — active strangers {pre}\n")
            return -1, True, sorted(pre)

    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    intruders: list[int] = []
    try:
        while True:
            try:
                proc.wait(timeout=monitor_interval)
                break  # subprocess exited normally
            except subprocess.TimeoutExpired:
                pass
            if not gpu_index:
                continue
            ours = _our_process_tree(os.getpid())
            active = _active_strangers(gpu_index, ours, sm_threshold)
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
    return proc.returncode, bool(intruders), intruders


def run_one(
    workload: dict,
    pool: GpuPool,
    log_dir: Path,
    *,
    impls_mode: str = "all",
    no_monitor: bool = False,
    log_tag: str | None = None,
) -> dict:
    kernel = workload["kernel"]
    config = workload["config"]
    warmup = workload.get("warmup")
    repeat = workload.get("repeat")
    timer = workload.get("timer")

    gpu = pool.next_gpu()
    started = now_iso()
    label = f"{kernel}/{config}/{impls_mode}"
    worker = threading.current_thread().name
    log(f"[tir-bench] {started} {worker} gpu={gpu} START {label}")

    json_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    json_tmp.close()
    log_path = log_dir / f"{kernel}__{config}__{log_tag or impls_mode}.log"

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
    ]
    if warmup is not None:
        cmd += ["--warmup", str(warmup)]
    if repeat is not None:
        cmd += ["--repeat", str(repeat)]
    if timer is not None:
        cmd += ["--timer", timer]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["TIRX_BENCH_IMPLS"] = impls_mode
    # Serialize the GPU-measurement phase per physical card (the subprocess
    # acquires a per-GPU flock around run_kernel_bench). Lets many subprocesses
    # import + compile in parallel while only one measures per card.
    env["TIR_BENCH_GPU_LOCK"] = "1"

    # Each workload gets its own scratch cwd so concurrent runs don't race on
    # proton's <proton_name>.hatchet file.
    workdir = tempfile.mkdtemp(prefix=f"tir-bench-{kernel}-{config}-")

    record: dict = {
        "kernel": kernel,
        "config": config,
        "gpu": gpu,
        "started_at": started,
    }
    interfered = False
    intruder_pids: list[int] = []
    try:
        # Pass the physical GPU index (not "" ) only when monitoring is on;
        # the monitor uses per-PID sm-util (pmon) keyed by this index.
        monitor_idx = "" if no_monitor else gpu
        returncode, interfered, intruder_pids = _run_subprocess_monitored(
            cmd,
            env,
            workdir,
            log_path,
            monitor_idx,
            MONITOR_INTERVAL,
            pool.util_threshold,
        )
        if interfered:
            record["status"] = "INTERFERED"
            record["intruder_pids"] = intruder_pids
            record["error"] = f"gpu {gpu}: intruder PIDs {intruder_pids}"
        elif returncode != 0:
            tail = "\n".join(log_path.read_text().splitlines()[-30:])
            record["status"] = "FAIL"
            record["error"] = f"exit {returncode}\n{tail}"
        else:
            payload = json.loads(Path(json_tmp.name).read_text())
            rows = payload.get("results") or []
            match = next(
                (r for r in rows if r.get("kernel") == kernel and r.get("label") == config),
                None,
            )
            if match is None:
                record["status"] = "FAIL"
                record["error"] = f"no matching row in bench JSON ({len(rows)} rows)"
            else:
                record.update(match)
                record.setdefault("status", "ok")
    except Exception as e:
        record["status"] = "FAIL"
        record["error"] = repr(e)
    finally:
        try:
            os.unlink(json_tmp.name)
        except FileNotFoundError:
            pass
        shutil.rmtree(workdir, ignore_errors=True)
        pool.release(gpu)

    record["finished_at"] = now_iso()
    status = record.get("status", "ok")
    impls = record.get("impls") or {}
    impl_str = ", ".join(f"{k}={v:.3f}ms" for k, v in impls.items())
    if interfered:
        # Make INTERFERED stand out — easy to spot when scrolling.
        log("[tir-bench] " + "*" * 70)
        log(f"[tir-bench] *** INTERFERED *** {worker} gpu={gpu} {label}")
        log(f"[tir-bench] ***   intruder PIDs on gpu {gpu}: {intruder_pids}")
        log("[tir-bench] ***   subprocess killed, will be retried")
        log("[tir-bench] " + "*" * 70)
    else:
        log(
            f"[tir-bench] {record['finished_at']} {worker} gpu={gpu} {status:4s} {label} {impl_str}"
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
BASELINE_PACKAGES = ["torch", "deep_gemm", "flashinfer", "flash_attn"]


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
    out_dir: Path,
    stamp: str,
    results: list[dict],
    label: str | None,
    probe: dict | None = None,
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
    "sparse_flashmla_prefill_head64_phase1": "flashmla",
    "sparse_flashmla_prefill_head128_phase1": "flashmla",
}


def _our_impl(row_impls: dict) -> str | None:
    """Pick our impl ('tir' or 'tirx') from a row's impls dict."""
    for name in ("tir", "tirx"):
        if name in row_impls:
            return name
    return None


def write_summary(out_dir: Path, current: dict) -> Path:
    """Human-readable per-run report, grouped by kernel.

    Times are in µs to match the existing tir-bench doc convention. Per row:
    config, one column per impl present in that kernel, baseline/ours ratio
    (against the kernel's reference impl from BASELINE_IMPL_BY_KERNEL),
    then attempt + gpu.
    """
    stamp = current["timestamp"]
    reports_dir = out_dir / "reports" / stamp
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# tir-bench run {stamp}")
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
        header = ["config", *impl_names, ratio_label, "attempt", "gpu"]
        align = ["---"] + ["---:"] * len(impl_names) + ["---:", "---:", "---:"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(align) + "|")
        for r in rows:
            cfg = r.get("label") or r.get("config") or "?"
            status = r.get("status", "ok")
            impls = r.get("impls") or {}
            row = [cfg]
            for impl in impl_names:
                ms = impls.get(impl)
                row.append(f"{ms * 1000:.2f}us" if ms is not None else "—")
            # Ratio column
            ratio_cell = "—"
            if baseline_impl and ours_impl:
                base_ms = impls.get(baseline_impl)
                ours_ms = impls.get(ours_impl)
                if base_ms is not None and ours_ms is not None and ours_ms > 0:
                    ratio = base_ms / ours_ms
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


def _flatten(payload: dict) -> dict[tuple[str, str, str], float]:
    """{(kernel, config, impl) -> avg_ms} for all ok results."""
    out: dict[tuple[str, str, str], float] = {}
    for r in payload.get("results") or []:
        if r.get("status") != "ok":
            continue
        for impl, ms in (r.get("impls") or {}).items():
            out[(r["kernel"], r.get("label") or r.get("config"), impl)] = ms
    return out


def load_baseline(combined=None):
    """Join the tir + ref pinned baselines into one payload for diffing.

    ``combined`` (optional path) overrides both with a single legacy
    baseline.json. Returns None if no baseline exists yet."""
    if combined is not None:
        return json.loads(Path(combined).read_text())
    if not DEFAULT_TIR_BASELINE.exists() and not DEFAULT_REF_BASELINE.exists():
        return None
    tir = json.loads(DEFAULT_TIR_BASELINE.read_text()) if DEFAULT_TIR_BASELINE.exists() else {}
    ref = json.loads(DEFAULT_REF_BASELINE.read_text()) if DEFAULT_REF_BASELINE.exists() else {}
    ref_idx = {
        (r["kernel"], r.get("label") or r.get("config")): {
            k: v for k, v in (r.get("impls") or {}).items() if k not in OURS_IMPLS
        }
        for r in ref.get("results", [])
    }
    out = {k: v for k, v in tir.items() if k != "results"}
    out["ref_timestamp"] = ref.get("timestamp")
    merged = []
    for r in tir.get("results", []):
        key = (r["kernel"], r.get("label") or r.get("config"))
        nr = dict(r)
        nr["impls"] = {**(r.get("impls") or {}), **ref_idx.get(key, {})}
        merged.append(nr)
    out["results"] = merged
    return out


def diff_report(baseline: dict, current: dict, threshold_pct: float) -> tuple[str, int]:
    base = baseline
    base_idx = _flatten(base)
    cur_idx = _flatten(current)

    regressions: list[tuple] = []
    improvements: list[tuple] = []
    unchanged: list[tuple] = []
    new_rows: list[tuple] = []

    for key, ms in cur_idx.items():
        if key not in base_idx:
            new_rows.append((key, ms))
            continue
        old = base_idx[key]
        if old <= 0:
            continue
        delta = (ms - old) / old * 100.0
        row = (key, old, ms, delta)
        if delta >= threshold_pct:
            regressions.append(row)
        elif delta <= -threshold_pct:
            improvements.append(row)
        else:
            unchanged.append(row)

    failed = [r for r in (current.get("results") or []) if r.get("status") != "ok"]

    def fmt_table(title: str, rows: list[tuple]) -> list[str]:
        if not rows:
            return []
        lines = [
            f"## {title} ({len(rows)})",
            "",
            "| kernel | config | impl | baseline (ms) | current (ms) | Δ |",
            "|---|---|---|---:|---:|---:|",
        ]
        for (k, c, impl), old, new, d in sorted(rows, key=lambda r: -abs(r[3])):
            lines.append(f"| {k} | {c} | {impl} | {old:.4f} | {new:.4f} | {d:+.2f}% |")
        lines.append("")
        return lines

    md: list[str] = []
    md.append("# tir-bench regression report")
    md.append("")
    md.append(f"- Current:  `{current['timestamp']}` ({current.get('label') or '-'})")
    md.append(
        f"- Baseline: `{base.get('timestamp')}` ({base.get('label') or '-'})  "
        f"from `tir.json` + `ref.json`"
    )
    md.append(f"- Threshold: ±{threshold_pct:.1f}%")
    md.append("")
    md.append(
        f"**Summary** — regressions: {len(regressions)}, "
        f"improvements: {len(improvements)}, unchanged: {len(unchanged)}, "
        f"failed: {len(failed)}, new: {len(new_rows)}"
    )
    md.append("")
    md += fmt_table("Regressions", regressions)
    md += fmt_table("Improvements", improvements)

    if failed:
        md.append(f"## Failed ({len(failed)})")
        md.append("")
        for r in failed:
            first = (r.get("error") or "?").splitlines()[0]
            md.append(f"- `{r['kernel']}/{r.get('label') or r.get('config')}`: {first}")
        md.append("")

    if new_rows:
        md.append(f"## New (no baseline) ({len(new_rows)})")
        md.append("")
        md.append("| kernel | config | impl | current (ms) |")
        md.append("|---|---|---|---:|")
        for (k, c, impl), ms in new_rows:
            md.append(f"| {k} | {c} | {impl} | {ms:.4f} |")
        md.append("")

    return "\n".join(md), len(regressions)


# ── Main ─────────────────────────────────────────────────────────────────────


def _drifted_workloads(
    baseline: dict, current: dict, threshold_pct: float
) -> list[tuple[str, str]]:
    """Return (kernel, config) keys where |ratio Δ vs baseline| > threshold.

    The ratio is ref/ours (ref = fastest non-ours impl in baseline, fixed
    across runs) — same convention as ratio_diff.py, so a row is flagged
    when either ours or ref moved significantly. Skips workloads with no
    comparable ours/ref pair in both runs.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from ratio_diff import OUR_IMPLS, pick_ref
    from ratio_diff import index as ratio_index

    base = ratio_index(baseline)
    cur = ratio_index(current)
    drifted: list[tuple[str, str]] = []
    for key, base_impls in base.items():
        ref = pick_ref(base_impls)
        ours = next((i for i in OUR_IMPLS if i in base_impls), None)
        if ref is None or ours is None or key not in cur:
            continue
        ci = cur[key]
        if ref not in ci or ours not in ci:
            continue
        ob, rb = base_impls[ours], base_impls[ref]
        oc, rc = ci[ours], ci[ref]
        if min(ob, rb, oc, rc) <= 0:
            continue
        delta = (rc / oc) - (rb / ob)
        if abs(delta) / (rb / ob) * 100.0 > threshold_pct:
            drifted.append(key)
    return drifted


def _roles_for_impls(impls: str) -> list[str]:
    if impls == "ours":
        return ["ours"]
    if impls == "baseline":
        return ["baseline"]
    return ["ours", "baseline"]


def run_job_with_retry(
    workload: dict,
    role: str,
    round_idx: int,
    pool: GpuPool,
    log_dir: Path,
    *,
    max_retry: int,
    no_monitor: bool,
    requeue_log: list[tuple[str, str, str, int, list[int]]] | None = None,
) -> dict:
    kernel = workload["kernel"]
    config = workload["config"]
    attempt = 1
    while True:
        record = run_one(
            workload,
            pool,
            log_dir,
            impls_mode=role,
            no_monitor=no_monitor,
            log_tag=f"{role}_r{round_idx}_a{attempt}",
        )
        record["role"] = role
        record["round_idx"] = round_idx
        record["attempt"] = attempt

        if record.get("status") == "INTERFERED" and attempt < max_retry:
            intruders = record.get("intruder_pids") or []
            if requeue_log is not None:
                requeue_log.append((kernel, config, role, attempt, intruders))
            log(
                f"[tir-bench] >>> REQUEUE {kernel}/{config}/{role} r{round_idx} — "
                f"attempt {attempt}/{max_retry} hit interference "
                f"(intruders {intruders}), retrying <<<"
            )
            attempt += 1
            continue

        if record.get("status") == "INTERFERED":
            record["status"] = "FAIL"
            record["error"] = f"INTERFERED after {max_retry} attempts: {record.get('error', '')}"
        return record


def run_scheduled_jobs(
    workloads: list[dict],
    roles: list[str],
    rounds: int,
    pool: GpuPool,
    log_dir: Path,
    *,
    max_retry: int,
    no_monitor: bool,
    cpu_workers: int,
) -> tuple[list[dict], list[tuple[str, str, str, int, list[int]]]]:
    jobs = [(w, role, r) for w in workloads for role in roles for r in range(rounds)]
    requeue_log: list[tuple[str, str, str, int, list[int]]] = []
    records: list[dict] = []
    n_jobs = len(jobs)
    with ThreadPoolExecutor(
        max_workers=min(cpu_workers, n_jobs) if n_jobs else 1,
        thread_name_prefix="bench",
    ) as ex:
        futs = [
            ex.submit(
                run_job_with_retry,
                w,
                role,
                r,
                pool,
                log_dir,
                max_retry=max_retry,
                no_monitor=no_monitor,
                requeue_log=requeue_log,
            )
            for w, role, r in jobs
        ]
        for fut in as_completed(futs):
            records.append(fut.result())
    return records, requeue_log


def aggregate_impl_times(values: list[float], method: str) -> float:
    """Combine per-round impl times for one workload."""
    import statistics

    if method == "mean":
        return statistics.mean(values)
    if method == "median":
        return statistics.median(values)
    if method == "trimmed_mean":
        # Drop fastest + slowest round; with 5 rounds this is the middle 3.
        if len(values) < 3:
            return statistics.mean(values)
        ranked = sorted(values)
        return statistics.mean(ranked[1:-1])
    raise ValueError(f"aggregate must be mean, median, or trimmed_mean, got {method!r}")


def aggregate_rounds(
    job_records: list[dict],
    workloads: list[dict],
    *,
    rounds: int,
    min_ok_rounds: int,
    aggregate: str,
    max_retry: int,
) -> list[dict]:
    """Merge per-role round job records into one row per workload."""
    if aggregate not in ("mean", "median", "trimmed_mean"):
        raise ValueError(f"aggregate must be mean, median, or trimmed_mean, got {aggregate!r}")

    samples: dict[tuple[str, str], dict[str, list[float]]] = {}
    ok_round_counts: dict[tuple[str, str], dict[str, int]] = {}

    for rec in job_records:
        if rec.get("status") != "ok":
            continue
        key = (rec["kernel"], rec.get("label") or rec.get("config"))
        for impl, ms in (rec.get("impls") or {}).items():
            if ms is None or ms <= 0:
                continue
            samples.setdefault(key, {}).setdefault(impl, []).append(ms)
            ok_round_counts.setdefault(key, {}).setdefault(impl, 0)
            ok_round_counts[key][impl] += 1

    records: list[dict] = []
    for w in workloads:
        key = (w["kernel"], w["config"])
        impl_samples = samples.get(key, {})
        ok_counts = ok_round_counts.get(key, {})

        qualified = {
            impl: aggregate_impl_times(vs, aggregate)
            for impl, vs in impl_samples.items()
            if ok_counts.get(impl, 0) >= min_ok_rounds
        }

        if qualified:
            records.append(
                {
                    "kernel": w["kernel"],
                    "config": w["config"],
                    "label": w["config"],
                    "status": "ok",
                    "impls": qualified,
                    "aggregated": {
                        "rounds": rounds,
                        "method": aggregate,
                        "max_retry": max_retry,
                        "min_ok_rounds": min_ok_rounds,
                        "ok_rounds": ok_counts,
                    },
                }
            )
        else:
            records.append(
                {
                    "kernel": w["kernel"],
                    "config": w["config"],
                    "label": w["config"],
                    "status": "FAIL",
                    "error": (
                        f"no impl reached min_ok_rounds={min_ok_rounds} in {rounds} round(s)"
                    ),
                }
            )
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="tir-bench: pre-commit regression benchmark")
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
        "(default: <tirx-kernels>/.tir-bench)",
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional combined baseline JSON to diff against instead of tir.json + ref.json",
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
        "--no-monitor",
        action="store_true",
        help="Don't monitor for GPU interference during workloads",
    )
    ap.add_argument(
        "--util-threshold",
        type=float,
        default=DEFAULT_UTIL_THRESHOLD,
        help="%% GPU/sm utilization at/above which a card counts as "
        "actively in use: selection skips such cards and the "
        "monitor requeues if a neighbor crosses it mid-run. "
        "Cards merely holding resident VRAM at lower util are "
        f"shared (default {DEFAULT_UTIL_THRESHOLD:g})",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Independent benchmark rounds per (workload, role) job before "
        "aggregation (default 1). Each job is retried on INTERFERED up to "
        "--max-retry times.",
    )
    ap.add_argument(
        "--bench-reps",
        type=int,
        default=None,
        help="Deprecated alias for --rounds.",
    )
    ap.add_argument(
        "--bench-aggregate",
        choices=("mean", "median", "trimmed_mean"),
        default="mean",
        help="How to combine --rounds samples per impl (default mean). "
        "trimmed_mean drops the fastest and slowest ok round.",
    )
    ap.add_argument(
        "--max-retry",
        type=int,
        default=MAX_INTERFERED_RETRIES,
        help="Max attempts per job when INTERFERED (default "
        f"{MAX_INTERFERED_RETRIES}). Real FAIL exits immediately.",
    )
    ap.add_argument(
        "--min-ok-rounds",
        type=int,
        default=1,
        help="Minimum ok rounds required per impl before aggregation (default 1).",
    )
    ap.add_argument(
        "--restable-threshold",
        type=float,
        default=DEFAULT_RESTABLE_THRESHOLD,
        help="After the main sweep, re-bench any workload whose "
        "|ratio Δ vs baseline| exceeds this %% "
        f"(default {DEFAULT_RESTABLE_THRESHOLD:g}). Higher than "
        "--threshold on purpose: sub-3%% drifts are usually "
        "contention noise and re-testing them all dominates "
        "wall-time; confirm a specific small mover with a "
        "targeted --filter run instead. Set --restable-reps=0 "
        "to disable.",
    )
    ap.add_argument(
        "--restable-reps",
        type=int,
        default=3,
        help="How many additional reps to bench each drifted "
        "workload during the restable phase (default 3; "
        "median-of-3 already absorbs single-run spikes). "
        "Set 0 to skip the phase entirely.",
    )
    ap.add_argument(
        "--impls",
        choices=("all", "ours", "baseline"),
        default="ours",
        help="Which impls to bench. 'all' (default): our kernel + "
        "reference impls, ratio + restable reports (use to "
        "(re)populate tir.json + ref.json). 'ours': only our kernel — "
        "fast per-change iteration; reference times come from "
        "the pinned baseline, diff is absolute-ms (current ours "
        "vs baseline ours) — promote with cp to tir.json. 'baseline': "
        "only reference impls — promote with cp to ref.json.",
    )
    ap.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="Max concurrent bench subprocesses (CPU overcommit). "
        "Default ~4x the usable-GPU count, capped at 32. Many "
        "subprocesses import+compile in parallel while a per-GPU "
        "flock serializes the measurement phase to one per card.",
    )
    ap.add_argument(
        "--check-imports",
        action="store_true",
        help="Import every unique kernel in --workloads and exit (for CI import gates)",
    )
    args = ap.parse_args()
    rounds = args.bench_reps if args.bench_reps is not None else args.rounds
    if rounds < 1:
        print("[tir-bench] --rounds must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.max_retry < 1:
        print("[tir-bench] --max-retry must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.min_ok_rounds < 1:
        print("[tir-bench] --min-ok-rounds must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.min_ok_rounds > rounds:
        print("[tir-bench] --min-ok-rounds cannot exceed --rounds", file=sys.stderr)
        sys.exit(2)

    workloads = load_workloads(args.workloads)
    if args.filter:
        workloads = [w for w in workloads if args.filter in w["kernel"]]
    if not workloads:
        print("[tir-bench] no workloads to run.", file=sys.stderr)
        sys.exit(2)

    if args.check_imports:
        from tirx_kernels.registry import check_workload_imports

        names = check_workload_imports(workloads, strict=True)
        print(f"[tir-bench] import check ok ({len(names)} kernels from {args.workloads})")
        return

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Run id: a simple incrementing integer — one more than the highest
    # existing numeric run in runs/. Short and readable: runs/7.json,
    # runs/7-stable.json, reports/7/..., latest -> 7. (Older timestamp-named
    # runs, if any, are ignored when picking the next number.) The wall-clock
    # time still lives in the JSON's started_at/finished_at fields.
    _existing = [
        int(n) for p in runs_dir.glob("*.json") if (n := p.stem.removesuffix("-stable")).isdigit()
    ]
    stamp = str(max(_existing, default=0) + 1)
    run_log_path = runs_dir / f"{stamp}.log"
    run_log_fh = open(run_log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, run_log_fh)
    sys.stderr = _Tee(sys.stderr, run_log_fh)
    # Repoint `latest.log` symlink immediately so `tail -f .tir-bench/latest.log`
    # picks up this run before any output happens.
    latest_log = out_dir / "latest.log"
    if latest_log.exists() or latest_log.is_symlink():
        latest_log.unlink()
    latest_log.symlink_to(run_log_path.relative_to(out_dir))

    print(f"[tir-bench] live log: {run_log_path}")
    print(f"[tir-bench]   tail : tail -f {latest_log}")
    print(f"[tir-bench] run id : {stamp}")

    # ── Automatic GPU selection (no manual override on purpose) ──
    # 1. Startup probe: run a tiny fp16 matmul on every visible card
    #    (including busy ones — the probe is light, finishes fine on a
    #    contended card; this catches broken drivers / ECC). Probe failures
    #    are banned for the rest of the run.
    # 2. Per-workload acquire: re-scan utilization every time we need a card
    #    and pick any probe-OK one whose util is below --util-threshold. A
    #    card pegged at sweep start is reusable the moment its util drops; a
    #    card merely holding resident VRAM at low util is shared right away.
    listing_pool = GpuPool(util_threshold=args.util_threshold)
    in_filter = [idx for idx, _ in listing_pool._all_gpus()]
    if not in_filter:
        print("[tir-bench] no visible GPUs.", file=sys.stderr)
        sys.exit(1)
    utils_now = listing_pool._utils()
    occupied_now = sorted(listing_pool._occupied_indices() & set(in_filter), key=int)
    resident = sorted(listing_pool._busy_indices() & set(in_filter), key=int)
    util_str = " ".join(f"{i}:{utils_now.get(i, 0):.0f}%" for i in sorted(in_filter, key=int))
    print(
        f"[tir-bench] visible: {len(in_filter)} {sorted(in_filter, key=int)}; "
        f"util now [{util_str}]",
        flush=True,
    )
    print(
        f"[tir-bench] gate: util-threshold={args.util_threshold:g}% — "
        f"occupied (skip): {occupied_now if occupied_now else 'none'}; "
        f"shareable incl. idle-but-resident: "
        f"{sorted((set(in_filter) - set(occupied_now)), key=int)} "
        f"(resident-VRAM cards: {resident if resident else 'none'})",
        flush=True,
    )

    if args.no_probe:
        usable = set(in_filter)
        probe_failures: dict[str, str] = {}
    else:
        print(
            f"[tir-bench] probing {len(in_filter)} GPU(s) with fp16 512x512 matmul ...", flush=True
        )
        usable, probe_failures = detect_usable_gpus(in_filter, args.probe_timeout)

    if not usable:
        print("[tir-bench] no usable GPUs (all probes failed).", file=sys.stderr)
        for idx, err in probe_failures.items():
            print(f"[tir-bench]   gpu {idx}: {err}", file=sys.stderr)
        sys.exit(1)

    pool = GpuPool(allowed=usable, util_threshold=args.util_threshold)
    n_gpus = len(usable)
    cpu_workers = args.cpu_workers or min(4 * n_gpus, 32)

    _repo_git = collect_repo_git()
    label = args.label or _repo_git.get("tirx-kernels") or _repo_git.get("tir") or "local"
    roles = _roles_for_impls(args.impls)
    agg_note = (
        f", {rounds} round(s)/role, aggregate={args.bench_aggregate}, "
        f"min_ok_rounds={args.min_ok_rounds}, max_retry={args.max_retry}"
        if rounds > 1 or args.min_ok_rounds > 1
        else f", max_retry={args.max_retry}"
    )
    print(
        f"[tir-bench] {len(workloads)} workloads, {n_gpus} probe-OK GPU(s) in pool, "
        f"{cpu_workers} CPU worker(s), roles={roles}, label={label}{agg_note}",
        flush=True,
    )

    job_records, requeue_log = run_scheduled_jobs(
        workloads,
        roles,
        rounds,
        pool,
        log_dir,
        max_retry=args.max_retry,
        no_monitor=args.no_monitor,
        cpu_workers=cpu_workers,
    )
    results = aggregate_rounds(
        job_records,
        workloads,
        rounds=rounds,
        min_ok_rounds=args.min_ok_rounds,
        aggregate=args.bench_aggregate,
        max_retry=args.max_retry,
    )

    if requeue_log:
        log(f"[tir-bench] interference summary: {len(requeue_log)} retry event(s)")
        for k, c, role, att, intr in requeue_log:
            log(f"[tir-bench]   - {k}/{c}/{role}: attempt {att} → intruders {intr}")
    else:
        log("[tir-bench] interference summary: none")

    results.sort(key=lambda r: (r["kernel"], r.get("label") or r.get("config")))
    probe_meta = {
        "enabled": not args.no_probe,
        "usable": sorted(usable),
        "failed": probe_failures,
    }
    run_path = write_run(out_dir, stamp, results, label, probe=probe_meta)
    current = json.loads(run_path.read_text())

    latest = out_dir / "latest.json"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_path.relative_to(out_dir))

    summary_path = write_summary(out_dir, current)
    print(f"[tir-bench] wrote {run_path}")
    print(f"[tir-bench] wrote {summary_path}")

    if args.no_report:
        return

    # Two pinned baselines (tir.json + ref.json), joined at read time. Promote a
    # fresh run by cp-ing it over the matching file (per-mode hints below).
    baseline = load_baseline(args.baseline)
    if baseline is None:
        print("[tir-bench] no baseline (tir.json / ref.json) — skipping regression report")
        print(f"[tir-bench]   set baseline: promote_baseline.py {run_path} --tir/--ref")
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
    from ratio_diff import load_ratio_baseline

    n_regress = 0
    if args.impls == "all":
        try:
            bench_md, n_regress = _build_bench_report(
                baseline,
                current,
                threshold_pct=args.threshold,
                ratio_baseline=load_ratio_baseline(DEFAULT_RATIO_BASELINE),
            )
        except Exception as e:
            print(f"[tir-bench] bench report failed: {e}", file=sys.stderr)
            sys.exit(3)
    else:
        bench_md, n_regress = diff_report(baseline, current, args.threshold)
        bench_md = bench_md.replace(
            "# tir-bench regression report",
            "# tir-bench bench report\n\n_Absolute-ms diff only "
            f"(--impls {args.impls}; run with --impls all for ref+ours+ratio)._",
            1,
        )
        if args.impls == "baseline":
            print(f"[tir-bench]   promote reference times: promote_baseline.py {run_path} --ref")
        else:
            print(
                "[tir-bench] --impls ours: absolute-ms diff only "
                "(run --impls all for ref+ours+ratio vs ratio.json)"
            )
            print(f"[tir-bench]   promote your kernel times: promote_baseline.py {run_path} --tir")

    bench_path = reports_dir / "bench.md"
    bench_path.write_text(bench_md)
    print(f"[tir-bench] wrote {bench_path}\n")
    print(bench_md)

    if args.impls != "all":
        if n_regress > 0:
            sys.exit(3)
        return

    # ── Auto-restable phase ─────────────────────────────────────────────
    # Pick workloads whose ratio Δ from baseline exceeds args.restable_threshold,
    # re-run each N times, replace the result with the per-impl median, and
    # emit a stabilized ratio report. This catches outlier baseline rows
    # (one bad GPU exposure recorded in the pinned baseline) and outlier current
    # rows alike — the median of N reps is much less affected by either.
    if args.restable_reps <= 0:
        if n_regress > 0:
            sys.exit(3)
        return

    drifted_keys = _drifted_workloads(baseline, current, args.restable_threshold)
    if not drifted_keys:
        print(
            f"[tir-bench] no workloads drifted > ±{args.restable_threshold:.1f}%; "
            "skipping restable phase"
        )
        if n_regress > 0:
            sys.exit(3)
        return

    workloads_by_key = {(w["kernel"], w["config"]): w for w in workloads}
    retest_specs = [workloads_by_key[k] for k in drifted_keys if k in workloads_by_key]
    if not retest_specs:
        print("[tir-bench] drifted keys not in original workloads list; skipping restable")
        if n_regress > 0:
            sys.exit(3)
        return

    print(
        f"[tir-bench] restabilizing {len(retest_specs)} workload(s) over "
        f"{args.restable_reps} round(s)/role (|ratio Δ| > {args.restable_threshold:.1f}%) ..."
    )
    restable_jobs, _ = run_scheduled_jobs(
        retest_specs,
        roles,
        args.restable_reps,
        pool,
        log_dir,
        max_retry=args.max_retry,
        no_monitor=args.no_monitor,
        cpu_workers=cpu_workers,
    )
    restable_records = aggregate_rounds(
        restable_jobs,
        retest_specs,
        rounds=args.restable_reps,
        min_ok_rounds=1,
        aggregate="median",
        max_retry=args.max_retry,
    )
    medians = {
        (r["kernel"], r.get("label") or r.get("config")): r["impls"]
        for r in restable_records
        if r.get("status") == "ok"
    }

    # Patch current results in-place with the stabilized per-impl medians.
    n_patched = 0
    for r in current["results"]:
        key = (r["kernel"], r.get("label") or r.get("config"))
        if key in medians:
            old_impls = dict(r.get("impls") or {})
            r["impls"] = medians[key]
            r["restabilized"] = {
                "rounds": args.restable_reps,
                "old_impls": old_impls,
            }
            n_patched += 1
    print(f"[tir-bench] patched {n_patched} restabilized workload(s) into current run")

    stable_path = runs_dir / f"{current['timestamp']}-stable.json"
    stable_path.write_text(json.dumps(current, indent=2))
    print(f"[tir-bench] wrote {stable_path}")

    try:
        bench_md, n_regress = _build_bench_report(
            baseline,
            current,
            threshold_pct=args.threshold,
            ratio_baseline=load_ratio_baseline(DEFAULT_RATIO_BASELINE),
        )
        bench_path.write_text(bench_md)
        print(f"[tir-bench] rewrote {bench_path} (after restable)\n")
    except Exception as e:
        print(f"[tir-bench] restable bench report failed: {e}", file=sys.stderr)

    if n_regress > 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
