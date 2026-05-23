#!/usr/bin/env python3
import errno
import os
import signal
import socket
import struct
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Deque, Dict, List, Optional, Tuple

from gpu_service_core import ProfileContext, feature_array_from_bytes, merge_segments


FRAME_MAX_BYTES = 256 * 1024 * 1024
RECENT_METRIC_HISTORY = 1000


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name, "").strip()
    return raw if raw else default


def _is_session_disconnect_error(exc: BaseException) -> bool:
    if isinstance(exc, EOFError):
        return True
    if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno in (
            errno.EBADF,
            errno.EPIPE,
            errno.ECONNRESET,
            errno.ENOTCONN,
            errno.ECONNABORTED,
        )
    return False


def _session_is_closed(session) -> bool:
    if session is None:
        return True
    checker = getattr(session, "is_closed", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return True
    return bool(getattr(session, "closed", False))


def read_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("socket closed")
        data.extend(chunk)
    return bytes(data)


def _decode_headers(data: bytes, offset: int) -> Tuple[Dict[str, str], int]:
    (count,) = struct.unpack_from(">i", data, offset)
    offset += 4
    headers: Dict[str, str] = {}
    for _ in range(count):
        (k_len,) = struct.unpack_from(">i", data, offset)
        offset += 4
        key = data[offset:offset + k_len].decode("utf-8")
        offset += k_len
        (v_len,) = struct.unpack_from(">i", data, offset)
        offset += 4
        value = data[offset:offset + v_len].decode("utf-8")
        offset += v_len
        headers[key] = value
    return headers, offset


def _encode_headers(headers: Dict[str, str]) -> bytes:
    parts = [struct.pack(">i", len(headers))]
    for key, value in headers.items():
        key_bytes = key.encode("utf-8")
        value_bytes = str(value).encode("utf-8")
        parts.append(struct.pack(">i", len(key_bytes)))
        parts.append(key_bytes)
        parts.append(struct.pack(">i", len(value_bytes)))
        parts.append(value_bytes)
    return b"".join(parts)


def read_request(sock: socket.socket) -> Tuple[int, int, Dict[str, str], bytes]:
    (frame_len,) = struct.unpack(">i", read_exact(sock, 4))
    if frame_len < 0 or frame_len > FRAME_MAX_BYTES:
        raise ValueError(f"invalid request frame length {frame_len}")
    body = read_exact(sock, frame_len)
    opcode, request_id = struct.unpack_from(">iq", body, 0)
    headers, offset = _decode_headers(body, 12)
    (payload_len,) = struct.unpack_from(">i", body, offset)
    offset += 4
    payload = body[offset:offset + payload_len]
    return opcode, request_id, headers, payload


def write_response(sock: socket.socket, send_lock: threading.Lock, status: int, request_id: int,
                   headers: Optional[Dict[str, str]] = None, payload: bytes = b"") -> None:
    headers = headers or {}
    payload = payload or b""
    body = [
        struct.pack(">i", status),
        struct.pack(">q", int(request_id)),
        _encode_headers(headers),
        struct.pack(">i", len(payload)),
        payload,
    ]
    frame = b"".join(body)
    with send_lock:
        sock.sendall(struct.pack(">i", len(frame)))
        sock.sendall(frame)


def unpack_segments(payload: bytes) -> List[bytes]:
    offset = 0
    (count,) = struct.unpack_from(">i", payload, offset)
    offset += 4
    segments: List[bytes] = []
    for _ in range(count):
        (seg_len,) = struct.unpack_from(">i", payload, offset)
        offset += 4
        segments.append(payload[offset:offset + seg_len])
        offset += seg_len
    return segments


class ConnectionSession:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.send_lock = threading.Lock()
        self.profile_id = None  # type: Optional[str]
        self._closed = threading.Event()

    def is_closed(self) -> bool:
        return self._closed.is_set()

    def reply(self, status: int, request_id: int, headers: Optional[Dict[str, str]] = None, payload: bytes = b"") -> None:
        if self.is_closed():
            raise OSError(errno.EBADF, "session closed")
        write_response(self.sock, self.send_lock, status, request_id, headers, payload)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


class ScoreTask:
    def __init__(
        self,
        session: ConnectionSession,
        request_id: int,
        profile_id: str,
        headers: Dict[str, str],
        segments: List[bytes],
        enqueued_at: Optional[float] = None,
    ) -> None:
        self.session = session
        self.request_id = request_id
        self.profile_id = profile_id
        self.headers = headers
        self.segments = segments
        self.enqueued_at = time.monotonic() if enqueued_at is None else enqueued_at

    @property
    def batch_key(self) -> Tuple[str, str, str, str, str, str]:
        return (
            self.profile_id,
            self.headers.get("policy_key", "train"),
            self.headers.get("head_id", "action"),
            self.headers.get("seq_len", "0"),
            self.headers.get("d_model", "0"),
            self.headers.get("max_candidates", "0"),
        )

    @property
    def batch_size(self) -> int:
        return max(1, int(self.headers.get("batch_size", "1")))


class TrainTask:
    def __init__(
        self,
        profile_id: str,
        headers: Dict[str, str],
        segments: List[bytes],
        enqueued_at: Optional[float] = None,
    ) -> None:
        self.profile_id = profile_id
        self.headers = headers
        self.segments = segments
        self.enqueued_at = time.monotonic() if enqueued_at is None else enqueued_at

    @property
    def step_count(self) -> int:
        return int(self.headers.get("batch_size", "0"))


class ProfileState:
    def __init__(self, infer_context: ProfileContext, learner_context: ProfileContext) -> None:
        self.infer_context = infer_context
        self.learner_context = learner_context
        self.pending_scores = deque()  # type: Deque[ScoreTask]
        self.pending_trains = deque()  # type: Deque[TrainTask]
        self.last_publish_at = 0.0
        self.last_reload_at = 0.0
        self.mulligan_train_count = 0
        self.mulligan_save_interval = max(1, env_int("MULLIGAN_SAVE_INTERVAL", 100))


class SharedGpuHost:
    def __init__(self) -> None:
        self.bind_host = env_str("GPU_SERVICE_BIND_HOST", "0.0.0.0")
        self.port = env_int("GPU_SERVICE_PORT", 26100)
        self.metrics_port = env_int("GPU_SERVICE_METRICS_PORT", 27100)
        self.batch_timeout_ms = max(1, env_int("PY_BATCH_TIMEOUT_MS", 50))
        self.batch_timeout_s = self.batch_timeout_ms / 1000.0
        self.train_batch_timeout_ms = max(1, env_int("GPU_SERVICE_TRAIN_BATCH_TIMEOUT_MS", min(100, self.batch_timeout_ms)))
        self.train_batch_timeout_s = self.train_batch_timeout_ms / 1000.0
        self.batch_max_size = max(1, env_int("PY_BATCH_MAX_SIZE", 64))
        self.learner_batch_max_episodes = max(1, env_int("LEARNER_BATCH_MAX_EPISODES", 64))
        self.learner_batch_max_steps = max(1, env_int("LEARNER_BATCH_MAX_STEPS", 16384))
        self.train_multi_max_steps = max(1, env_int("TRAIN_MULTI_MAX_STEPS", self.learner_batch_max_steps))
        self.model_publish_every_ms = max(0, env_int("GPU_SERVICE_MODEL_PUBLISH_EVERY_MS", 1000))
        self.model_reload_every_ms = max(0, env_int("GPU_SERVICE_MODEL_RELOAD_EVERY_MS", 250))
        self.service_role = env_str("GPU_SERVICE_ROLE", "both")  # "both", "infer"
        self.infer_cuda_device = os.getenv("INFER_CUDA_DEVICE", "").strip()
        self.train_cuda_device = os.getenv("TRAIN_CUDA_DEVICE", "").strip()
        self._score_worker_threads_raw = env_int("SCORE_WORKER_THREADS", 0)  # 0 = auto (match profile count)
        self.score_worker_threads = max(1, self._score_worker_threads_raw)
        self._profiles: Dict[str, ProfileState] = {}
        self._lock = threading.Condition()
        self._running = True
        self._processing_profiles: set = set()  # profile ids currently being scored
        self._training_profiles: set = set()  # profile ids currently being trained
        self._score_worker_count = 0
        self.train_worker_threads = max(1, env_int("TRAIN_WORKER_THREADS", 3))
        default_train_concurrency = 1 if (not self.train_cuda_device or self.train_cuda_device.lower().startswith("cuda")) else self.train_worker_threads
        self.train_gpu_max_concurrent = max(1, env_int("TRAIN_GPU_MAX_CONCURRENT", default_train_concurrency))
        self.train_vram_guard_requeue_sleep_s = max(0.1, env_int("TRAIN_VRAM_GUARD_REQUEUE_SLEEP_MS", 1000) / 1000.0)
        self.pending_train_max = max(0, env_int("PENDING_TRAIN_MAX", 32))
        self.pending_train_backpressure = env_str("PENDING_TRAIN_BACKPRESSURE", "block").lower()
        if self.pending_train_backpressure == "drop":
            self.pending_train_backpressure = "drop_oldest"
        if self.pending_train_backpressure not in ("block", "drop_oldest", "drop_newest"):
            self.pending_train_backpressure = "block"
        self.pending_train_offer_timeout_s = max(0.1, env_int("PENDING_TRAIN_OFFER_TIMEOUT_MS", 30000) / 1000.0)
        self._train_rr = 0
        self._score_batches = 0
        self._train_batches = 0
        self._score_failures = 0
        self._train_failures = 0
        self._model_publishes = 0
        self._model_reloads = 0
        self._score_flush_timeout_total = 0
        self._score_flush_full_total = 0
        self._score_flush_eager_total = 0
        self._last_error = ""
        self._active_train_batches = 0
        self._train_requeues_vram = 0
        self._train_tasks_dropped = 0
        self._train_enqueue_waits = 0
        self._train_enqueue_wait_ms = 0.0
        self._mulligan_saves = 0
        self._sigterm_saves = 0
        self._shutdown_event = threading.Event()
        self._infer_latency_ms: Deque[float] = deque(maxlen=RECENT_METRIC_HISTORY)
        self._infer_service_ms: Deque[float] = deque(maxlen=RECENT_METRIC_HISTORY)
        self._infer_batch_sizes: Deque[float] = deque(maxlen=RECENT_METRIC_HISTORY)
        self._train_latency_ms: Deque[float] = deque(maxlen=RECENT_METRIC_HISTORY)
        self._train_service_ms: Deque[float] = deque(maxlen=RECENT_METRIC_HISTORY)
        self._train_batch_episode_counts: Deque[float] = deque(maxlen=RECENT_METRIC_HISTORY)
        self._train_batch_step_counts: Deque[float] = deque(maxlen=RECENT_METRIC_HISTORY)
        # Saturation tracking
        self._active_connections = 0
        self._total_connections = 0
        self._score_samples_total = 0
        self._score_worker_busy_ms = 0.0
        self._score_worker_wall_ms = 0.0
        self._saturation_window_start = time.monotonic()
        self._saturation_window_samples = 0
        self._saturation_window_busy_ms = 0.0

    def get_or_create_profile(self, headers: Dict[str, str]) -> ProfileState:
        profile_id = headers.get("profile_id", "").strip()
        if not profile_id:
            raise ValueError("register_profile missing profile_id")
        with self._lock:
            existing = self._profiles.get(profile_id)
        if existing is not None:
            return existing
        infer_context = ProfileContext(profile_id, headers, role="inference",
                                       cuda_device=self.infer_cuda_device)
        learner_context = ProfileContext(profile_id, headers, role="learner",
                                         cuda_device=self.train_cuda_device)
        state = ProfileState(infer_context=infer_context, learner_context=learner_context)
        with self._lock:
            return self._profiles.setdefault(profile_id, state)

    def require_profile(self, profile_id: str) -> ProfileState:
        with self._lock:
            state = self._profiles.get(profile_id)
        if state is None:
            raise ValueError(f"profile {profile_id} is not registered")
        return state

    def enqueue_score(self, state: ProfileState, task: ScoreTask) -> None:
        if _session_is_closed(task.session):
            return
        with self._lock:
            state.pending_scores.append(task)
            self._lock.notify_all()

    def enqueue_train(self, state: ProfileState, task: TrainTask) -> Tuple[bool, int, float]:
        wait_started = None
        dropped = 0
        with self._lock:
            # Hard cap: each train task holds full serialized trajectory bytes.
            # In block mode, learner saturation propagates back to JVM actors
            # instead of silently discarding completed self-play games.
            while self.pending_train_max > 0 and len(state.pending_trains) >= self.pending_train_max:
                if self.pending_train_backpressure == "drop_oldest":
                    dropped_task = state.pending_trains.popleft()
                    self._train_tasks_dropped += 1
                    dropped += 1
                    if self._train_tasks_dropped % 50 == 1:
                        print(f"[BACKPRESSURE] dropped oldest train task (profile={dropped_task.profile_id}, total_dropped={self._train_tasks_dropped}, pending={len(state.pending_trains)})", flush=True)
                    continue
                if self.pending_train_backpressure == "drop_newest":
                    self._train_tasks_dropped += 1
                    if self._train_tasks_dropped % 50 == 1:
                        print(f"[BACKPRESSURE] dropped newest train task (profile={task.profile_id}, total_dropped={self._train_tasks_dropped}, pending={len(state.pending_trains)})", flush=True)
                    return False, dropped + 1, 0.0
                if wait_started is None:
                    wait_started = time.monotonic()
                self._lock.wait(timeout=self.pending_train_offer_timeout_s)
                waited_ms = (time.monotonic() - wait_started) * 1000.0
                if len(state.pending_trains) >= self.pending_train_max:
                    print(f"[BACKPRESSURE] train queue full for {waited_ms:.0f}ms "
                          f"(profile={state.infer_context.profile_id}, pending={len(state.pending_trains)}, "
                          f"max={self.pending_train_max}); blocking producer", flush=True)
            state.pending_trains.append(task)
            waited_ms = 0.0 if wait_started is None else (time.monotonic() - wait_started) * 1000.0
            if waited_ms > 0.0:
                self._train_enqueue_waits += 1
                self._train_enqueue_wait_ms += waited_ms
            self._lock.notify_all()
            return True, dropped, waited_ms

    def _maybe_publish_latest_model(self, state: ProfileState, now: Optional[float] = None) -> None:
        if state is None or self.model_publish_every_ms <= 0:
            return
        current = time.monotonic() if now is None else now
        if state.last_publish_at > 0.0 and ((current - state.last_publish_at) * 1000.0) < self.model_publish_every_ms:
            return
        if state.learner_context.save_latest_model_atomic():
            with self._lock:
                state.last_publish_at = current
                self._model_publishes += 1

    def _maybe_reload_inference_model(self, state: ProfileState, now: Optional[float] = None) -> None:
        if state is None or self.model_reload_every_ms <= 0:
            return
        current = time.monotonic() if now is None else now
        if state.last_reload_at > 0.0 and ((current - state.last_reload_at) * 1000.0) < self.model_reload_every_ms:
            return
        reloaded = state.infer_context.reload_latest_model_if_newer()
        with self._lock:
            state.last_reload_at = current
            if reloaded:
                self._model_reloads += 1

    def score_worker_loop(self, worker_id: int = 0) -> None:
        tag = f"SCORE_WORKER-{worker_id}"
        _diag_interval = 10.0
        _diag_last = time.monotonic()
        _diag_cycles = 0
        _diag_batches = 0
        _diag_samples = 0
        _diag_wait_ms_sum = 0.0
        _diag_run_ms_sum = 0.0
        while self._running:
            cycle_start = time.monotonic()
            # Claim ALL ready profiles at once -- process them all in one cycle
            # to reduce per-cycle lock/GIL overhead
            work_items = []
            with self._lock:
                while True:
                    work = self._claim_score_work_locked(worker_id)
                    if work is None:
                        break
                    work_items.append(work)
                if not work_items:
                    self._lock.wait(timeout=self.batch_timeout_s)
                    while True:
                        work = self._claim_score_work_locked(worker_id)
                        if work is None:
                            break
                        work_items.append(work)
                    if not work_items:
                        continue
            wait_end = time.monotonic()
            _diag_wait_ms_sum += (wait_end - cycle_start) * 1000
            _diag_cycles += 1
            for state, tasks, reason, profile_key in work_items:
                batch_samples = sum(task.batch_size for task in tasks)
                _diag_batches += 1
                try:
                    self._run_score_batch(state, tasks, reason)
                except Exception as exc:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    with self._lock:
                        self._score_failures += 1
                    traceback.print_exc()
                    for task in tasks:
                        try:
                            task.session.reply(1, task.request_id, {"error": str(exc)})
                        except Exception:
                            pass
                finally:
                    with self._lock:
                        self._processing_profiles.discard(profile_key)
                        self._lock.notify_all()
                run_end = time.monotonic()
                run_ms = (run_end - wait_end) * 1000
                _diag_run_ms_sum += run_ms
                _diag_samples += batch_samples
                wait_end = run_end  # for next sub-batch timing
            with self._lock:
                self._score_samples_total += batch_samples
                self._score_worker_busy_ms += run_ms
                self._saturation_window_samples += batch_samples
                self._saturation_window_busy_ms += run_ms
            now = time.monotonic()
            if now - _diag_last >= _diag_interval:
                elapsed = now - _diag_last
                avg_wait = _diag_wait_ms_sum / max(1, _diag_cycles)
                avg_run = _diag_run_ms_sum / max(1, _diag_cycles)
                duty = _diag_run_ms_sum / (elapsed * 1000) * 100
                throughput = _diag_samples / max(0.001, elapsed)
                with self._lock:
                    conns = self._active_connections
                    pending = sum(
                        sum(task.batch_size for task in s.pending_scores)
                        for s in self._profiles.values()
                    )
                print(f"[{tag}] {elapsed:.0f}s: cycles={_diag_cycles} batches={_diag_batches} "
                      f"samples={_diag_samples} throughput={throughput:.0f}/s "
                      f"avg_wait={avg_wait:.1f}ms avg_run={avg_run:.1f}ms "
                      f"duty={duty:.1f}% conns={conns} pending={pending}", flush=True)
                with self._lock:
                    wall_ms = (now - self._saturation_window_start) * 1000
                    self._score_worker_wall_ms += wall_ms
                    self._saturation_window_start = now
                    self._saturation_window_samples = 0
                    self._saturation_window_busy_ms = 0.0
                _diag_last = now
                _diag_cycles = 0
                _diag_batches = 0
                _diag_samples = 0
                _diag_wait_ms_sum = 0.0
                _diag_run_ms_sum = 0.0

    def _log_memory(self, tag: str) -> None:
        # Periodic memory diagnostic so we can see when RSS climbs unbounded.
        # Uses psutil if available, else skips. Called from train_worker_loop
        # every N batches.
        try:
            import psutil
            import os as _os
            p = psutil.Process(_os.getpid())
            rss_gb = p.memory_info().rss / (1024 ** 3)
            virt_gb = p.memory_info().vms / (1024 ** 3)
            print(f"[MEMORY] {tag} rss={rss_gb:.1f}GB virtual={virt_gb:.1f}GB", flush=True)
        except Exception:
            pass

        # Pending queue depth — answers "are score/train tasks accumulating?"
        try:
            with self._lock:
                score_pending = sum(len(s.pending_scores) for s in self._profiles.values())
                train_pending = sum(len(s.pending_trains) for s in self._profiles.values())
                conns = self._active_connections
            # Count live ScoreTask/TrainTask instances via gc to see if they
            # persist OUTSIDE the queues (would indicate a hidden reference).
            try:
                import gc as _gc
                score_alive = 0
                train_alive = 0
                for obj in _gc.get_objects():
                    t = type(obj).__name__
                    if t == "ScoreTask":
                        score_alive += 1
                    elif t == "TrainTask":
                        train_alive += 1
                print(f"[MEMORY] {tag} pending_score={score_pending} pending_train={train_pending} alive_ScoreTask={score_alive} alive_TrainTask={train_alive} conns={conns}", flush=True)
            except Exception:
                print(f"[MEMORY] {tag} pending_score={score_pending} pending_train={train_pending} conns={conns}", flush=True)
        except Exception:
            pass

        # tracemalloc: enabled via MEMORY_TRACEMALLOC=1. Uses diff-against-prior-snapshot
        # so we see which call sites GREW since last check — precisely identifies the
        # leaky line when memory is climbing batch over batch.
        try:
            if os.environ.get("MEMORY_TRACEMALLOC", "0") == "1":
                import tracemalloc
                if not tracemalloc.is_tracing():
                    tracemalloc.start(25)
                    self._prev_trace_snapshot = None
                    print(f"[MEMORY] tracemalloc started at tag={tag}", flush=True)
                    return
                snap = tracemalloc.take_snapshot()
                # Filter out tracemalloc internals
                snap = snap.filter_traces((
                    tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                    tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
                    tracemalloc.Filter(False, tracemalloc.__file__),
                ))
                prev = getattr(self, "_prev_trace_snapshot", None)
                if prev is not None:
                    # Diff against previous snapshot — shows which lines are GROWING
                    stats = snap.compare_to(prev, "lineno")[:15]
                    print(f"[MEMORY] {tag} tracemalloc diff top 15 (size_diff, count_diff):", flush=True)
                    for i, stat in enumerate(stats):
                        frame = stat.traceback[0]
                        print(f"[MEMORY]   #{i+1} +{stat.size_diff/1024/1024:+.1f}MB +{stat.count_diff:+d} total={stat.size/1024/1024:.1f}MB {frame.filename}:{frame.lineno}", flush=True)
                else:
                    stats = snap.statistics("lineno")[:15]
                    print(f"[MEMORY] {tag} tracemalloc absolute top 15:", flush=True)
                    for i, stat in enumerate(stats):
                        frame = stat.traceback[0]
                        print(f"[MEMORY]   #{i+1} {stat.size/1024/1024:.1f}MB count={stat.count} {frame.filename}:{frame.lineno}", flush=True)
                self._prev_trace_snapshot = snap
        except Exception as exc:
            print(f"[MEMORY] tracemalloc error: {exc}", flush=True)

        # GC object count — if this climbs, we have reference leaks in Python
        # objects (not numpy/torch which are invisible to gc).
        try:
            if os.environ.get("MEMORY_GC_STATS", "0") == "1":
                import gc
                counts = gc.get_count()
                total_objs = len(gc.get_objects())
                print(f"[MEMORY] {tag} gc_count={counts} gc_objects={total_objs}", flush=True)
        except Exception:
            pass

    def train_worker_loop(self, worker_id: int = 0) -> None:
        # Learner updates drain independently and publish fresh weights for the
        # inference lane to reload asynchronously.
        _memory_log_counter = 0
        while self._running:
            work = None
            sleep_for = 0.25
            with self._lock:
                if self._active_train_batches >= self.train_gpu_max_concurrent:
                    self._lock.wait(timeout=0.05)
                    continue
                work, sleep_for = self._select_train_work_locked()
                if work is None:
                    self._lock.wait(timeout=sleep_for)
                    continue
                profile_key = work[0].infer_context.profile_id
                self._training_profiles.add(profile_key)
                self._active_train_batches += 1
            state, tasks = work
            try:
                self._run_train_batch(state, tasks)
            except Exception as exc:
                if "VRAM_GUARD_NO_HEADROOM" in str(exc):
                    with self._lock:
                        for task in reversed(tasks):
                            state.pending_trains.appendleft(task)
                        self._train_requeues_vram += len(tasks)
                        self._last_error = f"VRAM guard requeued {len(tasks)} train episode(s): {exc}"
                        self._lock.notify_all()
                    print(f"[VRAM_GUARD] requeued train batch episodes={len(tasks)} "
                          f"profile={state.infer_context.profile_id}; sleeping "
                          f"{self.train_vram_guard_requeue_sleep_s:.1f}s", flush=True)
                    time.sleep(self.train_vram_guard_requeue_sleep_s)
                else:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    with self._lock:
                        self._train_failures += 1
                    traceback.print_exc()
            finally:
                with self._lock:
                    self._training_profiles.discard(profile_key)
                    self._active_train_batches = max(0, self._active_train_batches - 1)
                    self._lock.notify_all()
                _memory_log_counter += 1
                if _memory_log_counter % 10 == 0:
                    self._log_memory(f"after_train_batch_{_memory_log_counter}")
                # Defensive: force-collect reference cycles every N batches so leaks
                # caused by cycles (closures + tensors + autograd graphs) flatten out.
                # Does not help actual unbounded-growth leaks — those need a code fix.
                gc_every = env_int("GPU_SERVICE_GC_EVERY_BATCHES", 5)
                if gc_every > 0 and _memory_log_counter % gc_every == 0:
                    try:
                        import gc as _gc
                        collected = _gc.collect()
                        if collected > 0 and _memory_log_counter % 20 == 0:
                            print(f"[MEMORY] gc.collect() freed {collected} objects at batch {_memory_log_counter}", flush=True)
                    except Exception:
                        pass

    def _claim_score_work_locked(self, worker_id: int = 0):
        """Select one batch from a profile not currently being processed by another worker.
        Each worker starts scanning from a different offset to distribute work evenly."""
        now = time.monotonic()
        profiles = list(self._profiles.values())
        n = len(profiles)
        if n == 0:
            return None
        best_state = None
        best_age = -1.0
        best_pending = 0
        for i in range(n):
            state = profiles[(worker_id + i) % n]
            pid = state.infer_context.profile_id
            if pid in self._processing_profiles:
                continue
            self._prune_closed_score_tasks_locked(state)
            if not state.pending_scores:
                continue
            age = now - state.pending_scores[0].enqueued_at
            pending = sum(task.batch_size for task in state.pending_scores)
            ready = age >= self.batch_timeout_s or pending >= self.batch_max_size
            if not ready:
                continue
            # First ready profile in this worker's rotation wins
            best_state = state
            best_age = age
            best_pending = pending
            break
        if best_state is None:
            return None
        reason = "timeout" if best_age >= self.batch_timeout_s else "full"
        batch = self._pop_score_batch_locked(best_state)
        if not batch:
            return None
        profile_key = best_state.infer_context.profile_id
        self._processing_profiles.add(profile_key)
        return best_state, batch, reason, profile_key

    def _select_all_score_work_locked(self):
        now = time.monotonic()
        profiles = list(self._profiles.values())
        results = []
        oldest_score_age = None
        total_pending = 0
        for state in profiles:
            self._prune_closed_score_tasks_locked(state)
            if not state.pending_scores:
                continue
            age = now - state.pending_scores[0].enqueued_at
            if oldest_score_age is None or age > oldest_score_age:
                oldest_score_age = age
            pending_score_samples = sum(task.batch_size for task in state.pending_scores)
            total_pending += pending_score_samples
            if age >= self.batch_timeout_s or pending_score_samples >= self.batch_max_size:
                reason = "timeout" if age >= self.batch_timeout_s else "full"
                while state.pending_scores:
                    batch = self._pop_score_batch_locked(state)
                    if not batch:
                        break
                    results.append((state, batch, reason))
                    remaining = sum(task.batch_size for task in state.pending_scores)
                    if remaining < self.batch_max_size:
                        break
        # If no profile hit its timeout/max individually but there's enough
        # aggregate pending work, flush the largest queues eagerly to avoid
        # idle GPU cycles while requests wait for per-profile timeout.
        if not results and total_pending >= self.batch_max_size:
            for state in sorted(profiles, key=lambda s: len(s.pending_scores), reverse=True):
                if not state.pending_scores:
                    continue
                batch = self._pop_score_batch_locked(state)
                if batch:
                    results.append((state, batch, "eager"))
                    break
        if results:
            return results, 0.0
        if oldest_score_age is not None:
            return None, max(0.001, min(0.25, self.batch_timeout_s - oldest_score_age))
        return None, 0.25

    def _select_train_work_locked(self):
        now = time.monotonic()
        profiles = list(self._profiles.values())
        # Inference-priority mode: defer training when ANY inference is pending.
        # This ensures training never blocks inference on the dedicated inference GPU.
        if self.service_role == "infer":
            has_pending_scores = any(len(s.pending_scores) > 0 for s in profiles)
            if has_pending_scores:
                return None, 0.05  # short sleep, re-check soon
        train_state, train_wait = self._select_train_state_locked(now, profiles)
        if train_state is not None:
            train_work = self._pop_train_batch_locked(train_state)
            if train_work is not None:
                state, tasks = train_work
                return (state, tasks), 0.0
        return None, max(0.01, min(0.25, train_wait)) if train_wait is not None else 0.25

    def _select_train_state_locked(self, now: float, profiles: List[ProfileState]) -> Tuple[Optional[ProfileState], Optional[float]]:
        if not profiles:
            return None, None
        best_state = None
        best_age = -1.0
        min_wait = None
        max_steps = min(self.learner_batch_max_steps, self.train_multi_max_steps)
        for state in profiles:
            pid = state.infer_context.profile_id
            if pid in self._training_profiles:
                continue
            if not state.pending_trains:
                continue
            age = now - state.pending_trains[0].enqueued_at
            pending_episodes = len(state.pending_trains)
            pending_steps = 0
            for task in state.pending_trains:
                pending_steps += task.step_count
                if pending_steps >= max_steps:
                    break
            ready = (
                age >= self.train_batch_timeout_s
                or pending_episodes >= self.learner_batch_max_episodes
                or pending_steps >= max_steps
            )
            if ready:
                if age > best_age:
                    best_age = age
                    best_state = state
            else:
                wait = max(0.0, self.train_batch_timeout_s - age)
                if min_wait is None or wait < min_wait:
                    min_wait = wait
        return best_state, min_wait

    def _pop_train_batch_locked(self, state: ProfileState) -> Optional[Tuple[ProfileState, List[TrainTask]]]:
        if state is None or not state.pending_trains:
            return None
        profiles = list(self._profiles.values())
        if profiles:
            try:
                idx = profiles.index(state)
                self._train_rr = (idx + 1) % len(profiles)
            except ValueError:
                pass
        tasks: List[TrainTask] = []
        steps = 0
        max_steps = min(self.learner_batch_max_steps, self.train_multi_max_steps)
        while state.pending_trains and len(tasks) < self.learner_batch_max_episodes:
            task = state.pending_trains[0]
            if tasks and steps + task.step_count > max_steps:
                break
            tasks.append(state.pending_trains.popleft())
            steps += task.step_count
        if not tasks and state.pending_trains:
            tasks.append(state.pending_trains.popleft())
        if tasks:
            self._lock.notify_all()
        return (state, tasks) if tasks else None

    def _pop_score_batch_locked(self, state: ProfileState) -> List[ScoreTask]:
        first = state.pending_scores.popleft()
        tasks = [first]
        logical_batch_size = first.batch_size
        keep: Deque[ScoreTask] = deque()
        while state.pending_scores:
            task = state.pending_scores.popleft()
            if _session_is_closed(task.session):
                continue
            if (
                task.batch_key == first.batch_key
                and (logical_batch_size + task.batch_size) <= self.batch_max_size
            ):
                tasks.append(task)
                logical_batch_size += task.batch_size
            else:
                keep.append(task)
        state.pending_scores = keep
        return tasks

    def _prune_closed_score_tasks_locked(self, state: ProfileState) -> int:
        if state is None or not state.pending_scores:
            return 0
        dropped = 0
        keep: Deque[ScoreTask] = deque()
        while state.pending_scores:
            task = state.pending_scores.popleft()
            if _session_is_closed(task.session):
                dropped += task.batch_size
                continue
            keep.append(task)
        state.pending_scores = keep
        return dropped

    def discard_session_scores(self, session: ConnectionSession) -> int:
        if session is None:
            return 0
        removed = 0
        with self._lock:
            for state in self._profiles.values():
                if not state.pending_scores:
                    continue
                keep: Deque[ScoreTask] = deque()
                while state.pending_scores:
                    task = state.pending_scores.popleft()
                    if task.session is session or _session_is_closed(task.session):
                        removed += task.batch_size
                        continue
                    keep.append(task)
                state.pending_scores = keep
            if removed:
                self._lock.notify_all()
        return removed

    def _run_score_batch(self, state: ProfileState, tasks: List[ScoreTask], reason: str) -> None:
        tasks = [task for task in tasks if not _session_is_closed(task.session)]
        if not tasks:
            return
        first = tasks[0]
        headers = first.headers
        t0 = time.monotonic()
        merged = merge_segments([task.segments for task in tasks])
        t1 = time.monotonic()
        logical_batch_size = sum(task.batch_size for task in tasks)
        self._maybe_reload_inference_model(state)
        t2 = time.monotonic()
        result_bytes = state.infer_context.score_batch(
            merged[0],
            merged[1],
            merged[2],
            merged[3],
            merged[4],
            merged[5],
            headers.get("policy_key", "train"),
            headers.get("head_id", "action"),
            int(headers.get("pick_index", "0")),
            int(headers.get("min_targets", "0")),
            int(headers.get("max_targets", "0")),
            logical_batch_size,
            int(headers.get("seq_len", "0")),
            int(headers.get("d_model", "0")),
            int(headers.get("max_candidates", "0")),
            int(headers.get("cand_feat_dim", "0")),
        )
        finished = time.monotonic()
        merge_ms = (t1 - t0) * 1000
        reload_ms = (t2 - t1) * 1000
        infer_ms = (finished - t2) * 1000
        total_ms = (finished - t0) * 1000
        queue_age_ms = (t0 - first.enqueued_at) * 1000
        if total_ms > 200 or queue_age_ms > 200:
            print(f"[SCORE_DIAG] batch={logical_batch_size} reason={reason} "
                  f"queue_age={queue_age_ms:.0f}ms merge={merge_ms:.0f}ms "
                  f"reload={reload_ms:.0f}ms infer={infer_ms:.0f}ms total={total_ms:.0f}ms",
                  flush=True)
        service_ms = max(0.0, (finished - t0) * 1000.0)
        latency_samples = [max(0.0, (finished - task.enqueued_at) * 1000.0) for task in tasks]
        with self._lock:
            self._score_batches += 1
            if reason == "full":
                self._score_flush_full_total += 1
            elif reason == "eager":
                self._score_flush_eager_total += 1
            else:
                self._score_flush_timeout_total += 1
            self._infer_batch_sizes.append(float(logical_batch_size))
            self._infer_service_ms.append(service_ms)
            for task, latency_ms in zip(tasks, latency_samples):
                for _ in range(task.batch_size):
                    self._infer_latency_ms.append(latency_ms)
        max_candidates = int(headers.get("max_candidates", "0"))
        stride = (max_candidates + 1) * 4
        result_offset = 0
        for idx, task in enumerate(tasks):
            task_stride = task.batch_size * stride
            start = result_offset
            end = start + task_stride
            result_offset = end
            extra = {
                "host_batch_size": str(logical_batch_size),
                "host_flush_reason": reason,
            }
            if idx == 0:
                extra["flush_leader"] = "1"
            try:
                task.session.reply(0, task.request_id, extra, result_bytes[start:end])
            except Exception as exc:
                if _is_session_disconnect_error(exc):
                    task.session.close()
                    self.discard_session_scores(task.session)
                    continue
                raise

    def _run_train_batch(self, state: ProfileState, tasks: List[TrainTask]) -> None:
        first = tasks[0]
        merged = merge_segments([task.segments for task in tasks])
        total_steps = sum(task.step_count for task in tasks)
        total_episodes = len(tasks)
        started = time.monotonic()
        try:
            archetype_labels = merged[14] if len(merged) > 14 else None
            mcts_visits = merged[15] if len(merged) > 15 else None
            card_belief_labels = merged[16] if len(merged) > 16 else None
            num_archetypes = int(first.headers.get("num_archetypes", "0"))
            card_belief_dim = int(first.headers.get("card_belief_dim", "0"))
            state.learner_context.train_batch(
                merged[0],
                merged[1],
                merged[2],
                merged[3],
                merged[4],
                merged[5],
                merged[6],
                merged[7],
                merged[8],
                merged[9],
                merged[10],
                merged[11],
                merged[12],
                merged[13],
                total_steps,
                int(first.headers.get("seq_len", "0")),
                int(first.headers.get("d_model", "0")),
                int(first.headers.get("max_candidates", "0")),
                int(first.headers.get("cand_feat_dim", "0")),
                archetype_labels,
                num_archetypes,
                mcts_visits,
                card_belief_labels,
                card_belief_dim,
            )
        finally:
            # Release cached GPU memory after every training batch so ONNX
            # inference (running in the JVM on the same GPU) isn't starved.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        finished = time.monotonic()
        service_ms = max(0.0, (finished - started) * 1000.0)
        print(f"[TRAIN_DIAG] episodes={total_episodes} steps={total_steps} "
              f"service={service_ms:.0f}ms profile={state.infer_context.profile_id}", flush=True)
        latency_samples = [max(0.0, (finished - task.enqueued_at) * 1000.0) for task in tasks]
        with self._lock:
            self._train_batches += 1
            self._train_service_ms.append(service_ms)
            self._train_batch_episode_counts.append(float(len(tasks)))
            self._train_batch_step_counts.append(float(total_steps))
            for latency_ms in latency_samples:
                self._train_latency_ms.append(latency_ms)
        try:
            self._maybe_publish_latest_model(state, now=finished)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._train_failures += 1

    @staticmethod
    def _avg(values: Deque[float]) -> float:
        return (sum(values) / float(len(values))) if values else 0.0

    @staticmethod
    def _percentile(values: Deque[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        rank = (percentile / 100.0) * (len(ordered) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        weight = rank - lower
        return (ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight)

    @staticmethod
    def _oldest_age_ms(tasks: List[float]) -> float:
        if not tasks:
            return 0.0
        now = time.monotonic()
        return max(0.0, (now - min(tasks)) * 1000.0)

    def install_sigterm_handler(self) -> None:
        def handler(signum, frame):
            print("[GPU_HOST] SIGTERM received, saving all profiles...", flush=True)
            saves = 0
            with self._lock:
                profiles = list(self._profiles.items())
            for profile_id, state in profiles:
                try:
                    state.learner_context.save_model("")
                    state.learner_context.save_mulligan_model()
                    saves += 1
                    print(f"[GPU_HOST] SIGTERM: saved profile {profile_id}", flush=True)
                except Exception as exc:
                    print(f"[GPU_HOST] SIGTERM: failed to save profile {profile_id}: {exc}", flush=True)
            with self._lock:
                self._sigterm_saves += saves
            self._shutdown_event.set()
            self._running = False
            print(f"[GPU_HOST] SIGTERM: saved {saves} profile(s), shutting down", flush=True)
        signal.signal(signal.SIGTERM, handler)

    def handle_control(self, session: ConnectionSession, opcode: int, request_id: int,
                       headers: Dict[str, str], payload: bytes) -> None:
        profile_id = headers.get("profile_id", "").strip()
        state = self.require_profile(profile_id)
        if opcode == 4:
            state.learner_context.save_model(headers.get("path", ""))
            session.reply(0, request_id)
        elif opcode == 5:
            session.reply(0, request_id, {"device_info": state.infer_context.get_device_info()})
        elif opcode == 6:
            session.reply(0, request_id, {k: str(v) for k, v in state.learner_context.get_main_stats().items()})
        elif opcode == 7:
            session.reply(0, request_id, {k: str(v) for k, v in state.learner_context.get_mulligan_stats().items()})
        elif opcode == 8:
            learner_stats = state.learner_context.get_health_stats()
            infer_stats = state.infer_context.get_health_stats()
            stats = {k: str(max(int(learner_stats.get(k, 0)), int(infer_stats.get(k, 0)))) for k in set(learner_stats) | set(infer_stats)}
            with self._lock:
                stats["train_queue_depth"] = str(len(state.pending_trains))
                stats["dropped_train_episodes"] = str(self._train_tasks_dropped)
                stats["train_vram_requeues"] = str(self._train_requeues_vram)
                stats["train_gpu_active"] = str(self._active_train_batches)
            session.reply(0, request_id, stats)
        elif opcode == 9:
            state.learner_context.reset_health_stats()
            state.infer_context.reset_health_stats()
            session.reply(0, request_id)
        elif opcode == 10:
            state.learner_context.record_game_result(float(headers.get("last_value_prediction", "0")), headers.get("won", "false").lower() == "true")
            session.reply(0, request_id)
        elif opcode == 11:
            features = feature_array_from_bytes(unpack_segments(payload)[0])
            value = state.infer_context.predict_mulligan(features)
            session.reply(0, request_id, {"value": str(value)})
        elif opcode == 12:
            features = feature_array_from_bytes(unpack_segments(payload)[0])
            scores = state.infer_context.predict_mulligan_scores(features)
            session.reply(0, request_id, payload=scores)
        elif opcode == 13:
            segments = unpack_segments(payload)
            state.learner_context.train_mulligan(segments[0], segments[1], segments[2], segments[3], segments[4], segments[5],
                                         int(headers.get("batch_size", "0")))
            state.mulligan_train_count += 1
            if state.mulligan_train_count % state.mulligan_save_interval == 0:
                state.learner_context.save_mulligan_model()
                with self._lock:
                    self._mulligan_saves += 1
            session.reply(0, request_id)
        elif opcode == 14:
            state.learner_context.save_mulligan_model()
            session.reply(0, request_id)
        elif opcode == 15:
            metrics = state.learner_context.get_value_head_metrics()
            session.reply(0, request_id, {k: str(v) for k, v in metrics.items()})
        elif opcode == 16:
            session.reply(0, request_id)
        elif opcode == 17:
            segments = unpack_segments(payload)
            self._maybe_reload_inference_model(state)
            probs = state.infer_context.predict_archetype(
                segments[0],
                segments[1],
                segments[2],
                int(headers.get("batch_size", "1")),
                int(headers.get("seq_len", "0")),
                int(headers.get("d_model", "0")),
            )
            num_archetypes = int(headers.get("num_archetypes", "0"))
            session.reply(0, request_id, {"num_archetypes": str(num_archetypes)}, payload=probs)
        elif opcode == 18:
            segments = unpack_segments(payload)
            self._maybe_reload_inference_model(state)
            card_belief_dim = int(headers.get("card_belief_dim", "0"))
            preds = state.infer_context.predict_card_belief(
                segments[0],
                segments[1],
                segments[2],
                int(headers.get("batch_size", "1")),
                int(headers.get("seq_len", "0")),
                int(headers.get("d_model", "0")),
                card_belief_dim,
            )
            session.reply(0, request_id, {"card_belief_dim": str(card_belief_dim)}, payload=preds)
        else:
            raise ValueError(f"unknown control opcode {opcode}")

    def render_metrics(self) -> str:
        with self._lock:
            pending_scores = sum(task.batch_size for state in self._profiles.values() for task in state.pending_scores)
            pending_trains = sum(len(state.pending_trains) for state in self._profiles.values())
            pending_score_times = [task.enqueued_at for state in self._profiles.values() for task in state.pending_scores]
            pending_train_times = [task.enqueued_at for state in self._profiles.values() for task in state.pending_trains]
            lines = [
                "# HELP gpu_service_batch_timeout_ms Shared GPU host inference batch timeout in milliseconds",
                "# TYPE gpu_service_batch_timeout_ms gauge",
                f"gpu_service_batch_timeout_ms {self.batch_timeout_ms}",
                "# HELP gpu_service_batch_max_size Shared GPU host inference batch max size",
                "# TYPE gpu_service_batch_max_size gauge",
                f"gpu_service_batch_max_size {self.batch_max_size}",
                "# HELP gpu_service_profiles Number of registered profile contexts",
                "# TYPE gpu_service_profiles gauge",
                f"gpu_service_profiles {len(self._profiles)}",
                "# HELP gpu_service_pending_scores Total pending inference requests",
                "# TYPE gpu_service_pending_scores gauge",
                f"gpu_service_pending_scores {pending_scores}",
                "# HELP gpu_service_pending_scores_oldest_ms Age of the oldest pending inference request in milliseconds",
                "# TYPE gpu_service_pending_scores_oldest_ms gauge",
                f"gpu_service_pending_scores_oldest_ms {self._oldest_age_ms(pending_score_times):.3f}",
                "# HELP gpu_service_pending_trains Total pending train episodes",
                "# TYPE gpu_service_pending_trains gauge",
                f"gpu_service_pending_trains {pending_trains}",
                "# HELP gpu_service_pending_train_max Per-profile pending train queue cap",
                "# TYPE gpu_service_pending_train_max gauge",
                f"gpu_service_pending_train_max {self.pending_train_max}",
                "# HELP gpu_service_train_gpu_max_concurrent Maximum concurrent CUDA train batches allowed by shared host",
                "# TYPE gpu_service_train_gpu_max_concurrent gauge",
                f"gpu_service_train_gpu_max_concurrent {self.train_gpu_max_concurrent}",
                "# HELP gpu_service_train_gpu_active Current active train batches consuming learner GPU",
                "# TYPE gpu_service_train_gpu_active gauge",
                f"gpu_service_train_gpu_active {self._active_train_batches}",
                "# HELP gpu_service_pending_trains_oldest_ms Age of the oldest pending train request in milliseconds",
                "# TYPE gpu_service_pending_trains_oldest_ms gauge",
                f"gpu_service_pending_trains_oldest_ms {self._oldest_age_ms(pending_train_times):.3f}",
                "# HELP gpu_service_score_batches_total Score batches executed by shared GPU host",
                "# TYPE gpu_service_score_batches_total counter",
                f"gpu_service_score_batches_total {self._score_batches}",
                "# HELP gpu_service_infer_flush_timeout_total Inference batches flushed by timeout on the shared GPU host",
                "# TYPE gpu_service_infer_flush_timeout_total counter",
                f"gpu_service_infer_flush_timeout_total {self._score_flush_timeout_total}",
                "# HELP gpu_service_infer_flush_full_total Inference batches flushed because the batch max size was reached on the shared GPU host",
                "# TYPE gpu_service_infer_flush_full_total counter",
                f"gpu_service_infer_flush_full_total {self._score_flush_full_total}",
                "# HELP gpu_service_infer_flush_eager_total Inference batches flushed eagerly when aggregate queue exceeds batch max",
                "# TYPE gpu_service_infer_flush_eager_total counter",
                f"gpu_service_infer_flush_eager_total {self._score_flush_eager_total}",
                "# HELP gpu_service_infer_batch_avg_size Average recent shared-host inference batch size",
                "# TYPE gpu_service_infer_batch_avg_size gauge",
                f"gpu_service_infer_batch_avg_size {self._avg(self._infer_batch_sizes):.3f}",
                "# HELP gpu_service_infer_batch_p50_size 50th percentile recent shared-host inference batch size",
                "# TYPE gpu_service_infer_batch_p50_size gauge",
                f"gpu_service_infer_batch_p50_size {self._percentile(self._infer_batch_sizes, 50):.3f}",
                "# HELP gpu_service_infer_batch_p95_size 95th percentile recent shared-host inference batch size",
                "# TYPE gpu_service_infer_batch_p95_size gauge",
                f"gpu_service_infer_batch_p95_size {self._percentile(self._infer_batch_sizes, 95):.3f}",
                "# HELP gpu_service_infer_latency_recent_avg_ms Average recent shared-host inference request latency in milliseconds",
                "# TYPE gpu_service_infer_latency_recent_avg_ms gauge",
                f"gpu_service_infer_latency_recent_avg_ms {self._avg(self._infer_latency_ms):.3f}",
                "# HELP gpu_service_infer_latency_p50_ms 50th percentile recent shared-host inference request latency in milliseconds",
                "# TYPE gpu_service_infer_latency_p50_ms gauge",
                f"gpu_service_infer_latency_p50_ms {self._percentile(self._infer_latency_ms, 50):.3f}",
                "# HELP gpu_service_infer_latency_p95_ms 95th percentile recent shared-host inference request latency in milliseconds",
                "# TYPE gpu_service_infer_latency_p95_ms gauge",
                f"gpu_service_infer_latency_p95_ms {self._percentile(self._infer_latency_ms, 95):.3f}",
                "# HELP gpu_service_infer_service_recent_avg_ms Average recent shared-host inference batch service time in milliseconds",
                "# TYPE gpu_service_infer_service_recent_avg_ms gauge",
                f"gpu_service_infer_service_recent_avg_ms {self._avg(self._infer_service_ms):.3f}",
                "# HELP gpu_service_infer_service_p50_ms 50th percentile recent shared-host inference batch service time in milliseconds",
                "# TYPE gpu_service_infer_service_p50_ms gauge",
                f"gpu_service_infer_service_p50_ms {self._percentile(self._infer_service_ms, 50):.3f}",
                "# HELP gpu_service_infer_service_p95_ms 95th percentile recent shared-host inference batch service time in milliseconds",
                "# TYPE gpu_service_infer_service_p95_ms gauge",
                f"gpu_service_infer_service_p95_ms {self._percentile(self._infer_service_ms, 95):.3f}",
                "# HELP gpu_service_train_batches_total Train batches executed by shared GPU host",
                "# TYPE gpu_service_train_batches_total counter",
                f"gpu_service_train_batches_total {self._train_batches}",
                "# HELP gpu_service_score_failures_total Score batches that failed after dequeue",
                "# TYPE gpu_service_score_failures_total counter",
                f"gpu_service_score_failures_total {self._score_failures}",
                "# HELP gpu_service_train_failures_total Train batches that failed after dequeue",
                "# TYPE gpu_service_train_failures_total counter",
                f"gpu_service_train_failures_total {self._train_failures}",
                "# HELP gpu_service_train_tasks_dropped_total Train tasks dropped by GPU-host backpressure policy",
                "# TYPE gpu_service_train_tasks_dropped_total counter",
                f"gpu_service_train_tasks_dropped_total {self._train_tasks_dropped}",
                "# HELP gpu_service_train_vram_requeues_total Train episodes requeued because learner VRAM guard had no headroom",
                "# TYPE gpu_service_train_vram_requeues_total counter",
                f"gpu_service_train_vram_requeues_total {self._train_requeues_vram}",
                "# HELP gpu_service_train_enqueue_backpressure_wait_total Train enqueue calls that waited for pending queue capacity",
                "# TYPE gpu_service_train_enqueue_backpressure_wait_total counter",
                f"gpu_service_train_enqueue_backpressure_wait_total {self._train_enqueue_waits}",
                "# HELP gpu_service_train_enqueue_backpressure_wait_ms_total Total milliseconds producers waited for pending train capacity",
                "# TYPE gpu_service_train_enqueue_backpressure_wait_ms_total counter",
                f"gpu_service_train_enqueue_backpressure_wait_ms_total {self._train_enqueue_wait_ms:.3f}",
                "# HELP gpu_service_model_publishes_total Learner-to-inference model publish operations",
                "# TYPE gpu_service_model_publishes_total counter",
                f"gpu_service_model_publishes_total {self._model_publishes}",
                "# HELP gpu_service_model_reloads_total Inference model reload operations",
                "# TYPE gpu_service_model_reloads_total counter",
                f"gpu_service_model_reloads_total {self._model_reloads}",
                "# HELP gpu_service_mulligan_saves_total Autonomous mulligan model saves",
                "# TYPE gpu_service_mulligan_saves_total counter",
                f"gpu_service_mulligan_saves_total {self._mulligan_saves}",
                "# HELP gpu_service_sigterm_saves_total Profiles saved on SIGTERM",
                "# TYPE gpu_service_sigterm_saves_total counter",
                f"gpu_service_sigterm_saves_total {self._sigterm_saves}",
                "# HELP gpu_service_train_batch_avg_episodes Average recent shared-host train batch size in episodes",
                "# TYPE gpu_service_train_batch_avg_episodes gauge",
                f"gpu_service_train_batch_avg_episodes {self._avg(self._train_batch_episode_counts):.3f}",
                "# HELP gpu_service_train_batch_avg_steps Average recent shared-host train batch size in steps",
                "# TYPE gpu_service_train_batch_avg_steps gauge",
                f"gpu_service_train_batch_avg_steps {self._avg(self._train_batch_step_counts):.3f}",
                "# HELP gpu_service_train_latency_recent_avg_ms Average recent shared-host train request latency in milliseconds",
                "# TYPE gpu_service_train_latency_recent_avg_ms gauge",
                f"gpu_service_train_latency_recent_avg_ms {self._avg(self._train_latency_ms):.3f}",
                "# HELP gpu_service_train_latency_p50_ms 50th percentile recent shared-host train request latency in milliseconds",
                "# TYPE gpu_service_train_latency_p50_ms gauge",
                f"gpu_service_train_latency_p50_ms {self._percentile(self._train_latency_ms, 50):.3f}",
                "# HELP gpu_service_train_latency_p95_ms 95th percentile recent shared-host train request latency in milliseconds",
                "# TYPE gpu_service_train_latency_p95_ms gauge",
                f"gpu_service_train_latency_p95_ms {self._percentile(self._train_latency_ms, 95):.3f}",
                "# HELP gpu_service_train_service_recent_avg_ms Average recent shared-host train batch service time in milliseconds",
                "# TYPE gpu_service_train_service_recent_avg_ms gauge",
                f"gpu_service_train_service_recent_avg_ms {self._avg(self._train_service_ms):.3f}",
                "# HELP gpu_service_train_service_p50_ms 50th percentile recent shared-host train batch service time in milliseconds",
                "# TYPE gpu_service_train_service_p50_ms gauge",
                f"gpu_service_train_service_p50_ms {self._percentile(self._train_service_ms, 50):.3f}",
                "# HELP gpu_service_train_service_p95_ms 95th percentile recent shared-host train batch service time in milliseconds",
                "# TYPE gpu_service_train_service_p95_ms gauge",
                f"gpu_service_train_service_p95_ms {self._percentile(self._train_service_ms, 95):.3f}",
            ]
            # Saturation metrics
            duty_pct = 0.0
            if self._score_worker_wall_ms > 0:
                duty_pct = min(100.0, self._score_worker_busy_ms / self._score_worker_wall_ms * 100.0)
            window_elapsed = max(0.001, time.monotonic() - self._saturation_window_start)
            window_throughput = self._saturation_window_samples / window_elapsed
            total_elapsed = self._score_worker_wall_ms / 1000.0 if self._score_worker_wall_ms > 0 else 1.0
            avg_throughput = self._score_samples_total / max(0.001, total_elapsed)
            lines.extend([
                "# HELP gpu_service_active_connections Number of active TCP connections to the GPU service",
                "# TYPE gpu_service_active_connections gauge",
                f"gpu_service_active_connections {self._active_connections}",
                "# HELP gpu_service_total_connections_total Total TCP connections accepted",
                "# TYPE gpu_service_total_connections_total counter",
                f"gpu_service_total_connections_total {self._total_connections}",
                "# HELP gpu_service_score_samples_total Total inference samples processed",
                "# TYPE gpu_service_score_samples_total counter",
                f"gpu_service_score_samples_total {self._score_samples_total}",
                "# HELP gpu_service_score_duty_pct Score worker duty cycle (0-100). Higher = more saturated.",
                "# TYPE gpu_service_score_duty_pct gauge",
                f"gpu_service_score_duty_pct {duty_pct:.1f}",
                "# HELP gpu_service_score_throughput_per_sec Current inference throughput (samples/sec)",
                "# TYPE gpu_service_score_throughput_per_sec gauge",
                f"gpu_service_score_throughput_per_sec {window_throughput:.1f}",
                "# HELP gpu_service_score_avg_throughput_per_sec Average inference throughput since start (samples/sec)",
                "# TYPE gpu_service_score_avg_throughput_per_sec gauge",
                f"gpu_service_score_avg_throughput_per_sec {avg_throughput:.1f}",
                f"# HELP gpu_service_role Service role (both or infer)",
                f"# TYPE gpu_service_role gauge",
                f'gpu_service_role{{role="{self.service_role}"}} 1',
                "# HELP gpu_service_score_worker_threads Number of active score worker threads",
                "# TYPE gpu_service_score_worker_threads gauge",
                f"gpu_service_score_worker_threads {self._score_worker_count}",
            ])
            if self._last_error:
                lines.extend([
                    "# HELP gpu_service_last_error_info Last scheduler error as an info metric",
                    "# TYPE gpu_service_last_error_info gauge",
                    f'gpu_service_last_error_info{{message="{self._last_error.replace(chr(34), chr(39))}"}} 1',
                ])
            return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    host_ref: Optional[SharedGpuHost] = None

    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        host = self.host_ref
        payload = host.render_metrics().encode("utf-8") if host is not None else b""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def connection_loop(host: SharedGpuHost, session: ConnectionSession) -> None:
    with host._lock:
        host._active_connections += 1
        host._total_connections += 1
    current_request_id = -1
    try:
        while True:
            opcode, request_id, headers, payload = read_request(session.sock)
            current_request_id = request_id
            if opcode == 1:
                print(f"[CONN] register_profile request: {headers.get('profile_id', '?')}", flush=True)
                try:
                    state = host.get_or_create_profile(headers)
                except Exception as reg_exc:
                    import traceback as _tb
                    print(f"[CONN] REGISTER FAILED: {reg_exc}", flush=True)
                    _tb.print_exc()
                    raise
                state = state  # no-op, just to keep the variable
                session.profile_id = state.infer_context.profile_id
                session.reply(0, request_id, {"device_info": state.infer_context.get_device_info()})
            elif opcode == 2:
                profile_id = headers.get("profile_id", "").strip()
                state = host.require_profile(profile_id)
                host.enqueue_score(state, ScoreTask(session=session, request_id=request_id, profile_id=profile_id,
                                                    headers=headers, segments=unpack_segments(payload)))
            elif opcode == 3:
                profile_id = headers.get("profile_id", "").strip()
                state = host.require_profile(profile_id)
                queued, dropped, waited_ms = host.enqueue_train(
                    state,
                    TrainTask(profile_id=profile_id, headers=headers, segments=unpack_segments(payload))
                )
                with host._lock:
                    queue_depth = len(state.pending_trains)
                    dropped_total = host._train_tasks_dropped
                session.reply(0, request_id, {
                    "queued": "1" if queued else "0",
                    "queue_depth": str(queue_depth),
                    "dropped_train_episodes": str(dropped_total),
                    "enqueue_wait_ms": f"{waited_ms:.3f}",
                    "dropped_this_request": str(dropped),
                })
            else:
                host.handle_control(session, opcode, request_id, headers, payload)
            current_request_id = -1
    except Exception as exc:
        if not _is_session_disconnect_error(exc):
            host._last_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            if current_request_id >= 0 and not _session_is_closed(session):
                try:
                    error_headers = {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                    session.reply(1, current_request_id, error_headers)
                except Exception:
                    pass
    finally:
        with host._lock:
            host._active_connections = max(0, host._active_connections - 1)
        session.close()
        host.discard_session_scores(session)


def _autoscale_score_workers(host: SharedGpuHost) -> None:
    """Auto-scale score workers to match the number of registered profiles."""
    while host._running:
        time.sleep(2.0)
        with host._lock:
            profile_count = len(host._profiles)
            current = host._score_worker_count
        if profile_count > current:
            for i in range(current, profile_count):
                threading.Thread(
                    target=host.score_worker_loop, args=(i,),
                    name=f"GpuServiceScore-{i}", daemon=True,
                ).start()
            host._score_worker_count = profile_count
            print(f"[GPU_HOST] auto-scaled score workers: {current} -> {profile_count}", flush=True)


def _launch_score_workers(host: SharedGpuHost, count: int) -> None:
    for i in range(count):
        threading.Thread(
            target=host.score_worker_loop, args=(i,),
            name=f"GpuServiceScore-{i}", daemon=True,
        ).start()
    host._score_worker_count = count
    print(f"[GPU_HOST] launched {count} score worker threads", flush=True)


def main() -> int:
    host = SharedGpuHost()
    if host.infer_cuda_device or host.train_cuda_device:
        print(f"[GPU_HOST] split-device mode: infer={host.infer_cuda_device or 'default'} "
              f"train={host.train_cuda_device or 'default'}", flush=True)
    initial_workers = host.score_worker_threads if host.score_worker_threads > 0 else 1
    _launch_score_workers(host, initial_workers)
    for tw in range(host.train_worker_threads):
        threading.Thread(
            target=host.train_worker_loop, args=(tw,),
            name=f"GpuServiceTrain-{tw}", daemon=True,
        ).start()
    print(f"[GPU_HOST] launched {host.train_worker_threads} train worker threads", flush=True)
    if host.service_role == "infer":
        print(f"[GPU_HOST] inference-priority mode (GPU_SERVICE_ROLE=infer): training yields to inference", flush=True)
    # Auto-scale score workers when profiles register (if SCORE_WORKER_THREADS=0/auto)
    if host._score_worker_threads_raw == 0:
        threading.Thread(
            target=_autoscale_score_workers, args=(host,),
            name="GpuServiceScoreAutoscale", daemon=True,
        ).start()

    if host.metrics_port > 0:
        MetricsHandler.host_ref = host
        metrics_server = ThreadingHTTPServer((host.bind_host, host.metrics_port), MetricsHandler)
        threading.Thread(target=metrics_server.serve_forever, name="GpuServiceMetrics", daemon=True).start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host.bind_host, host.port))
        listener.listen()
        listener.settimeout(1.0)
        host.install_sigterm_handler()
        print(
            f"[GPU_HOST] listening host={host.bind_host} port={host.port} "
            f"metrics_port={host.metrics_port} batch_timeout_ms={host.batch_timeout_ms} "
            f"batch_max_size={host.batch_max_size}",
            flush=True,
        )
        while not host._shutdown_event.is_set():
            try:
                client, _addr = listener.accept()
            except socket.timeout:
                continue
            session = ConnectionSession(sock=client)
            threading.Thread(target=connection_loop, args=(host, session), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
