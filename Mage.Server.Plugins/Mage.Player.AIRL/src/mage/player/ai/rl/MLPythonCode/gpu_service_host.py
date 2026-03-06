#!/usr/bin/env python3
import os
import socket
import struct
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Deque, Dict, List, Optional, Tuple

from gpu_service_core import ProfileContext, feature_array_from_bytes, merge_segments


FRAME_MAX_BYTES = 256 * 1024 * 1024


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


@dataclass
class ConnectionSession:
    sock: socket.socket
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    profile_id: Optional[str] = None

    def reply(self, status: int, request_id: int, headers: Optional[Dict[str, str]] = None, payload: bytes = b"") -> None:
        write_response(self.sock, self.send_lock, status, request_id, headers, payload)

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


@dataclass
class ScoreTask:
    session: ConnectionSession
    request_id: int
    profile_id: str
    headers: Dict[str, str]
    segments: List[bytes]
    enqueued_at: float = field(default_factory=time.time)

    @property
    def batch_key(self) -> Tuple[str, str, str, str, str, str, str, str, str]:
        return (
            self.profile_id,
            self.headers.get("policy_key", "train"),
            self.headers.get("head_id", "action"),
            self.headers.get("pick_index", "0"),
            self.headers.get("min_targets", "0"),
            self.headers.get("max_targets", "0"),
            self.headers.get("seq_len", "0"),
            self.headers.get("d_model", "0"),
            self.headers.get("max_candidates", "0"),
        )

    @property
    def batch_size(self) -> int:
        return int(self.headers.get("batch_size", "1"))


@dataclass
class TrainTask:
    profile_id: str
    headers: Dict[str, str]
    segments: List[bytes]

    @property
    def step_count(self) -> int:
        return int(self.headers.get("batch_size", "0"))


@dataclass
class ProfileState:
    context: ProfileContext
    pending_scores: Deque[ScoreTask] = field(default_factory=deque)
    pending_trains: Deque[TrainTask] = field(default_factory=deque)


class SharedGpuHost:
    def __init__(self) -> None:
        self.bind_host = env_str("GPU_SERVICE_BIND_HOST", "127.0.0.1")
        self.port = env_int("GPU_SERVICE_PORT", 26100)
        self.metrics_port = env_int("GPU_SERVICE_METRICS_PORT", 27100)
        self.batch_timeout_ms = max(1, env_int("PY_BATCH_TIMEOUT_MS", 200))
        self.batch_timeout_s = self.batch_timeout_ms / 1000.0
        self.batch_max_size = max(1, env_int("PY_BATCH_MAX_SIZE", 64))
        self.learner_batch_max_episodes = max(1, env_int("LEARNER_BATCH_MAX_EPISODES", 8))
        self.learner_batch_max_steps = max(1, env_int("LEARNER_BATCH_MAX_STEPS", 4096))
        self.train_multi_max_steps = max(1, env_int("TRAIN_MULTI_MAX_STEPS", self.learner_batch_max_steps))
        self.train_slices_per_tick = max(1, env_int("GPU_SERVICE_SCHED_TRAIN_SLICES_PER_TICK", 1))
        self._profiles: Dict[str, ProfileState] = {}
        self._lock = threading.Condition()
        self._running = True
        self._train_rr = 0
        self._score_batches = 0
        self._train_batches = 0
        self._last_error = ""

    def get_or_create_profile(self, headers: Dict[str, str]) -> ProfileState:
        profile_id = headers.get("profile_id", "").strip()
        if not profile_id:
            raise ValueError("register_profile missing profile_id")
        with self._lock:
            existing = self._profiles.get(profile_id)
        if existing is not None:
            return existing
        context = ProfileContext(profile_id, headers)
        state = ProfileState(context=context)
        with self._lock:
            return self._profiles.setdefault(profile_id, state)

    def require_profile(self, profile_id: str) -> ProfileState:
        with self._lock:
            state = self._profiles.get(profile_id)
        if state is None:
            raise ValueError(f"profile {profile_id} is not registered")
        return state

    def enqueue_score(self, state: ProfileState, task: ScoreTask) -> None:
        with self._lock:
            state.pending_scores.append(task)
            self._lock.notify_all()

    def enqueue_train(self, state: ProfileState, task: TrainTask) -> None:
        with self._lock:
            state.pending_trains.append(task)
            self._lock.notify_all()

    def scheduler_loop(self) -> None:
        while self._running:
            work = None
            sleep_for = 0.25
            with self._lock:
                work, sleep_for = self._select_work_locked()
                if work is None:
                    self._lock.wait(timeout=sleep_for)
                    continue
            kind, state, tasks, reason = work
            try:
                if kind == "score":
                    self._run_score_batch(state, tasks, reason)
                else:
                    self._run_train_batch(state, tasks)
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                if kind == "score":
                    for task in tasks:
                        try:
                            task.session.reply(1, task.request_id, {"error": str(exc)})
                        except Exception:
                            pass

    def _select_work_locked(self):
        now = time.time()
        profiles = list(self._profiles.values())
        best_score_state = None
        best_age = -1.0
        oldest_age = None
        for state in profiles:
            if not state.pending_scores:
                continue
            age = now - state.pending_scores[0].enqueued_at
            if oldest_age is None or age > oldest_age:
                oldest_age = age
            if age >= self.batch_timeout_s or len(state.pending_scores) >= self.batch_max_size:
                if age > best_age:
                    best_age = age
                    best_score_state = state
        if best_score_state is not None:
            return ("score", best_score_state, self._pop_score_batch_locked(best_score_state), "timeout" if best_age >= self.batch_timeout_s else "full"), 0.0

        train_work = self._pop_train_batch_locked()
        if train_work is not None:
            state, tasks = train_work
            return ("train", state, tasks, ""), 0.0

        if oldest_age is not None:
            wait = max(0.01, min(0.25, self.batch_timeout_s - oldest_age))
            return None, wait
        return None, 0.25

    def _pop_score_batch_locked(self, state: ProfileState) -> List[ScoreTask]:
        first = state.pending_scores.popleft()
        tasks = [first]
        keep: Deque[ScoreTask] = deque()
        while state.pending_scores:
            task = state.pending_scores.popleft()
            if len(tasks) < self.batch_max_size and task.batch_key == first.batch_key:
                tasks.append(task)
            else:
                keep.append(task)
        state.pending_scores = keep
        return tasks

    def _pop_train_batch_locked(self) -> Optional[Tuple[ProfileState, List[TrainTask]]]:
        profiles = list(self._profiles.values())
        if not profiles:
            return None
        for offset in range(len(profiles)):
            idx = (self._train_rr + offset) % len(profiles)
            state = profiles[idx]
            if not state.pending_trains:
                continue
            self._train_rr = (idx + 1) % len(profiles)
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
            return state, tasks
        return None

    def _run_score_batch(self, state: ProfileState, tasks: List[ScoreTask], reason: str) -> None:
        first = tasks[0]
        headers = first.headers
        merged = merge_segments([task.segments for task in tasks])
        result_bytes = state.context.score_batch(
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
            len(tasks),
            int(headers.get("seq_len", "0")),
            int(headers.get("d_model", "0")),
            int(headers.get("max_candidates", "0")),
            int(headers.get("cand_feat_dim", "0")),
        )
        self._score_batches += 1
        max_candidates = int(headers.get("max_candidates", "0"))
        stride = (max_candidates + 1) * 4
        for idx, task in enumerate(tasks):
            start = idx * stride
            end = start + stride
            extra = {
                "host_batch_size": str(len(tasks)),
                "host_flush_reason": reason,
            }
            if idx == 0:
                extra["flush_leader"] = "1"
            task.session.reply(0, task.request_id, extra, result_bytes[start:end])

    def _run_train_batch(self, state: ProfileState, tasks: List[TrainTask]) -> None:
        first = tasks[0]
        merged = merge_segments([task.segments for task in tasks])
        total_steps = sum(task.step_count for task in tasks)
        state.context.train_batch(
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
        )
        self._train_batches += 1

    def handle_control(self, session: ConnectionSession, opcode: int, request_id: int,
                       headers: Dict[str, str], payload: bytes) -> None:
        profile_id = headers.get("profile_id", "").strip()
        state = self.require_profile(profile_id)
        if opcode == 4:
            state.context.save_model(headers.get("path", ""))
            session.reply(0, request_id)
        elif opcode == 5:
            session.reply(0, request_id, {"device_info": state.context.get_device_info()})
        elif opcode == 6:
            session.reply(0, request_id, {k: str(v) for k, v in state.context.get_main_stats().items()})
        elif opcode == 7:
            session.reply(0, request_id, {k: str(v) for k, v in state.context.get_mulligan_stats().items()})
        elif opcode == 8:
            stats = {k: str(v) for k, v in state.context.get_health_stats().items()}
            with self._lock:
                stats["train_queue_depth"] = str(len(state.pending_trains))
                stats["dropped_train_episodes"] = "0"
            session.reply(0, request_id, stats)
        elif opcode == 9:
            state.context.reset_health_stats()
            session.reply(0, request_id)
        elif opcode == 10:
            state.context.record_game_result(float(headers.get("last_value_prediction", "0")), headers.get("won", "false").lower() == "true")
            session.reply(0, request_id)
        elif opcode == 11:
            features = feature_array_from_bytes(unpack_segments(payload)[0])
            value = state.context.predict_mulligan(features)
            session.reply(0, request_id, {"value": str(value)})
        elif opcode == 12:
            features = feature_array_from_bytes(unpack_segments(payload)[0])
            scores = state.context.predict_mulligan_scores(features)
            session.reply(0, request_id, payload=scores)
        elif opcode == 13:
            segments = unpack_segments(payload)
            state.context.train_mulligan(segments[0], segments[1], segments[2], segments[3], segments[4], segments[5],
                                         int(headers.get("batch_size", "0")))
            session.reply(0, request_id)
        elif opcode == 14:
            state.context.save_mulligan_model()
            session.reply(0, request_id)
        elif opcode == 15:
            metrics = state.context.get_value_head_metrics()
            session.reply(0, request_id, {k: str(v) for k, v in metrics.items()})
        elif opcode == 16:
            session.reply(0, request_id)
        else:
            raise ValueError(f"unknown control opcode {opcode}")

    def render_metrics(self) -> str:
        with self._lock:
            lines = [
                "# HELP gpu_service_profiles Number of registered profile contexts",
                "# TYPE gpu_service_profiles gauge",
                f"gpu_service_profiles {len(self._profiles)}",
                "# HELP gpu_service_pending_scores Total pending inference requests",
                "# TYPE gpu_service_pending_scores gauge",
                f"gpu_service_pending_scores {sum(len(state.pending_scores) for state in self._profiles.values())}",
                "# HELP gpu_service_pending_trains Total pending train episodes",
                "# TYPE gpu_service_pending_trains gauge",
                f"gpu_service_pending_trains {sum(len(state.pending_trains) for state in self._profiles.values())}",
                "# HELP gpu_service_score_batches_total Score batches executed by shared GPU host",
                "# TYPE gpu_service_score_batches_total counter",
                f"gpu_service_score_batches_total {self._score_batches}",
                "# HELP gpu_service_train_batches_total Train batches executed by shared GPU host",
                "# TYPE gpu_service_train_batches_total counter",
                f"gpu_service_train_batches_total {self._train_batches}",
            ]
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
    try:
        while True:
            opcode, request_id, headers, payload = read_request(session.sock)
            if opcode == 1:
                state = host.get_or_create_profile(headers)
                session.profile_id = state.context.profile_id
                session.reply(0, request_id, {"device_info": state.context.get_device_info()})
            elif opcode == 2:
                profile_id = headers.get("profile_id", "").strip()
                state = host.require_profile(profile_id)
                host.enqueue_score(state, ScoreTask(session=session, request_id=request_id, profile_id=profile_id,
                                                    headers=headers, segments=unpack_segments(payload)))
            elif opcode == 3:
                profile_id = headers.get("profile_id", "").strip()
                state = host.require_profile(profile_id)
                host.enqueue_train(state, TrainTask(profile_id=profile_id, headers=headers, segments=unpack_segments(payload)))
                with host._lock:
                    queue_depth = len(state.pending_trains)
                session.reply(0, request_id, {"queued": "1", "queue_depth": str(queue_depth), "dropped_train_episodes": "0"})
            else:
                host.handle_control(session, opcode, request_id, headers, payload)
    except Exception as exc:
        try:
            session.reply(1, -1, {"error": str(exc)})
        except Exception:
            pass
    finally:
        session.close()


def main() -> int:
    host = SharedGpuHost()
    scheduler = threading.Thread(target=host.scheduler_loop, name="GpuServiceScheduler", daemon=True)
    scheduler.start()

    if host.metrics_port > 0:
        MetricsHandler.host_ref = host
        metrics_server = ThreadingHTTPServer((host.bind_host, host.metrics_port), MetricsHandler)
        threading.Thread(target=metrics_server.serve_forever, name="GpuServiceMetrics", daemon=True).start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host.bind_host, host.port))
        listener.listen()
        while True:
            client, _addr = listener.accept()
            session = ConnectionSession(sock=client)
            threading.Thread(target=connection_loop, args=(host, session), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
