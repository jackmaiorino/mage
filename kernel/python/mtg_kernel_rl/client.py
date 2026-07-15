"""Strict subprocess JSONL client for the Rust kernel RL environment."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2
PROTOCOL_NAME = "kernel_rl_jsonl"
PROTOCOL_VERSION = 2
MAX_LINE_BYTES = 8 * 1024 * 1024


class KernelRlError(Exception):
    pass


class ProtocolError(KernelRlError):
    pass


class EnvProcessError(KernelRlError):
    pass


@dataclass(frozen=True)
class Decision:
    episode_id: int
    step: int
    acting_player: str
    observation: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Terminal:
    episode_id: int
    terminal_outcome: str
    terminal_classification: str
    terminal_code: str
    winner: str | None
    terminal_reward: list[int]
    decision_count: int
    provenance: dict[str, Any]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ProtocolError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_constant(value: str) -> None:
    raise ProtocolError(f"non-finite JSON constant rejected: {value}")


def strict_json_loads(line: str) -> dict[str, Any]:
    try:
        value = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ProtocolError:
        raise
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"stdout line is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("stdout response is not a JSON object")
    return value


def strict_json_dumps(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _keys(value: dict[str, Any], expected: Iterable[str], context: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    missing = expected_set - actual
    extra = actual - expected_set
    if missing or extra:
        raise ProtocolError(f"{context} fields mismatch: missing={sorted(missing)} extra={sorted(extra)}")


def _int(value: Any, context: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ProtocolError(f"{context} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ProtocolError(f"{context} must be >= {minimum}")
    return value


def _str(value: Any, context: str) -> str:
    if type(value) is not str:
        raise ProtocolError(f"{context} must be a string")
    return value


def _bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ProtocolError(f"{context} must be a bool")
    return value


def _list(value: Any, context: str, length: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise ProtocolError(f"{context} must be a list")
    if length is not None and len(value) != length:
        raise ProtocolError(f"{context} must have length {length}")
    return value


def _optional_seat(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _seat(value, context)


def _seat(value: Any, context: str) -> str:
    seat = _str(value, context)
    if seat not in {"p0", "p1"}:
        raise ProtocolError(f"{context} must be p0 or p1")
    return seat


def _validate_provenance(value: Any, expected: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError("provenance must be an object")
    _keys(value, ["protocol", "protocol_version", "schema_version", "kernel_version", "surface_version", "card_db_hash"], "provenance")
    if value["protocol"] != PROTOCOL_NAME:
        raise ProtocolError("unexpected protocol name")
    if _int(value["protocol_version"], "provenance.protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("unexpected protocol version")
    if _int(value["schema_version"], "provenance.schema_version") != SCHEMA_VERSION:
        raise ProtocolError("unexpected provenance schema version")
    _str(value["kernel_version"], "provenance.kernel_version")
    _int(value["surface_version"], "provenance.surface_version", minimum=0)
    _int(value["card_db_hash"], "provenance.card_db_hash", minimum=0)
    if expected is not None and value != expected:
        raise ProtocolError("provenance drifted within process")
    return value


def _validate_card_ref(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ProtocolError(f"{context} must be an object")
    _keys(value, ["arena_id", "card_db_id", "owner", "controller", "zone", "zone_change_count"], context)
    _int(value["arena_id"], f"{context}.arena_id", minimum=0)
    _int(value["card_db_id"], f"{context}.card_db_id", minimum=0)
    _seat(value["owner"], f"{context}.owner")
    _seat(value["controller"], f"{context}.controller")
    _str(value["zone"], f"{context}.zone")
    _int(value["zone_change_count"], f"{context}.zone_change_count", minimum=0)


def _validate_target_ref(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ProtocolError(f"{context} must be an object")
    kind = _str(value.get("target_kind"), f"{context}.target_kind")
    if kind == "player":
        _keys(value, ["target_kind", "player"], context)
        _seat(value["player"], f"{context}.player")
    elif kind == "object":
        _keys(value, ["target_kind", "object"], context)
        _validate_card_ref(value["object"], f"{context}.object")
    else:
        raise ProtocolError(f"unsupported target_kind {kind}")


def _validate_action_semantic(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ProtocolError(f"{context} must be an object")
    kind = _str(value.get("action_kind"), f"{context}.action_kind")
    if kind == "ambiguous":
        raise ProtocolError("ambiguous action semantic is not admissible")
    fields = {
        "pass": ["action_kind", "actor"],
        "play_land": ["action_kind", "actor", "source"],
        "cast_spell": ["action_kind", "actor", "source"],
        "activate_mana_ability": ["action_kind", "actor", "source"],
        "activate_ability": ["action_kind", "actor", "source", "ability_index"],
        "plot_spell": ["action_kind", "actor", "source"],
        "choose_target": ["action_kind", "actor", "source", "remaining", "target"],
        "choose_cost_target": ["action_kind", "actor", "source", "cost_kind", "remaining", "candidate"],
        "choose_cast_mode": ["action_kind", "actor", "source", "mode"],
        "choose_kicker": ["action_kind", "actor", "source", "pay"],
        "choose_spell_mode": ["action_kind", "actor", "source", "mode_index", "mode_count"],
        "choose_optional_cost_use": ["action_kind", "actor", "use_cost"],
        "choose_optional_cost_which": ["action_kind", "actor", "choice"],
        "choose_madness_cast": ["action_kind", "actor", "card", "cast_it"],
        "discard": ["action_kind", "actor", "cards"],
        "declare_attackers": ["action_kind", "actor", "attackers"],
        "declare_blockers_for_attacker": ["action_kind", "actor", "attacker", "blockers"],
        "order_triggers": ["action_kind", "actor", "pending_sources", "order"],
    }.get(kind)
    if fields is None:
        raise ProtocolError(f"unsupported action_kind {kind}")
    _keys(value, fields, context)
    _seat(value["actor"], f"{context}.actor")
    for field in ("source", "candidate", "card", "attacker"):
        if field in value:
            _validate_card_ref(value[field], f"{context}.{field}")
    if "target" in value:
        _validate_target_ref(value["target"], f"{context}.target")
    for field in ("cards", "attackers", "blockers", "pending_sources"):
        if field in value:
            for i, ref in enumerate(_list(value[field], f"{context}.{field}")):
                _validate_card_ref(ref, f"{context}.{field}[{i}]")
    if "order" in value:
        for i, item in enumerate(_list(value["order"], f"{context}.order")):
            _int(item, f"{context}.order[{i}]", minimum=0)
    for field in ("ability_index", "remaining", "mode_index", "mode_count"):
        if field in value:
            _int(value[field], f"{context}.{field}", minimum=0)
    for field in ("pay", "use_cost", "cast_it"):
        if field in value:
            _bool(value[field], f"{context}.{field}")
    for field in ("mode", "cost_kind", "choice"):
        if field in value:
            _str(value[field], f"{context}.{field}")


def _validate_legal_actions(actions: Any) -> list[dict[str, Any]]:
    legal_actions = _list(actions, "legal_actions")
    if not legal_actions:
        raise ProtocolError("decision has no legal actions")
    seen: set[str] = set()
    for i, action in enumerate(legal_actions):
        if not isinstance(action, dict):
            raise ProtocolError("legal action must be an object")
        _keys(action, ["schema_version", "selected_index", "stable_id", "semantic", "display_text"], f"legal_actions[{i}]")
        if _int(action["schema_version"], f"legal_actions[{i}].schema_version") != SCHEMA_VERSION:
            raise ProtocolError("legal action schema mismatch")
        selected_index = _int(action["selected_index"], f"legal_actions[{i}].selected_index", minimum=0)
        if selected_index != i:
            raise ProtocolError("legal action selected_index is not contiguous")
        stable_id = _str(action["stable_id"], f"legal_actions[{i}].stable_id")
        if stable_id in seen:
            raise ProtocolError("duplicate legal action stable_id")
        seen.add(stable_id)
        if action["display_text"] is not None:
            _str(action["display_text"], f"legal_actions[{i}].display_text")
        _validate_action_semantic(action["semantic"], f"legal_actions[{i}].semantic")
    return legal_actions


def _validate_observation(value: Any, response: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError("observation must be an object")
    _keys(
        value,
        ["schema_version", "kernel_version", "surface_version", "card_db_hash", "acting_player", "step_index", "projection", "own_hand", "visible_projection_hash"],
        "observation",
    )
    if _int(value["schema_version"], "observation.schema_version") != SCHEMA_VERSION:
        raise ProtocolError("observation schema mismatch")
    if value["kernel_version"] != provenance["kernel_version"]:
        raise ProtocolError("observation kernel_version drift")
    if _int(value["surface_version"], "observation.surface_version") != provenance["surface_version"]:
        raise ProtocolError("observation surface_version drift")
    if _int(value["card_db_hash"], "observation.card_db_hash") != provenance["card_db_hash"]:
        raise ProtocolError("observation card_db_hash drift")
    if _seat(value["acting_player"], "observation.acting_player") != response["acting_player"]:
        raise ProtocolError("observation acting_player mismatch")
    if _int(value["step_index"], "observation.step_index", minimum=0) != response["step"]:
        raise ProtocolError("observation step_index mismatch")
    _int(value["visible_projection_hash"], "observation.visible_projection_hash", minimum=0)
    for i, card in enumerate(_list(value["own_hand"], "observation.own_hand")):
        if not isinstance(card, dict):
            raise ProtocolError("own_hand card must be an object")
        _keys(card, ["stable", "card_name"], f"observation.own_hand[{i}]")
        _validate_card_ref(card["stable"], f"observation.own_hand[{i}].stable")
        _str(card["card_name"], f"observation.own_hand[{i}].card_name")
    from .features import assert_observation_classified

    assert_observation_classified(value)
    return value


def _validate_reward(value: Any, context: str) -> list[int]:
    reward = _list(value, context, length=2)
    out = [_int(reward[0], f"{context}[0]"), _int(reward[1], f"{context}[1]")]
    return out


def _stdout_reader(stream: Any, output: queue.Queue[bytes | BaseException], max_line_bytes: int) -> None:
    try:
        while True:
            line = stream.readline()
            if line == b"":
                output.put(b"")
                return
            if len(line) > max_line_bytes:
                output.put(ProtocolError("stdout line exceeds maximum size"))
                return
            output.put(line)
    except BaseException as exc:
        output.put(exc)


def _stderr_reader(stream: Any, chunks: list[bytes]) -> None:
    while True:
        chunk = stream.readline()
        if chunk == b"":
            return
        chunks.append(chunk)


class KernelRlClient:
    def __init__(
        self,
        env_bin: str | os.PathLike[str],
        *,
        timeout_s: float = 5.0,
        max_line_bytes: int = MAX_LINE_BYTES,
    ) -> None:
        self.env_bin = str(Path(env_bin))
        if not Path(self.env_bin).is_file():
            raise FileNotFoundError(self.env_bin)
        self.timeout_s = timeout_s
        self.max_line_bytes = max_line_bytes
        self._proc = subprocess.Popen(
            [self.env_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._stdout: queue.Queue[bytes | BaseException] = queue.Queue()
        self._stderr_chunks: list[bytes] = []
        self._stdout_thread = threading.Thread(
            target=_stdout_reader,
            args=(self._proc.stdout, self._stdout, max_line_bytes),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=_stderr_reader,
            args=(self._proc.stderr, self._stderr_chunks),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._request_counter = 0
        self._provenance: dict[str, Any] | None = None
        self._episode_id: int | None = None
        self._expected_step: int | None = None
        self._closed = False

    def stderr_text(self) -> str:
        return b"".join(self._stderr_chunks).decode("utf-8", errors="replace")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        self._stdout_thread.join(timeout=1)
        self._stderr_thread.join(timeout=1)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def __enter__(self) -> "KernelRlClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"py-{self._request_counter:012d}"

    def reset(self, *, episode_id: int, env_seed: int, max_decisions: int) -> Decision:
        self._episode_id = _int(episode_id, "episode_id", minimum=0)
        self._expected_step = 0
        request = {
            "request_type": "reset",
            "schema_version": SCHEMA_VERSION,
            "request_id": self._next_request_id(),
            "episode_id": episode_id,
            "env_seed": _int(env_seed, "env_seed", minimum=0),
            "max_decisions": _int(max_decisions, "max_decisions", minimum=0),
        }
        response = self._exchange(request)
        if not isinstance(response, Decision):
            raise ProtocolError("reset did not return an initial decision")
        return response

    def step(self, selected_index: int, selected_action_id: str) -> Decision | Terminal:
        if self._episode_id is None or self._expected_step is None:
            raise ProtocolError("step before reset")
        request = {
            "request_type": "step",
            "schema_version": SCHEMA_VERSION,
            "request_id": self._next_request_id(),
            "episode_id": self._episode_id,
            "expected_step": self._expected_step,
            "selected_index": _int(selected_index, "selected_index", minimum=0),
            "selected_action_id": _str(selected_action_id, "selected_action_id"),
        }
        return self._exchange(request)

    def _exchange(self, request: dict[str, Any]) -> Decision | Terminal:
        if self._closed:
            raise EnvProcessError("client is closed")
        if self._proc.poll() is not None:
            raise EnvProcessError(f"environment exited before request: code={self._proc.returncode} stderr={self.stderr_text()!r}")
        assert self._proc.stdin is not None
        self._proc.stdin.write(strict_json_dumps(request))
        self._proc.stdin.flush()
        line = self._read_line()
        response = strict_json_loads(line.decode("utf-8"))
        return self._validate_response(response, request)

    def _read_line(self) -> bytes:
        try:
            item = self._stdout.get(timeout=self.timeout_s)
        except queue.Empty as exc:
            self.close()
            raise EnvProcessError(f"timeout waiting for environment stdout; stderr={self.stderr_text()!r}") from exc
        if isinstance(item, BaseException):
            self.close()
            raise item
        if item == b"":
            code = self._proc.poll()
            raise EnvProcessError(f"environment EOF; code={code} stderr={self.stderr_text()!r}")
        return item.rstrip(b"\r\n")

    def _validate_response(self, response: dict[str, Any], request: dict[str, Any]) -> Decision | Terminal:
        response_type = _str(response.get("response_type"), "response_type")
        if response_type == "error":
            _keys(response, ["response_type", "schema_version", "request_id", "error"], "error response")
            _int(response["schema_version"], "error.schema_version")
            error = response["error"]
            if not isinstance(error, dict):
                raise ProtocolError("error payload must be object")
            _keys(error, ["code", "message"], "error")
            raise ProtocolError(f"environment error {error['code']}: {error['message']}")
        if response_type == "decision":
            _keys(response, ["response_type", "schema_version", "request_id", "provenance", "episode_id", "step", "acting_player", "observation", "legal_actions", "reward"], "decision response")
            if _int(response["schema_version"], "decision.schema_version") != SCHEMA_VERSION:
                raise ProtocolError("decision schema mismatch")
            if _str(response["request_id"], "decision.request_id") != request["request_id"]:
                raise ProtocolError("decision request_id mismatch")
            provenance = _validate_provenance(response["provenance"], self._provenance)
            self._provenance = provenance
            episode_id = _int(response["episode_id"], "decision.episode_id", minimum=0)
            if episode_id != request["episode_id"]:
                raise ProtocolError("decision episode_id mismatch")
            step = _int(response["step"], "decision.step", minimum=0)
            expected_step = 0 if request["request_type"] == "reset" else request["expected_step"] + 1
            if step != expected_step:
                raise ProtocolError(f"decision step drift: expected {expected_step}, got {step}")
            acting = _seat(response["acting_player"], "decision.acting_player")
            observation = _validate_observation(response["observation"], response, provenance)
            legal_actions = _validate_legal_actions(response["legal_actions"])
            if _validate_reward(response["reward"], "decision.reward") != [0, 0]:
                raise ProtocolError("intermediate decision reward must be [0, 0]")
            self._episode_id = episode_id
            self._expected_step = step
            return Decision(episode_id, step, acting, observation, legal_actions, provenance)
        if response_type == "terminal":
            _keys(
                response,
                ["response_type", "schema_version", "request_id", "provenance", "episode_id", "terminal_outcome", "terminal_classification", "terminal_code", "winner", "terminal_reward", "terminal_reason", "decision_count"],
                "terminal response",
            )
            if _int(response["schema_version"], "terminal.schema_version") != SCHEMA_VERSION:
                raise ProtocolError("terminal schema mismatch")
            if _str(response["request_id"], "terminal.request_id") != request["request_id"]:
                raise ProtocolError("terminal request_id mismatch")
            provenance = _validate_provenance(response["provenance"], self._provenance)
            self._provenance = provenance
            episode_id = _int(response["episode_id"], "terminal.episode_id", minimum=0)
            if episode_id != request["episode_id"]:
                raise ProtocolError("terminal episode_id mismatch")
            decision_count = _int(response["decision_count"], "terminal.decision_count", minimum=0)
            minimum_count = 0 if request["request_type"] == "reset" else request["expected_step"] + 1
            if decision_count < minimum_count:
                raise ProtocolError("terminal decision_count regressed")
            outcome = _str(response["terminal_outcome"], "terminal.terminal_outcome")
            classification = _str(response["terminal_classification"], "terminal.terminal_classification")
            code = _str(response["terminal_code"], "terminal.terminal_code")
            reward = _validate_reward(response["terminal_reward"], "terminal.terminal_reward")
            winner = _optional_seat(response["winner"], "terminal.winner")
            _str(response["terminal_reason"], "terminal.terminal_reason")
            self._validate_terminal_tuple(outcome, classification, code, winner, reward)
            self._expected_step = None
            return Terminal(episode_id, outcome, classification, code, winner, reward, decision_count, provenance)
        raise ProtocolError(f"unsupported response_type {response_type}")

    @staticmethod
    def _validate_terminal_tuple(outcome: str, classification: str, code: str, winner: str | None, reward: list[int]) -> None:
        if classification != "natural" or code != "natural_game_over":
            raise ProtocolError("only natural terminal outcomes are admitted")
        expected: dict[str, tuple[str | None, list[int]]] = {
            "p0_win": ("p0", [1, -1]),
            "p1_win": ("p1", [-1, 1]),
            "draw": (None, [0, 0]),
        }
        if outcome not in expected:
            raise ProtocolError("terminal outcome is not a natural win/draw")
        expected_winner, expected_reward = expected[outcome]
        if winner != expected_winner or reward != expected_reward:
            raise ProtocolError("invalid natural terminal winner/reward tuple")
