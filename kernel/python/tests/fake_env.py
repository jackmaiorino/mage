from __future__ import annotations

import json
import os
import sys
import time

from fixtures import decision_response, terminal_response


def emit(value: dict) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    start_marker = os.environ.get("FAKE_START_MARKER")
    if start_marker:
        with open(start_marker, "w", encoding="utf-8") as fh:
            fh.write("started\n")
    scenario = os.environ.get("FAKE_SCENARIO", "valid")
    if scenario == "timeout":
        time.sleep(60)
        return 0
    if scenario == "eof_nonzero":
        return 17
    count = 0
    episode_steps: dict[int, int] = {}
    for line in sys.stdin:
        req = json.loads(line)
        count += 1
        if scenario == "duplicate_keys":
            sys.stdout.write('{"response_type":"decision","response_type":"decision"}\n')
            sys.stdout.flush()
            continue
        if scenario == "nonfinite_json":
            sys.stdout.write('{"response_type":"decision","schema_version":NaN}\n')
            sys.stdout.flush()
            continue
        if scenario == "nonfinite_overflow":
            sys.stdout.write('{"response_type":"decision","schema_version":1e999}\n')
            sys.stdout.flush()
            continue
        if scenario == "noise":
            sys.stdout.write("not json\n")
            sys.stdout.flush()
            continue
        if scenario == "error_valid":
            emit({"response_type": "error", "schema_version": 2, "request_id": req["request_id"], "error": {"code": "bad_request", "message": "line one\nline two"}})
            continue
        if scenario == "error_bad_schema":
            emit({"response_type": "error", "schema_version": 1, "request_id": req["request_id"], "error": {"code": "bad_request", "message": "bad"}})
            continue
        if scenario == "error_bad_request_id":
            emit({"response_type": "error", "schema_version": 2, "request_id": "wrong", "error": {"code": "bad_request", "message": "bad"}})
            continue
        if scenario == "error_empty_code":
            emit({"response_type": "error", "schema_version": 2, "request_id": req["request_id"], "error": {"code": "", "message": "bad"}})
            continue
        if req["request_type"] == "reset":
            episode_steps[req["episode_id"]] = 0
            if scenario == "train_pair_slow":
                time.sleep(2.0)
            if scenario == "train_zero_learner":
                learner = "p0" if req["episode_id"] % 2 == 0 else "p1"
                actor = "p1" if learner == "p0" else "p0"
            else:
                actor = "p0"
            resp = decision_response(req["request_id"], req["episode_id"], 0, actor=actor)
            if scenario == "extra_field":
                resp["extra"] = True
            elif scenario == "missing_field":
                del resp["reward"]
            elif scenario == "bool_int":
                resp["schema_version"] = True
            elif scenario == "u64_overflow":
                resp["episode_id"] = 18446744073709551616
            elif scenario == "episode_drift":
                resp["episode_id"] = req["episode_id"] + 1
            elif scenario == "step_drift":
                resp["step"] = 1
                resp["observation"]["step_index"] = 1
            elif scenario == "empty_actions":
                resp["legal_actions"] = []
            elif scenario == "noncontiguous_actions":
                resp["legal_actions"][1]["selected_index"] = 9
            elif scenario == "duplicate_actions":
                resp["legal_actions"][1]["stable_id"] = resp["legal_actions"][0]["stable_id"]
            elif scenario == "mismatched_action_actor":
                resp["legal_actions"][1]["semantic"]["actor"] = "p1"
            elif scenario == "mixed_action_actors":
                resp["legal_actions"].append(resp["legal_actions"][1].copy())
                resp["legal_actions"][3] = json.loads(json.dumps(resp["legal_actions"][3]))
                resp["legal_actions"][3]["selected_index"] = 3
                resp["legal_actions"][3]["stable_id"] = "legal-action-v2:d"
                resp["legal_actions"][3]["semantic"]["actor"] = "p1"
            elif scenario == "nonzero_reward":
                resp["reward"] = [1, 0]
            emit(resp)
        else:
            if scenario == "train_pair_assert_latest0" and count == 2:
                latest_path = os.environ.get("FAKE_EXPECT_LATEST_JSON")
                if not latest_path:
                    raise RuntimeError("FAKE_EXPECT_LATEST_JSON not set")
                with open(latest_path, "r", encoding="utf-8") as fh:
                    latest = json.load(fh)
                if latest.get("update") != 0:
                    raise RuntimeError(f"latest.json update was not 0 before first action: {latest!r}")
            expected_step = req["expected_step"] + 1
            episode_steps[req["episode_id"]] = expected_step
            if scenario in ("train_pair", "train_pair_assert_latest0", "train_pair_slow"):
                if expected_step == 1:
                    emit(decision_response(req["request_id"], req["episode_id"], expected_step, actor="p1"))
                else:
                    outcomes = ["p0_win", "p1_win", "draw"]
                    emit(terminal_response(req["request_id"], req["episode_id"], expected_step, outcome=outcomes[req["episode_id"] % 3]))
            elif scenario == "train_zero_learner":
                outcomes = ["p0_win", "p1_win", "draw"]
                emit(terminal_response(req["request_id"], req["episode_id"], expected_step, outcome=outcomes[req["episode_id"] % 3]))
            elif scenario == "train_late_fault":
                if req["episode_id"] == 1:
                    resp = terminal_response(req["request_id"], req["episode_id"], expected_step)
                    resp["terminal_outcome"] = "halted"
                    resp["terminal_classification"] = "halted"
                    resp["terminal_code"] = "fail_closed"
                    resp["winner"] = None
                    resp["terminal_reward"] = [0, 0]
                    emit(resp)
                else:
                    emit(terminal_response(req["request_id"], req["episode_id"], expected_step))
            elif scenario == "provenance_drift":
                resp = decision_response(req["request_id"], req["episode_id"], req["expected_step"] + 1)
                resp["provenance"]["card_db_hash"] += 1
                emit(resp)
            elif scenario == "invalid_terminal":
                resp = terminal_response(req["request_id"], req["episode_id"], req["expected_step"] + 1)
                resp["winner"] = "p1"
                emit(resp)
            elif scenario == "halted_terminal":
                resp = terminal_response(req["request_id"], req["episode_id"], req["expected_step"] + 1)
                resp["terminal_outcome"] = "halted"
                resp["terminal_classification"] = "halted"
                resp["terminal_code"] = "fail_closed"
                resp["winner"] = None
                resp["terminal_reward"] = [0, 0]
                emit(resp)
            elif scenario == "truncated_terminal":
                resp = terminal_response(req["request_id"], req["episode_id"], req["expected_step"] + 1)
                resp["terminal_outcome"] = "truncated"
                resp["terminal_classification"] = "truncated"
                resp["terminal_code"] = "decision_cap"
                resp["winner"] = None
                resp["terminal_reward"] = [0, 0]
                emit(resp)
            elif scenario == "terminal_jump":
                emit(terminal_response(req["request_id"], req["episode_id"], req["expected_step"] + 2))
            else:
                emit(terminal_response(req["request_id"], req["episode_id"], req["expected_step"] + 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
