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
    scenario = os.environ.get("FAKE_SCENARIO", "valid")
    if scenario == "timeout":
        time.sleep(60)
        return 0
    if scenario == "eof_nonzero":
        return 17
    count = 0
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
        if scenario == "noise":
            sys.stdout.write("not json\n")
            sys.stdout.flush()
            continue
        if req["request_type"] == "reset":
            resp = decision_response(req["request_id"], req["episode_id"], 0)
            if scenario == "extra_field":
                resp["extra"] = True
            elif scenario == "missing_field":
                del resp["reward"]
            elif scenario == "bool_int":
                resp["schema_version"] = True
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
            elif scenario == "nonzero_reward":
                resp["reward"] = [1, 0]
            emit(resp)
        else:
            if scenario == "provenance_drift":
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
            else:
                emit(terminal_response(req["request_id"], req["episode_id"], req["expected_step"] + 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
