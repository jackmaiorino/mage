import json, os, glob, re, sys

LANDS = {"Forest", "Swamp"}
BASE = "C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
DIRS = ["baseline_auc_ab", "baseline_auc_s9999", "baseline_auc_big5151"]
SUB = "game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity"

RL_STATE = re.compile(r"PlayerRL1 L(\d+) H\d+(?:\[[^\]]*\])? B\d+(?:\[[^\]]*\])? G(\d+)")

def parse_game(path):
    decisions, result, states, spy_lines = [], None, [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh):
            s = line.strip()
            if s.startswith("REPLAY_DECISION_JSON:"):
                try:
                    decisions.append((ln, json.loads(s[len("REPLAY_DECISION_JSON:"):].strip())))
                except Exception:
                    pass
            elif s.startswith("RESULT:"):
                result = s.split("RESULT:")[1].strip()
            else:
                m = RL_STATE.search(line)
                if m:
                    states.append((ln, int(m.group(1)), int(m.group(2))))
                if "SELECTED: Cast Balustrade Spy" in s:
                    spy_lines.append(ln)
    return decisions, result, states, spy_lines

def main():
    rows = []
    for dd in DIRS:
        gdir = os.path.join(BASE, dd, SUB)
        if not os.path.isdir(gdir):
            continue
        for path in sorted(glob.glob(os.path.join(gdir, "*.txt"))):
            decisions, result, states, spy_lines = parse_game(path)
            spy_decs = [(ln, d) for (ln, d) in decisions
                        if "Balustrade Spy" in (d.get("selected_text") or "")
                        and "cast" in (d.get("selected_text") or "").lower()]
            if not (spy_decs or spy_lines):
                continue
            first_cast_ln = spy_lines[0] if spy_lines else spy_decs[0][0]
            # gy just BEFORE cast
            gy_before = None
            for (ln, lib, gy) in states:
                if ln < first_cast_ln:
                    gy_before = gy
            lands_at_cast = lib_at_cast = None
            if spy_decs:
                lib = spy_decs[0][1].get("library", [])
                lands_at_cast = sum(1 for c in lib if c in LANDS)
                lib_at_cast = spy_decs[0][1].get("library_size", len(lib))
            after = [(ln, lib, gy) for (ln, lib, gy) in states if ln > first_cast_ln]
            max_gy_after = max((gy for (_, _, gy) in after), default=None)
            min_lib_after = min((lib for (_, lib, _) in after), default=None)
            # mill delta = how many cards the spy mill dumped into gy
            mill_delta = (max_gy_after - gy_before) if (max_gy_after is not None and gy_before is not None) else None
            rows.append({"dir": dd, "file": os.path.basename(path),
                         "n_casts": len(spy_decs) if spy_decs else len(spy_lines),
                         "lands_at_cast": lands_at_cast, "lib_at_cast": lib_at_cast,
                         "gy_before": gy_before, "max_gy_after": max_gy_after,
                         "min_lib_after": min_lib_after, "mill_delta": mill_delta,
                         "result": result})
    # TRUE classification: did the mill empty the library?
    # Use max graveyard after cast: full mill => gy reaches >= 45 (library ~49-53, minus a few).
    EMPTIED = 44
    milled = [r for r in rows if r["max_gy_after"] is not None and r["max_gy_after"] >= EMPTIED]
    fizzle = [r for r in rows if r["max_gy_after"] is not None and r["max_gy_after"] < EMPTIED]
    print(f"games w/ spy cast: {len(rows)}")
    print(f"MILLED OUT (maxGyAfter>={EMPTIED}): {len(milled)}  W/L={sum(1 for r in milled if r['result']=='WIN')}/{len(milled)}")
    print(f"FIZZLE (maxGyAfter<{EMPTIED}):     {len(fizzle)}  W/L={sum(1 for r in fizzle if r['result']=='WIN')}/{len(fizzle)}")
    print("\n=== TRUE FIZZLE GAMES (library NOT emptied) ===")
    print(f"{'file':<26} {'casts':>5} {'landsAtCast':>11} {'libAtCast':>9} {'gyBefore':>8} {'maxGyAfter':>10} {'millDelta':>9} {'res':>5}")
    for r in sorted(fizzle, key=lambda x: (x['mill_delta'] if x['mill_delta'] is not None else 0)):
        print(f"{r['dir'][9:]+'/'+r['file'][14:23]:<26} {r['n_casts']:>5} {str(r['lands_at_cast']):>11} "
              f"{str(r['lib_at_cast']):>9} {str(r['gy_before']):>8} {str(r['max_gy_after']):>10} "
              f"{str(r['mill_delta']):>9} {r['result']:>5}")

if __name__ == "__main__":
    main()
