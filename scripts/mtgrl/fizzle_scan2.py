import json, os, glob, re, sys

LANDS = {"Forest", "Swamp"}
BASE = "C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
DIRS = ["baseline_auc_ab", "baseline_auc_s9999", "baseline_auc_big5151"]
SUB = "game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity"

# STATE line format: ... || PlayerRL1 L<lib> H<n>[...] B<n>[...] G<n>[...] X<n>[...] || EvalBot...
# capture PlayerRL1's L (library) and G (graveyard) counts
RL_STATE = re.compile(r"PlayerRL1 L(\d+) H\d+(?:\[[^\]]*\])? B\d+(?:\[[^\]]*\])? G(\d+)")

def parse_game(path):
    decisions = []
    result = None
    states = []  # (line_no, lib, gy) for PlayerRL1 from STATE/TURN lines
    spy_cast_lines = []
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
                    spy_cast_lines.append(ln)
    return decisions, result, states, spy_cast_lines

def main():
    rows = []
    for dd in DIRS:
        gdir = os.path.join(BASE, dd, SUB)
        if not os.path.isdir(gdir):
            continue
        for path in sorted(glob.glob(os.path.join(gdir, "*.txt"))):
            decisions, result, states, spy_lines = parse_game(path)
            # Spy cast detection from decision JSON
            spy_decs = [(ln, d) for (ln, d) in decisions
                        if "Balustrade Spy" in (d.get("selected_text") or "")
                        and "cast" in (d.get("selected_text") or "").lower()]
            cast_spy = len(spy_decs) > 0 or len(spy_lines) > 0
            if not cast_spy:
                continue
            first_cast_ln = spy_lines[0] if spy_lines else spy_decs[0][0]
            # library at cast (from decision JSON if available): lands remaining
            lands_at_cast = None
            lib_at_cast = None
            for ln, d in spy_decs:
                lib = d.get("library", [])
                lands_at_cast = sum(1 for c in lib if c in LANDS)
                lib_at_cast = d.get("library_size", len(lib))
                break
            # min library AFTER cast from STATE lines (real, post-resolution)
            after_states = [(ln, lib, gy) for (ln, lib, gy) in states if ln > first_cast_ln]
            min_lib_after = min((lib for (ln, lib, gy) in after_states), default=None)
            max_gy_after = max((gy for (ln, lib, gy) in after_states), default=None)
            rows.append({
                "dir": dd, "file": os.path.basename(path),
                "n_casts": len(spy_decs) if spy_decs else len(spy_lines),
                "lands_at_cast": lands_at_cast, "lib_at_cast": lib_at_cast,
                "min_lib_after": min_lib_after, "max_gy_after": max_gy_after,
                "result": result,
            })
    # Classify using real post-cast library/graveyard:
    # milled out: min_lib_after <= 3  (library emptied)
    # FIZZLE: min_lib_after stays high (e.g. >= 10) -> mill stopped early
    milled = [r for r in rows if r["min_lib_after"] is not None and r["min_lib_after"] <= 3]
    fizzle = [r for r in rows if r["min_lib_after"] is not None and r["min_lib_after"] >= 10]
    unknown = [r for r in rows if r["min_lib_after"] is None]
    print(f"games with spy cast: {len(rows)} | milled(libAfter<=3): {len(milled)} | "
          f"fizzle(libAfter>=10): {len(fizzle)} | no-after-state: {len(unknown)}")
    print(f"\nmilled W/L: {sum(1 for r in milled if r['result']=='WIN')}/{len(milled)}")
    print(f"fizzle W/L: {sum(1 for r in fizzle if r['result']=='WIN')}/{len(fizzle)}")
    print("\n=== FIZZLE GAMES (mill stopped early; lib stayed high) ===")
    print(f"{'file':<46} {'casts':>5} {'landsAtCast':>11} {'libAtCast':>9} {'minLibAfter':>11} {'maxGyAfter':>10} {'res':>5}")
    for r in sorted(fizzle, key=lambda x: -(x['min_lib_after'] or 0)):
        print(f"{r['dir'][:9]+'/'+r['file'][14:28]:<46} {r['n_casts']:>5} {str(r['lands_at_cast']):>11} "
              f"{str(r['lib_at_cast']):>9} {str(r['min_lib_after']):>11} {str(r['max_gy_after']):>10} {r['result']:>5}")

if __name__ == "__main__":
    main()
