import json, os, glob, sys

LANDS = {"Forest", "Swamp"}
BASE = "C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
DIRS = ["baseline_auc_ab", "baseline_auc_s9999", "baseline_auc_big5151"]
SUB = "game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity"

def parse_game(path):
    decisions = []
    result = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("REPLAY_DECISION_JSON:"):
                try:
                    d = json.loads(line[len("REPLAY_DECISION_JSON:"):].strip())
                    decisions.append(d)
                except Exception:
                    pass
            elif line.startswith("RESULT:"):
                result = line.split("RESULT:")[1].strip()
    return decisions, result

def main():
    rows = []
    for dd in DIRS:
        gdir = os.path.join(BASE, dd, SUB)
        for path in sorted(glob.glob(os.path.join(gdir, "*.txt"))):
            decisions, result = parse_game(path)
            # find Spy casts
            spy_cast_idxs = [i for i, d in enumerate(decisions)
                             if "Balustrade Spy" in (d.get("selected_text") or "")
                             and "cast" in (d.get("selected_text") or "").lower()]
            if not spy_cast_idxs:
                continue
            # min library_size at any decision AT OR AFTER first spy cast
            first = spy_cast_idxs[0]
            # library size before cast (at cast decision) and lands in library at cast
            cast_dec = decisions[first]
            lib_at_cast = cast_dec.get("library", [])
            lands_in_lib_at_cast = sum(1 for c in lib_at_cast if c in LANDS)
            lib_size_at_cast = cast_dec.get("library_size", len(lib_at_cast))
            # min library_size after the cast (mill happens after)
            min_lib_after = min((d.get("library_size", 9999) for d in decisions[first:]), default=9999)
            # also min over whole game
            min_lib_all = min((d.get("library_size", 9999) for d in decisions), default=9999)
            rows.append({
                "dir": dd,
                "file": os.path.basename(path),
                "n_spy_casts": len(spy_cast_idxs),
                "lib_size_at_cast": lib_size_at_cast,
                "lands_in_lib_at_cast": lands_in_lib_at_cast,
                "min_lib_after_cast": min_lib_after,
                "min_lib_all": min_lib_all,
                "result": result,
                "n_decisions": len(decisions),
            })
    # classify
    milled = [r for r in rows if r["min_lib_after_cast"] <= 2]
    fizzle = [r for r in rows if r["min_lib_after_cast"] > 5]  # never got near empty
    print("total games with spy cast:", len(rows))
    print("milled out (min_lib_after<=2):", len(milled))
    print("fizzle (min_lib_after>5):", len(fizzle))
    print()
    print("FIZZLE GAMES (lands still in lib, never emptied):")
    print(f"{'file':<40} {'casts':>5} {'libAtCast':>9} {'landsAtCast':>11} {'minLibAfter':>11} {'result':>6}")
    for r in sorted(fizzle, key=lambda x: -x["min_lib_after_cast"]):
        print(f"{r['dir'][:12]+'/'+r['file'][:22]:<40} {r['n_spy_casts']:>5} {r['lib_size_at_cast']:>9} {r['lands_in_lib_at_cast']:>11} {r['min_lib_after_cast']:>11} {r['result']:>6}")

if __name__ == "__main__":
    main()
