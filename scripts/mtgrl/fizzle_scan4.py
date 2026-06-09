import json, os, glob, re

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
                try: decisions.append((ln, json.loads(s[len("REPLAY_DECISION_JSON:"):].strip())))
                except Exception: pass
            elif s.startswith("RESULT:"):
                result = s.split("RESULT:")[1].strip()
            else:
                m = RL_STATE.search(line)
                if m: states.append((ln, int(m.group(1)), int(m.group(2))))
                if "SELECTED: Cast Balustrade Spy" in s: spy_lines.append(ln)
    return decisions, result, states, spy_lines

def main():
    rows = []
    for dd in DIRS:
        gdir = os.path.join(BASE, dd, SUB)
        if not os.path.isdir(gdir): continue
        for path in sorted(glob.glob(os.path.join(gdir, "*.txt"))):
            decisions, result, states, spy_lines = parse_game(path)
            spy_idxs = [i for i,(ln,d) in enumerate(decisions)
                        if "Balustrade Spy" in (d.get("selected_text") or "")
                        and "cast" in (d.get("selected_text") or "").lower()]
            if not spy_idxs: continue
            di = spy_idxs[0]
            cast_ln, cast_d = decisions[di]
            lib = cast_d.get("library", [])
            lands_at_cast = sum(1 for c in lib if c in LANDS)
            lib_at_cast = cast_d.get("library_size", len(lib))
            turn = cast_d.get("turn")
            # find the SELECT_TARGETS decision shortly after the cast
            target = None
            for j in range(di+1, min(di+4, len(decisions))):
                _, dj = decisions[j]
                if dj.get("action_type") == "SELECT_TARGETS":
                    target = dj.get("selected_text"); break
            # gy before / max gy after (real)
            gy_before = None
            for (ln, l, g) in states:
                if ln < cast_ln: gy_before = g
            after = [(ln,l,g) for (ln,l,g) in states if ln > cast_ln]
            max_gy_after = max((g for (_,_,g) in after), default=None)
            mill_delta = (max_gy_after - gy_before) if (max_gy_after is not None and gy_before is not None) else None
            self_target = target is not None and ("PlayerRL1" in target or "you" in target.lower())
            milled_out = max_gy_after is not None and max_gy_after >= 44
            rows.append(dict(dir=dd, file=os.path.basename(path), turn=turn,
                             lands_at_cast=lands_at_cast, lib_at_cast=lib_at_cast,
                             target=target, self_target=self_target,
                             gy_before=gy_before, max_gy_after=max_gy_after,
                             mill_delta=mill_delta, milled_out=milled_out, result=result))
    fizzle = [r for r in rows if not r["milled_out"]]
    print(f"total spy-cast games: {len(rows)} | milled_out: {sum(r['milled_out'] for r in rows)} | fizzle: {len(fizzle)}")
    # split fizzle by target
    opp_fizzle = [r for r in fizzle if r["self_target"] is False and r["target"] and "EvalBot" in (r["target"] or "")]
    self_fizzle = [r for r in fizzle if r["self_target"]]
    other = [r for r in fizzle if r not in opp_fizzle and r not in self_fizzle]
    print(f"  fizzle targeting OPPONENT (wrong target): {len(opp_fizzle)}")
    print(f"  fizzle targeting SELF (sequencing/land left): {len(self_fizzle)}")
    print(f"  fizzle unknown-target: {len(other)}")
    print("\n--- SELF-target fizzles (real combo attempts that fizzled) ---")
    print(f"{'file':<26} {'turn':>4} {'landsLib':>8} {'libSz':>6} {'gyBef':>6} {'maxGyAft':>8} {'millD':>6} {'res':>5}")
    for r in sorted(self_fizzle, key=lambda x:(x['mill_delta'] or 0)):
        print(f"{r['dir'][9:]+'/'+r['file'][14:23]:<26} {str(r['turn']):>4} {r['lands_at_cast']:>8} {str(r['lib_at_cast']):>6} "
              f"{str(r['gy_before']):>6} {str(r['max_gy_after']):>8} {str(r['mill_delta']):>6} {r['result']:>5}")
    print("\n--- OPPONENT-target fizzles (targeting error) ---")
    for r in sorted(opp_fizzle, key=lambda x:(x['mill_delta'] or 0)):
        print(f"{r['dir'][9:]+'/'+r['file'][14:23]:<26} t{r['turn']} landsLib={r['lands_at_cast']} target={r['target']} res={r['result']}")

if __name__ == "__main__":
    main()
