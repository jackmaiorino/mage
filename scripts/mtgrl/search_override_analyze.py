"""Analyze SEARCH_OP override firings vs game outcomes in an eval run.

For each game log: read RESULT (WIN/LOSS) + every [SEARCH_OP_ARB] firing
(creatures, override flag, per-candidate terminal winrates). Classifies each
firing as SATURATED (all candidate winrates within `flat` of each other -> the
rollout can't discriminate -> any override is arbitrary noise) vs SPREAD (a real
gap the search can act on). For SPREAD firings, flags a "Spy-steer" = the search
moved away from a clearly-dominated Cast Balustrade Spy. Then asks the only
question that matters: do games where the search made a real Spy-steer actually
WIN more than games where it didn't?

Usage: python search_override_analyze.py <run_game_logs_dir> [flat=0.30]
"""
import os
import re
import sys

ARB = re.compile(r"\[SEARCH_OP_ARB\] creatures=(\d+) gy=(\d+) lib=(\d+) override=([01]) \| (.*)")
PAIR = re.compile(r"(.+?)=(-?\d+\.\d+)\s*;")


def parse_line(rest):
    return [(m.group(1).strip(), float(m.group(2))) for m in PAIR.finditer(rest)]


def main():
    d = sys.argv[1]
    flat = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".txt")]
    n_win = 0
    fired_games, steer_games = set(), set()
    tot_fire = tot_override = tot_sat = tot_spread = tot_spy_steer = 0
    per_game_outcome = {}
    for fp in files:
        with open(fp, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        win = "RESULT: WIN" in txt
        per_game_outcome[fp] = win
        if win:
            n_win += 1
        for m in ARB.finditer(txt):
            cre = int(m.group(1)); override = m.group(4) == "1"
            pairs = parse_line(m.group(5))
            tot_fire += 1
            fired_games.add(fp)
            if override:
                tot_override += 1
            wrs = [v for _, v in pairs if v >= -1.0]
            if len(wrs) < 2:
                continue
            spread = max(wrs) - min(wrs)
            if spread < flat:
                tot_sat += 1
            else:
                tot_spread += 1
                # Spy-steer: a Cast Balustrade Spy candidate is clearly below the best
                spy = [v for t, v in pairs if "Balustrade Spy" in t]
                if override and spy and min(spy) <= max(wrs) - flat:
                    tot_spy_steer += 1
                    steer_games.add(fp)

    n = len(files)
    print(f"games={n} winrate={n_win/n:.3f} ({n_win}/{n})")
    print(f"firings={tot_fire} overrides={tot_override} "
          f"saturated(spread<{flat})={tot_sat} spread>={flat}={tot_spread} "
          f"spy_steer_overrides={tot_spy_steer}")
    print(f"override rate among firings = {tot_override/max(tot_fire,1):.2f}; "
          f"of overrides, saturated(noise) approx = {tot_sat}/{tot_fire} firings flat")

    def wr(group):
        if not group:
            return float("nan"), 0
        w = sum(1 for g in group if per_game_outcome[g])
        return w / len(group), len(group)

    steer = [g for g in files if g in steer_games]
    nosteer_fired = [g for g in files if g in fired_games and g not in steer_games]
    never = [g for g in files if g not in fired_games]
    for name, grp in [("games w/ real Spy-steer", steer),
                      ("games fired but no Spy-steer", nosteer_fired),
                      ("games search never fired", never)]:
        w, k = wr(grp)
        print(f"  {name:32s}: winrate={w:.3f} (n={k})")


if __name__ == "__main__":
    main()
