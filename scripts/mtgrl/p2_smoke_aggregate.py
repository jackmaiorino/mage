#!/usr/bin/env python3
"""Aggregate the Step-2 Part-2 search smoke (p2_baseline vs p2_search).

GATE (Codex #16): conditional search-win found/calls >= 40% (>=2% of games must also
REACH a combo-ready ACTIVATE root, i.e. calls>0 in a healthy share of games).
Also reports winrate + combo-execution (Lotleth Giant reanimated to battlefield) for both
passes so we see if autopilot-applying the search-found finish actually lifts results.

Usage: python p2_smoke_aggregate.py <sweeps_root>   (root holds p2_baseline/ and p2_search/)
"""
import glob, os, re, sys

CALLS_RE = re.compile(r"calls=(\d+) found=(\d+) overrides=(\d+) timeouts=(\d+)")
LOTLETH_BF_RE = re.compile(r"B\d+\[[^]]*Lotleth Giant")


def winrate(run_dir):
    wins = total = 0
    for csv in glob.glob(os.path.join(run_dir, "results", "*.csv")):
        try:
            parts = open(csv, encoding="utf-8", errors="replace").read().strip().split(",")
            if len(parts) >= 3:
                wins += int(parts[0]); total += int(parts[1])
        except Exception:
            pass
    return wins, total


def search_counters(run_dir):
    """Sum the FINAL (max) cumulative calls/found per job log."""
    calls = found = overrides = timeouts = 0
    n_logs_with_calls = 0
    for log in glob.glob(os.path.join(run_dir, "logs", "*.log")):
        mx = (0, 0, 0, 0)
        for line in open(log, encoding="utf-8", errors="replace"):
            m = CALLS_RE.search(line)
            if m:
                v = tuple(int(x) for x in m.groups())
                if v[0] >= mx[0]:
                    mx = v
        calls += mx[0]; found += mx[1]; overrides += mx[2]; timeouts += mx[3]
        if mx[0] > 0:
            n_logs_with_calls += 1
    return calls, found, overrides, timeouts, n_logs_with_calls


def combo_and_reach(run_dir):
    """Per game-log: did Lotleth reach the battlefield (combo executed); did [ONLINE_PREFIX] fire."""
    logs = glob.glob(os.path.join(run_dir, "game_logs", "**", "*.txt"), recursive=True)
    n = combo = reached = won = 0
    for p in logs:
        result = None; lotleth = False; fired = False
        for line in open(p, encoding="utf-8", errors="replace"):
            s = line.strip()
            if s.startswith("RESULT:"):
                r = s.split(":", 1)[1].strip().upper()
                result = "win" if r.startswith("WIN") else "loss"
            elif "STATE:" in s and LOTLETH_BF_RE.search(s):
                lotleth = True
            elif "[ONLINE_PREFIX]" in s:
                fired = True
        if result is None:
            continue
        n += 1
        if lotleth: combo += 1
        if fired: reached += 1
        if result == "win": won += 1
    return n, combo, reached, won


def report(root, run_id):
    rd = os.path.join(root, run_id)
    if not os.path.isdir(rd):
        print(f"[{run_id}] MISSING dir {rd}"); return None
    w, t = winrate(rd)
    calls, found, ov, to, nlc = search_counters(rd)
    n, combo, reached, won = combo_and_reach(rd)
    wr = w / t if t else 0.0
    cr = combo / n if n else 0.0
    print(f"\n[{run_id}]")
    print(f"  winrate          = {wr:.3f}  ({w}/{t})")
    print(f"  combo-exec rate  = {cr:.3f}  (Lotleth on bf in {combo}/{n} game-logs)")
    if calls or run_id.endswith("search"):
        ss = found / calls if calls else 0.0
        print(f"  search calls     = {calls}  (combo-ready ACTIVATE roots; in {nlc} job-logs)")
        print(f"  search found     = {found}  -> conditional search-win = {ss:.3f}")
        print(f"  overrides/timeout= {ov}/{to}")
        print(f"  games that fired = {reached}/{n} ({(reached/n if n else 0):.0%})  (>= ~2% gate)")
    return dict(wr=wr, w=w, t=t, cr=cr, calls=calls, found=found, n=n, reached=reached)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "local-training/local_pbt/cp7_eval_sweeps"
    base = report(root, "p2_baseline")
    srch = report(root, "p2_search")
    print("\n=== VERDICT (Codex #16) ===")
    if srch and srch["calls"]:
        ss = srch["found"] / srch["calls"]
        reach = srch["reached"] / srch["n"] if srch["n"] else 0.0
        gate_search = ss >= 0.40
        gate_reach = reach >= 0.02
        print(f"  conditional search-win = {ss:.3f}  (gate >=0.40 -> {'PASS' if gate_search else 'FAIL'})")
        print(f"  reach rate (games fired) = {reach:.3f}  (gate >=0.02 -> {'PASS' if gate_reach else 'FAIL'})")
        if base:
            print(f"  winrate  lift = {srch['wr']-base['wr']:+.3f}  ({base['wr']:.3f} -> {srch['wr']:.3f})")
            print(f"  combo-rt lift = {srch['cr']-base['cr']:+.3f}  ({base['cr']:.3f} -> {srch['cr']:.3f})")
        if gate_search and gate_reach:
            print("  => BUILD the finish-teacher (search reliably finds the finish from reached roots).")
        elif not gate_reach:
            print("  => SETUP is the bottleneck (rarely reach combo-ready); finish-search is moot.")
        else:
            print("  => Search too weak from generic ordering; try larger topK/nodes before declaring dead.")
    else:
        print("  no search calls recorded -- check that p2_search ran with the gate enabled.")


if __name__ == "__main__":
    main()
