#!/usr/bin/env python3
"""Step-2 smoke, Part 1 (Codex #16): does the agent REACH a generic combo/finish-ready
state often enough to bother building a finish-teacher? If reach-rate is tiny, the SETUP
is the bottleneck and finish-search is moot (the observation-lever trap).

GENERIC gate (NO card names, thesis-clean): a big self-mill happened (graveyard large =>
library milled => ~0 library lands) AND a board exists to sacrifice for Dread Return.
Readable from STATE-line COUNTS (exact; immune to name truncation):
  PlayerRL1 L# H#[..] B<board># G<gy>#  -> board_count, gy_count.
finish-ready(thr) := max over states of (gy_count >= thr AND board_count >= 3).

Also reports, among finish-ready games, win-rate and combo-execution (Lotleth Giant
reanimated to battlefield) -- to see if reaching ready -> winning/comboing.

Usage: python combo_reach_rate.py DIR [DIR ...]
"""
import glob, os, re, sys

STATE_RE = re.compile(r"PlayerRL1 L-?\d+ H\d+(?:\[[^]]*\])? B(\d+)(?:\[[^]]*\])? G(\d+)")


def parse_game(path):
    result = None; max_gy = 0; board_at_maxgy = 0
    ready = {15: False, 20: False, 25: False, 30: False}
    lotleth_bf = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("RESULT:"):
                r = s.split(":", 1)[1].strip().upper()
                result = "win" if r.startswith("WIN") else ("loss" if r.startswith("LOSS") else None)
            elif "STATE:" in s:
                m = STATE_RE.search(s)
                if m:
                    b = int(m.group(1)); g = int(m.group(2))
                    if g > max_gy:
                        max_gy = g; board_at_maxgy = b
                    for thr in ready:
                        if g >= thr and b >= 3:
                            ready[thr] = True
                # combo signal: Lotleth Giant on agent battlefield
                if re.search(r"B\d+\[[^]]*Lotleth Giant", s):
                    lotleth_bf = True
    if result is None:
        return None
    return dict(result=result, max_gy=max_gy, board_at_maxgy=board_at_maxgy, ready=ready, combo=lotleth_bf)


def main():
    dirs = sys.argv[1:]
    logs = []
    for d in dirs:
        logs += glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)
    games = [g for g in (parse_game(p) for p in logs) if g]
    n = len(games)
    if not n:
        print("no games"); return
    import statistics as st
    print(f"games={n}  overall WR={sum(1 for g in games if g['result']=='win')/n:.3f}")
    print(f"max graveyard reached: median={st.median([g['max_gy'] for g in games])}  "
          f"p90={sorted(g['max_gy'] for g in games)[int(0.9*n)-1]}  max={max(g['max_gy'] for g in games)}")
    print("\n=== FINISH-READY reach rate (gy>=thr AND board>=3) + outcomes when reached ===")
    for thr in (15, 20, 25, 30):
        rg = [g for g in games if g['ready'][thr]]
        if rg:
            wr = sum(1 for g in rg if g['result'] == 'win') / len(rg)
            cb = sum(1 for g in rg if g['combo']) / len(rg)
            print(f"  gy>={thr}: reached in {len(rg)}/{n} ({len(rg)/n:.0%})  | WR|reached={wr:.2f}  combo|reached={cb:.2f}")
        else:
            print(f"  gy>={thr}: reached in 0/{n} (0%)")
    print("\nGATE (Codex #16): need >=2% of games finish-ready (>=5% strong) to justify building the finish-teacher.")
    print("  If reach-rate is high but combo|reached is low -> FINISH execution is the gap (build finish-search).")
    print("  If reach-rate is LOW -> SETUP is the bottleneck (finish-search is moot; reaching combo-ready is the problem).")


if __name__ == "__main__":
    main()
