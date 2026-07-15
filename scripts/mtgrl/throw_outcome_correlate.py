#!/usr/bin/env python3
"""Correlate fanout-detected THROWS with actual game outcomes.

Decisive question: do the agent's throws (picking a candidate with lower terminal
win-rate than the search-best) live in games it LOST? If yes, fixing them converts
losses -> winrate lifts. If they're in games it won anyway -> winrate-irrelevant.

Per game log: parse GAME OUTCOME (Winner) + every [SEARCH_OP] line (indexed wr,
best/orig). A 'throw' = best_wr - orig_wr >= gap. Tabulate throws vs W/L, and the
key metric: among LOST games, how many had a winnable pivotal decision (a candidate
with high wr) that the agent passed up = a RECOVERABLE loss.

Usage: python throw_outcome_correlate.py <game_log_dir> [gap=0.20] [winnable=0.50]
"""
import re, glob, sys, os

def main():
    d = sys.argv[1]
    gap = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    winnable = float(sys.argv[3]) if len(sys.argv) > 3 else 0.50
    files = glob.glob(os.path.join(d, "*.txt"))

    games = 0; won = 0; lost = 0; no_outcome = 0
    throws_in_won = 0; throws_in_lost = 0
    lost_with_winnable_throw = 0
    won_games_searched = 0; lost_games_searched = 0
    recoverable = []  # (file, best_wr, orig_wr, gap)
    line_re = re.compile(r"\[SEARCH_OP\].*?best=(\d+) orig=(\d+).*?wr=\[([^\]]*)\]")

    for f in files:
        txt = open(f, encoding="utf-8", errors="replace").read()
        m = re.search(r"GAME OUTCOME\s*=*\s*Winner:\s*(\S+)", txt)
        if not m:
            no_outcome += 1; continue
        games += 1
        agent_won = (m.group(1) == "PlayerRL1")
        if agent_won: won += 1
        else: lost += 1

        # collect contested throws in this game
        throws = []  # (best_wr, orig_wr)
        best_wr_avail = 0.0
        searched = False
        for ln in txt.splitlines():
            mm = line_re.search(ln)
            if not mm: continue
            bi, oi = int(mm.group(1)), int(mm.group(2))
            raw = [x.strip() for x in mm.group(3).split(",")]
            wr = [None if x == "NaN" else float(x) for x in raw]
            if bi >= len(wr) or oi >= len(wr) or wr[bi] is None or wr[oi] is None:
                continue
            searched = True
            best_wr_avail = max(best_wr_avail, wr[bi])
            g = wr[bi] - wr[oi]
            if g >= gap:
                throws.append((wr[bi], wr[oi], g))
        if searched:
            if agent_won: won_games_searched += 1
            else: lost_games_searched += 1
        for (bw, ow, g) in throws:
            if agent_won: throws_in_won += 1
            else: throws_in_lost += 1
        if not agent_won and any(bw >= winnable for (bw, ow, g) in throws):
            lost_with_winnable_throw += 1
            bw = max(bw for (bw, ow, g) in throws if bw >= winnable)
            ow = min(ow for (bw, ow, g) in throws)
            recoverable.append((os.path.basename(f), bw, ow))

    print(f"games_with_outcome={games}  (won={won} lost={lost})  files_no_outcome={no_outcome}")
    print(f"games_with_>=1_search: won={won_games_searched} lost={lost_games_searched}")
    print(f"THROWS (gap>={gap}): in_won_games={throws_in_won}  in_lost_games={throws_in_lost}")
    print(f"--- DECISIVE: among LOST games that were searched, how many had a WINNABLE (best_wr>={winnable}) thrown decision? ---")
    if lost_games_searched:
        pct = 100.0 * lost_with_winnable_throw / lost_games_searched
        print(f"RECOVERABLE LOSSES = {lost_with_winnable_throw}/{lost_games_searched} searched-lost games ({pct:.0f}%)")
        print(f"  (these are losses where a clearly-better line existed at a pivotal decision but the agent passed it up)")
    else:
        print("no searched lost games")
    if recoverable:
        print("--- recoverable-loss details (best_wr available vs orig_wr chosen) ---")
        for (fn, bw, ow) in recoverable[:20]:
            print(f"  {fn}: had a {bw:.0%} line, agent's pick ~{ow:.0%}")

if __name__ == "__main__":
    main()
