"""Characterize games where SEARCH_OP never fired: are they reach-starved?

For each game log, classify fired vs never-fired (any [SEARCH_OP_ARB] line).
For never-fired games, measure the upstream-reach proxies: mulligans taken,
whether the agent ever cast Balustrade Spy, ever reached the Dread Return combo,
and final self life. If never-fired losses are dominated by high-mulligan /
never-cast-Spy games, the bottleneck is reachability (mulligan/mana), not the
in-game cast choice -- so no eval-time search can rescue them.

Usage: python never_fired_characterize.py <run_game_logs_dir>
"""
import os
import re
import sys

MULL = re.compile(r'"mulligans_taken":(\d+)')


def main():
    d = sys.argv[1]
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".txt")]
    cats = {"fired": [], "never": []}
    for fp in files:
        with open(fp, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        fired = "[SEARCH_OP_ARB]" in txt
        win = "RESULT: WIN" in txt
        cast_spy = "Cast Balustrade Spy" in txt and "chosen_texts" in txt
        # cast actually chosen (not just available): look for chosen Spy in decision json
        cast_spy_chosen = '"chosen_texts":["Cast Balustrade Spy' in txt or \
            bool(re.search(r'"selected_text":"Cast Balustrade Spy', txt))
        combo = "Lotleth Giant" in txt and ("Dread Return" in txt)
        mulls = [int(x) for x in MULL.findall(txt)]
        mull = max(mulls) if mulls else -1
        cats["fired" if fired else "never"].append(
            (win, cast_spy_chosen, combo, mull))
    for k, rows in cats.items():
        n = len(rows)
        if not n:
            print(f"{k}: 0 games")
            continue
        w = sum(1 for r in rows if r[0])
        spy = sum(1 for r in rows if r[1])
        cmb = sum(1 for r in rows if r[2])
        mulls = [r[3] for r in rows if r[3] >= 0]
        avgm = sum(mulls) / len(mulls) if mulls else float("nan")
        print(f"{k:6s}: n={n} winrate={w/n:.3f} cast_Spy={spy}/{n} "
              f"reached_combo_cards={cmb}/{n} avg_mulligans={avgm:.2f}")
    # never-fired losers specifically
    losers = [r for r in cats["never"] if not r[0]]
    if losers:
        spy = sum(1 for r in losers if r[1])
        mulls = [r[3] for r in losers if r[3] >= 0]
        avgm = sum(mulls) / len(mulls) if mulls else float("nan")
        print(f"\nNEVER-FIRED LOSSES (n={len(losers)}): cast_Spy={spy}/{len(losers)} "
              f"avg_mulligans={avgm:.2f}")
        print(" => if cast_Spy is low and mulligans high, these are reach-starved "
              "(mulligan/mana bottleneck), unreachable by in-game search.")


if __name__ == "__main__":
    main()
