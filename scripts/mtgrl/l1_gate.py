#!/usr/bin/env python3
"""L1 long-clean-run gate (Codex #38). Reads the per-chunk uniform-gauntlet evals and
decides CONTINUE / STOP-SUCCESS / STOP-FAIL vs ref C (Rally 20, uniform 47).

Rules:
  SUCCESS (stop, keep): any chunk hits Rally >= 28 AND uniform mean >= 47 (flat/up).
  FAIL after >=3 chunks (30k): Rally < 17 (refC-3) OR uniform < 45 (refC-2).
  FAIL after >=5 chunks (50k): no positive Rally slope (ck5 Rally <= ck1 Rally) AND uniform <= 47.
  else CONTINUE.

Usage: python l1_gate.py <eval_results_glob> <current_chunk>
  prints one of: CONTINUE | STOP-SUCCESS | STOP-FAIL:<reason>
"""
import re, glob, sys, collections

REFC_RALLY, REFC_UNIFORM = 20.0, 47.0

def parse_eval(path):
    raw = open(path, "rb").read().decode("utf-16", errors="replace")
    agg = collections.defaultdict(lambda: [0, 0])
    for ln in raw.splitlines():
        m = re.search(r"vs Deck - (.+?) chunk=\d+/\d+:\s+(\d+)/(\d+)", ln)
        if m:
            agg[m.group(1)][0] += int(m.group(2)); agg[m.group(1)][1] += int(m.group(3))
    if not agg:
        return None
    tot = [sum(v[0] for v in agg.values()), sum(v[1] for v in agg.values())]
    rally = next((100.0*v[0]/v[1] for k, v in agg.items() if "Rally" in k and v[1]), None)
    uni = 100.0*tot[0]/tot[1] if tot[1] else None
    return rally, uni

def main():
    pat, cur = sys.argv[1], int(sys.argv[2])
    chunks = {}
    for f in glob.glob(pat):
        m = re.search(r"_ck(\d+)_", f)
        if not m: continue
        r = parse_eval(f)
        if r and r[0] is not None and r[1] is not None:
            chunks[int(m.group(1))] = r
    if not chunks:
        print("CONTINUE"); return
    ks = sorted(chunks)
    traj = "  ".join(f"ck{k}:Rally{chunks[k][0]:.0f}/uni{chunks[k][1]:.1f}" for k in ks)
    sys.stderr.write(f"L1 trajectory (refC Rally{REFC_RALLY:.0f}/uni{REFC_UNIFORM:.0f}): {traj}\n")
    # SUCCESS
    for k in ks:
        rally, uni = chunks[k]
        if rally >= 28.0 and uni >= REFC_UNIFORM:
            print(f"STOP-SUCCESS:ck{k} Rally{rally:.0f} uni{uni:.1f}"); return
    last_rally, last_uni = chunks[ks[-1]]
    # FAIL gate at 30k+
    if cur >= 3:
        if last_rally < REFC_RALLY - 3 or last_uni < REFC_UNIFORM - 2:
            print(f"STOP-FAIL:30k-gate Rally{last_rally:.0f}(<{REFC_RALLY-3:.0f}) or uni{last_uni:.1f}(<{REFC_UNIFORM-2:.0f})"); return
    # FAIL gate at 50k+
    if cur >= 5:
        r1 = chunks[ks[0]][0]
        if last_rally <= r1 and last_uni <= REFC_UNIFORM:
            print(f"STOP-FAIL:50k-no-slope Rally{r1:.0f}->{last_rally:.0f} uni{last_uni:.1f}"); return
    print("CONTINUE")

if __name__ == "__main__":
    main()
