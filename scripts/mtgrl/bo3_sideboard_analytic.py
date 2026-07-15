"""Analytic best-of-3 match-level pilot-adaptation (Codex #51).
Game 1 = 7k generalist (identify archetype). Games 2-3 = whole-game routed model.
BO3 match WR (>=2 of 3) with game1~Bern(p=g1), games2,3~Bern(q=g23): q^2 + 2*p*q*(1-q).
all-7k BO3 (p=q=g7k): g7k^2*(3 - 2*g7k).
Compares: all-7k BO3 baseline | detected-Rally-routed (q=15k for Rally else 7k) | oracle-table (q=best per matchup).
"""
import re, sys

def parse(p):
    try: t=open(p,encoding='utf-16').read()
    except Exception: t=open(p,encoding='utf-8',errors='ignore').read()
    a={}
    for ln in t.splitlines():
        m=re.search(r'vs Deck - ([^c]+?)\s+chunk=\d+/\d+:\s+(\d+)/(\d+)\s+wr=',ln)
        if m: o=m.group(1).strip(); a.setdefault(o,[0,0]); a[o][0]+=int(m.group(2)); a[o][1]+=int(m.group(3))
    return {o:(v[0]/v[1] if v[1] else 0.0) for o,v in a.items()}

g7 = parse("local-training/affinity_7k_matched_RESULT.log")
s15 = parse("local-training/affinity_15k_matched_RESULT.log")

def bo3(p, q): return q*q + 2*p*q*(1-q)          # game1~p, games2,3~q
def bo3_same(g): return g*g*(3 - 2*g)            # all three ~g

RALLY = 'Mono Red Rally'
matchups = sorted(set(g7) | set(s15))
print(f"{'matchup':22s} {'7k(g)':>6s} {'15k':>6s} | {'all7k-BO3':>9s} {'Rally-routed':>12s} {'oracle-tbl':>10s}")
sum_base=sum_rr=sum_or=0.0; n=0
for o in matchups:
    p = g7.get(o, 0.0)            # game 1 generalist single-game wr
    g15 = s15.get(o, 0.0)
    base = bo3_same(p)                                   # all-7k BO3
    q_rr = g15 if (o == RALLY) else p                    # detected-Rally routing (conservative)
    rr = bo3(p, q_rr)
    q_or = max(p, g15)                                   # oracle-table: best per matchup
    orc = bo3(p, q_or)
    sum_base+=base; sum_rr+=rr; sum_or+=orc; n+=1
    star=' <<<' if o==RALLY else ''
    print(f"{o:22s} {p:5.1%} {g15:5.1%} | {base:8.1%} {rr:11.1%} {orc:9.1%}{star}")
print(f"{'UNIFORM (match WR)':22s} {'':6s} {'':6s} | {sum_base/n:8.1%} {sum_rr/n:11.1%} {sum_or/n:9.1%}")
rb=bo3_same(g7.get(RALLY,0)); rr=bo3(g7.get(RALLY,0), s15.get(RALLY,0))
print(f"\nRally match WR: all-7k {rb:.1%} -> detected-routed {rr:.1%} (d{(rr-rb)*100:+.1f}pp)")
du=(sum_rr-sum_base)/n*100
print(f"Uniform match WR: all-7k {sum_base/n:.1%} -> routed {sum_rr/n:.1%} (d{du:+.1f}pp)")
print(f"\nGATE (Codex #51): Rally match WR +>=10pp ({(rr-rb)*100:+.1f}) AND uniform +>=1.5pp ({du:+.1f}) AND no non-Rally regression")
ok = (rr-rb)>=0.10 and du>=1.5
print("ANALYTIC VERDICT:", "CLEARS -> build measured BO3 harness" if ok else "below gate")
