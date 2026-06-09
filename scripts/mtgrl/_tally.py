import json, glob, os
LANDS={"Forest","Swamp","Island","Mountain","Plains"}
BASE="C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
M="game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity"
RUNS=["baseline_auc_ab","baseline_auc_s9999","baseline_auc_big5151"]
def load(path):
    decs=[]; result=None
    for line in open(path,encoding="utf-8",errors="replace"):
        line=line.strip()
        if line.startswith("RESULT:"): result=line.split("RESULT:",1)[1].strip()
        elif line.startswith("REPLAY_DECISION_JSON:"):
            try: decs.append(json.loads(line.split("REPLAY_DECISION_JSON:",1)[1].strip()))
            except: pass
    return decs,result
single=0; multi=0; firstspy_turns=[]; agentturns=[]
for run in RUNS:
    for f in sorted(glob.glob(os.path.join(BASE,run,M,"*.txt"))):
        decs,result=load(f)
        if result!="WIN": continue
        spy_casts=[(i,d) for i,d in enumerate(decs) if "Cast Balustrade Spy" in (d.get("selected_text") or "")]
        if not spy_casts: continue
        # milled out?
        milled=any((d.get("library_size") or 99)<=1 for d in decs)
        if not milled: continue
        n=len(spy_casts)
        ft=spy_casts[0][1].get("turn")
        firstspy_turns.append(ft)
        if n==1: single+=1
        else: multi+=1
import statistics
print("combo wins single-Spy:",single," multi-Spy(fizzle-recovery):",multi)
print("first-Spy turn distribution (engine turns, agent turn = (t+1)/2):")
from collections import Counter
print(sorted(Counter(firstspy_turns).items()))
print("median first-spy engine turn:", statistics.median(firstspy_turns), " => agent turn ~", (statistics.median(firstspy_turns)+1)/2)
