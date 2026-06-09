import json, glob, os
base = r"C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
runs=["baseline_auc_ab","baseline_auc_s9999","baseline_auc_big5151"]
BLUE=["Saruli Caretaker","Lotus Petal"]

def parse(fp):
    ds=[]; res=None
    with open(fp,encoding="utf-8",errors="replace") as f:
        for line in f:
            line=line.strip()
            if line.startswith("REPLAY_DECISION_JSON:"):
                try: ds.append(json.loads(line[len("REPLAY_DECISION_JSON:"):].strip()))
                except: pass
            elif line.startswith("RESULT:"): res=line.split(":",1)[1].strip()
    return ds,res

cast_mulls=[]; nocast_mulls=[]
cast_finalsize=[]; nocast_finalsize=[]
cast_finalblue=0; cast_n=0; nocast_finalblue=0; nocast_n=0
for run in runs:
    for fp in glob.glob(os.path.join(base,run,"game_logs","*","*.txt")):
        ds,res=parse(fp)
        if not ds: continue
        cast=any("Cast Balustrade Spy" in (d.get("selected_text") or "") for d in ds)
        mulls=max((d.get("mulligans_taken") or 0) for d in ds)
        keep=[d for d in ds if d.get("action_type")=="MULLIGAN" and d.get("selected_text")=="KEEP"]
        fh=(keep[-1].get("hand") if keep else ds[0].get("hand")) or []
        hasblue=any(b in fh for b in BLUE)
        if cast:
            cast_mulls.append(mulls); cast_finalsize.append(len(fh)); cast_n+=1; cast_finalblue+=int(hasblue)
        else:
            nocast_mulls.append(mulls); nocast_finalsize.append(len(fh)); nocast_n+=1; nocast_finalblue+=int(hasblue)

def stats(l): return f"mean={sum(l)/len(l):.2f} n={len(l)}"
print("CAST-SPY games:    avg mulligans",stats(cast_mulls),"| avg opening size",stats(cast_finalsize),f"| opening had blue src: {cast_finalblue}/{cast_n} ({100*cast_finalblue/cast_n:.0f}%)")
print("NO-CAST games:     avg mulligans",stats(nocast_mulls),"| avg opening size",stats(nocast_finalsize),f"| opening had blue src: {nocast_finalblue}/{nocast_n} ({100*nocast_finalblue/nocast_n:.0f}%)")
import statistics
print(f"\nmull distribution CAST: {sorted(cast_mulls)}")
print(f"mull distribution NOCAST: {sorted(nocast_mulls)}")
