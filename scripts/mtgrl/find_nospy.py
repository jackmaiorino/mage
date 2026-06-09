import json, glob, os, sys

base = r"C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
runs = ["baseline_auc_ab","baseline_auc_s9999","baseline_auc_big5151"]

def parse(fp):
    decisions=[]
    result=None
    with open(fp,encoding="utf-8",errors="replace") as f:
        for line in f:
            line=line.strip()
            if line.startswith("REPLAY_DECISION_JSON:"):
                try:
                    decisions.append(json.loads(line[len("REPLAY_DECISION_JSON:"):].strip()))
                except: pass
            elif line.startswith("RESULT:"):
                result=line.split(":",1)[1].strip()
    return decisions,result

total=0
nospy=[]
for run in runs:
    for fp in glob.glob(os.path.join(base,run,"game_logs","*","*.txt")):
        decisions,result=parse(fp)
        total+=1
        cast_spy=any("Cast Balustrade Spy" in (d.get("selected_text") or "") for d in decisions)
        if not cast_spy:
            # also check chosen_texts variants
            cast2=any("Balustrade Spy" in (d.get("selected_text") or "") and "Cast" in (d.get("selected_text") or "") for d in decisions)
            if not cast2:
                nospy.append((run,os.path.basename(fp),result,len(decisions)))

print(f"TOTAL games: {total}")
print(f"NEVER cast Spy: {len(nospy)}")
for r in nospy:
    print(r)
