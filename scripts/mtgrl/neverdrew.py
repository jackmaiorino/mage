import json, os
base = r"C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
MD="game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity"
never_drew=[
('baseline_auc_ab','game_20260530_143147_0009.txt'),('baseline_auc_ab','game_20260530_143350_0022.txt'),
('baseline_auc_ab','game_20260530_143442_0025.txt'),('baseline_auc_ab','game_20260530_143549_0028.txt'),
('baseline_auc_s9999','game_20260530_145233_0031.txt'),('baseline_auc_s9999','game_20260530_145237_0032.txt'),
('baseline_auc_big5151','game_20260530_154559_0081.txt'),('baseline_auc_big5151','game_20260530_154741_0092.txt'),
('baseline_auc_big5151','game_20260530_154950_0107.txt'),('baseline_auc_big5151','game_20260530_155104_0115.txt'),
]
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
DIG=["Lead the Stampede","Winding Way"]
for run,fn in never_drew:
    ds,res=parse(os.path.join(base,run,MD,fn))
    mulls=max((d.get("mulligans_taken") or 0) for d in ds)
    digcast=sum(1 for d in ds if any((d.get("selected_text") or "")==("Cast "+x) for x in DIG))
    # how many dig spells in library at start (unused dig potential)
    lastturn=max((d.get("turn") or 0) for d in ds)
    # any dig spell ever castable but skipped?
    dig_skipped=0
    for d in ds:
        if d.get("action_type")=="ACTIVATE_ABILITY_OR_SPELL":
            cands=d.get("candidate_texts") or []
            sel=d.get("selected_text") or ""
            if any(("Cast "+x) in cands for x in DIG) and not any(("Cast "+x)==sel for x in DIG):
                dig_skipped+=1
    print(f"{fn[-8:-4]} {res} mulls={mulls} lastT={lastturn} digCast={digcast} digCastableButSkipped={dig_skipped}")
