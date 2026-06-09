import json, os
base = r"C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
MD="game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity"
BLUE=["Saruli Caretaker","Lotus Petal"]

# the 18 mana-screw + 10 never-drew files for full no-spy analysis
files=[
('baseline_auc_ab','game_20260530_143037_0003.txt'),('baseline_auc_ab','game_20260530_143134_0007.txt'),
('baseline_auc_ab','game_20260530_143147_0009.txt'),('baseline_auc_ab','game_20260530_143230_0011.txt'),
('baseline_auc_ab','game_20260530_143233_0012.txt'),('baseline_auc_ab','game_20260530_143350_0022.txt'),
('baseline_auc_ab','game_20260530_143428_0024.txt'),('baseline_auc_ab','game_20260530_143442_0025.txt'),
('baseline_auc_ab','game_20260530_143451_0026.txt'),('baseline_auc_ab','game_20260530_143503_0027.txt'),
('baseline_auc_ab','game_20260530_143549_0028.txt'),('baseline_auc_ab','game_20260530_143610_0030.txt'),
('baseline_auc_s9999','game_20260530_145004_0012.txt'),('baseline_auc_s9999','game_20260530_145017_0015.txt'),
('baseline_auc_s9999','game_20260530_145233_0031.txt'),('baseline_auc_s9999','game_20260530_145237_0032.txt'),
('baseline_auc_big5151','game_20260530_154559_0081.txt'),('baseline_auc_big5151','game_20260530_154620_0083.txt'),
('baseline_auc_big5151','game_20260530_154658_0088.txt'),('baseline_auc_big5151','game_20260530_154741_0092.txt'),
('baseline_auc_big5151','game_20260530_154816_0097.txt'),('baseline_auc_big5151','game_20260530_154838_0099.txt'),
('baseline_auc_big5151','game_20260530_154928_0105.txt'),('baseline_auc_big5151','game_20260530_154936_0106.txt'),
('baseline_auc_big5151','game_20260530_154950_0107.txt'),('baseline_auc_big5151','game_20260530_155104_0115.txt'),
('baseline_auc_big5151','game_20260530_155155_0118.txt'),('baseline_auc_big5151','game_20260530_155252_0122.txt'),
('baseline_auc_big5151','game_20260530_155253_0123.txt'),('baseline_auc_big5151','game_20260530_155320_0124.txt'),
]

def parse(fp):
    ds=[]
    with open(fp,encoding="utf-8",errors="replace") as f:
        for line in f:
            line=line.strip()
            if line.startswith("REPLAY_DECISION_JSON:"):
                try: ds.append(json.loads(line[len("REPLAY_DECISION_JSON:"):].strip()))
                except: pass
    return ds

mulled_away_spy=0   # saw 7-card hand w/ Spy, chose MULLIGAN
bottomed_spy=0      # bottomed a Spy during London
bottomed_blue=0     # bottomed Saruli/Petal during London
final_kept_4orless=0
final_no_blue=0
final_no_spy=0
total=len(files)
final_sizes=[]

for run,fn in files:
    ds=parse(fp:=os.path.join(base,run,MD,fn))
    # walk mulligan decisions
    saw_spy_then_mulled=False
    for d in ds:
        if d.get("action_type")=="MULLIGAN" and d.get("selected_text")=="MULLIGAN":
            if "Balustrade Spy" in " ".join(d.get("hand") or []):
                saw_spy_then_mulled=True
        if d.get("action_type")=="LONDON_MULLIGAN":
            sel=d.get("selected_text") or ""
            if sel=="Balustrade Spy": bottomed_spy+=1
            if sel in BLUE: bottomed_blue+=1
    if saw_spy_then_mulled: mulled_away_spy+=1
    # final kept hand
    keep=[d for d in ds if d.get("action_type")=="MULLIGAN" and d.get("selected_text")=="KEEP"]
    fh=keep[-1].get("hand") if keep else (ds[0].get("hand") if ds else [])
    fh=fh or []
    final_sizes.append(len(fh))
    if len(fh)<=4: final_kept_4orless+=1
    if not any(b in fh for b in BLUE): final_no_blue+=1
    if "Balustrade Spy" not in fh: final_no_spy+=1

print(f"Across {total} no-Spy-cast games:")
print(f"  Mulliganed AWAY a 7-card hand that contained Spy: {mulled_away_spy}")
print(f"  London-bottomed a Balustrade Spy (instances): {bottomed_spy}")
print(f"  London-bottomed a blue source Saruli/Petal (instances): {bottomed_blue}")
print(f"  Final kept hand <=4 cards: {final_kept_4orless}")
print(f"  Final kept hand with NO blue source: {final_no_blue}")
print(f"  Final kept hand with NO Spy: {final_no_spy}")
print(f"  Final hand sizes: {sorted(final_sizes)}  mean={sum(final_sizes)/len(final_sizes):.2f}")
