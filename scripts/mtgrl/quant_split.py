import json, glob, os

base = r"C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
MD="game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity"

nospy_files=[
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
    decisions=[]; res=None
    with open(fp,encoding="utf-8",errors="replace") as f:
        for line in f:
            line=line.strip()
            if line.startswith("REPLAY_DECISION_JSON:"):
                try: decisions.append(json.loads(line[len("REPLAY_DECISION_JSON:"):].strip()))
                except: pass
            elif line.startswith("RESULT:"): res=line.split(":",1)[1].strip()
    return decisions,res

cat_never_drew=[]      # Spy never in hand
cat_legal_skipped=[]   # Spy WAS a legal cast candidate but agent chose not to
cat_mana_screw=[]      # Spy in hand but never legal (couldn't make 2U)

for run,fn in nospy_files:
    fp=os.path.join(base,run,MD,fn)
    decisions,res=parse(fp)
    spy_in_hand=any("Balustrade Spy" in " ".join(d.get("hand") or []) for d in decisions)
    spy_legal=any(any("Balustrade Spy" in c and "Cast" in c for c in (d.get("candidate_texts") or [])) for d in decisions)
    last_turn=max((d.get("turn") or 0) for d in decisions)
    # count blue sources cast (Saruli, Lotus Petal) and how many lands played
    saruli=sum(1 for d in decisions if "Cast Saruli Caretaker"==(d.get("selected_text") or ""))
    petal=sum(1 for d in decisions if "Cast Lotus Petal"==(d.get("selected_text") or ""))
    mulls=max((d.get("mulligans_taken") or 0) for d in decisions)
    rec=(run[:14],fn[-8:-4],res,last_turn,mulls,f"saruli={saruli}",f"petal={petal}")
    if not spy_in_hand:
        cat_never_drew.append(rec)
    elif spy_legal:
        cat_legal_skipped.append(rec)
    else:
        cat_mana_screw.append(rec)

print(f"=== (a) NEVER DREW SPY: {len(cat_never_drew)}")
for r in cat_never_drew: print("   ",r)
print(f"\n=== (c) HAD SPY but NEVER could cast (no 2U / mana screw): {len(cat_mana_screw)}")
for r in cat_mana_screw: print("   ",r)
print(f"\n=== (b) SPY WAS LEGAL but agent CHOSE not to cast: {len(cat_legal_skipped)}")
for r in cat_legal_skipped: print("   ",r)

# winrate by category
def wr(cat): 
    if not cat: return "n/a"
    w=sum(1 for r in cat if r[2]=="WIN"); return f"{w}/{len(cat)}"
print(f"\nWinrate: never_drew={wr(cat_never_drew)}  mana_screw={wr(cat_mana_screw)}  legal_skipped={wr(cat_legal_skipped)}")
