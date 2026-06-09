import json, glob, os

base = r"C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
runs = ["baseline_auc_ab","baseline_auc_s9999","baseline_auc_big5151"]

BLUE_SOURCES = ["Saruli Caretaker","Lotus Petal"]
DIG = ["Lead the Stampede","Winding Way"]

def parse(fp):
    decisions=[]; result=None; mulls=0
    with open(fp,encoding="utf-8",errors="replace") as f:
        for line in f:
            line=line.strip()
            if line.startswith("REPLAY_DECISION_JSON:"):
                try: decisions.append(json.loads(line[len("REPLAY_DECISION_JSON:"):].strip()))
                except: pass
            elif line.startswith("RESULT:"):
                result=line.split(":",1)[1].strip()
    return decisions,result

def analyze(fp):
    decisions,result=parse(fp)
    if not decisions: return None
    # mulligan info
    mull_decisions=[d for d in decisions if d.get("action_type")=="MULLIGAN"]
    mulligans_taken=0
    final_keep_hand=None
    for d in decisions:
        if d.get("action_type")=="MULLIGAN":
            mt=d.get("mulligans_taken",0)
            if mt>mulligans_taken: mulligans_taken=mt
    # find final kept opening hand: last MULLIGAN decision with KEEP
    keep_hands=[d for d in mull_decisions if d.get("selected_text")=="KEEP"]
    if keep_hands: final_keep_hand=keep_hands[-1].get("hand")
    # whether Spy ever in hand at any decision
    spy_in_hand_ever=False
    spy_first_turn=None
    turns_seen=set()
    last_turn=0
    last_lib_size=None
    # did agent ever cast a dig spell
    dig_cast=[]
    # track blue source presence/cast
    blue_src_in_hand_with_spy=False
    spy_castable_states=0
    # count how many lands ever drawn into hand
    for d in decisions:
        hand=d.get("hand") or []
        if any("Balustrade Spy" in c for c in hand):
            spy_in_hand_ever=True
            if spy_first_turn is None: spy_first_turn=d.get("turn")
        st=d.get("selected_text") or ""
        for dg in DIG:
            if ("Cast %s"%dg) in st:
                dig_cast.append((dg,d.get("turn")))
        last_turn=max(last_turn,d.get("turn") or 0)
        if d.get("library_size") is not None: last_lib_size=d.get("library_size")
    # final hand at last decision
    final_hand = decisions[-1].get("hand")
    # spy ever in library at start
    spy_in_lib_start = any("Balustrade Spy" in c for c in (decisions[0].get("library") or []))
    return {
        "result":result,
        "n_dec":len(decisions),
        "mulligans_taken":mulligans_taken,
        "keep_hand":final_keep_hand,
        "spy_in_hand_ever":spy_in_hand_ever,
        "spy_first_turn":spy_first_turn,
        "dig_cast":dig_cast,
        "last_turn":last_turn,
        "last_lib_size":last_lib_size,
        "final_hand":final_hand,
    }

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

drew_spy=0; never_drew_spy=0
for run,fn in nospy_files:
    fp=os.path.join(base,run,"game_logs","Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity",fn)
    a=analyze(fp)
    if a is None: 
        print("PARSE FAIL",fn); continue
    if a["spy_in_hand_ever"]: drew_spy+=1
    else: never_drew_spy+=1
    print(f"\n=== {run}/{fn[:30]} RESULT={a['result']} dec={a['n_dec']} lastTurn={a['last_turn']} mulls={a['mulligans_taken']}")
    print(f"  spy_in_hand_ever={a['spy_in_hand_ever']} (firstTurn={a['spy_first_turn']})  digCast={a['dig_cast']}  lastLibSize={a['last_lib_size']}")
    print(f"  KEEP_HAND={a['keep_hand']}")
print(f"\n\nSPLIT: drew/had Spy in hand at some point={drew_spy}  never saw Spy={never_drew_spy}  total={len(nospy_files)}")
