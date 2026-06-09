import json, sys, os

fp=sys.argv[1]
decisions=[]
with open(fp,encoding="utf-8",errors="replace") as f:
    for line in f:
        line=line.strip()
        if line.startswith("REPLAY_DECISION_JSON:"):
            try: decisions.append(json.loads(line[len("REPLAY_DECISION_JSON:"):].strip()))
            except: pass
        elif line.startswith("RESULT:"):
            res=line.split(":",1)[1].strip()

# Show every cast/play decision + whether Spy was a candidate, tracking battlefield via casts
print("RESULT:",res)
spy_was_candidate_turns=set()
casts=[]
for d in decisions:
    st=(d.get("selected_text") or "")
    cands=d.get("candidate_texts") or []
    turn=d.get("turn")
    if any("Balustrade Spy" in c and "Cast" in c for c in cands):
        spy_was_candidate_turns.add(turn)
    if "Cast " in st or "Play " in st:
        casts.append((turn,st,d.get("value_score")))
print("Turns where 'Cast Balustrade Spy' was a LEGAL candidate:",sorted(spy_was_candidate_turns))
print("\nAll Cast/Play actions taken:")
for t,s,v in casts:
    print(f"  T{t}: {s}  v={v}")
