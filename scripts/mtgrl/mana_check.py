import json, sys, os
fp=sys.argv[1]
decisions=[]
with open(fp,encoding="utf-8",errors="replace") as f:
    for line in f:
        line=line.strip()
        if line.startswith("REPLAY_DECISION_JSON:"):
            try: decisions.append(json.loads(line[len("REPLAY_DECISION_JSON:"):].strip()))
            except: pass

# For each turn, list the full candidate set on the main-phase ACTIVATE decisions to see what was castable
seen=set()
for d in decisions:
    if d.get("action_type")!="ACTIVATE_ABILITY_OR_SPELL": continue
    turn=d.get("turn"); 
    cands=d.get("candidate_texts") or []
    hand=d.get("hand") or []
    spy_in_hand="Balustrade Spy" in " ".join(hand)
    if not spy_in_hand: continue
    key=(turn,tuple(cands))
    if key in seen: continue
    seen.add(key)
    casts=[c for c in cands if c.startswith("Cast ") or c.startswith("Play ")]
    print(f"T{turn} selected={d.get('selected_text')!r}")
    print(f"   castable: {casts}")
