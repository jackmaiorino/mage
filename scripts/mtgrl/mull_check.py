import json, sys, os
fp=sys.argv[1]
decisions=[]
with open(fp,encoding="utf-8",errors="replace") as f:
    for line in f:
        line=line.strip()
        if line.startswith("REPLAY_DECISION_JSON:"):
            try: decisions.append(json.loads(line[len("REPLAY_DECISION_JSON:"):].strip()))
            except: pass

# Print all mulligan-related decisions in order to see KEEP and bottoming
for d in decisions:
    at=d.get("action_type")
    if at in ("MULLIGAN","LONDON_MULLIGAN"):
        print(f"[{at}] selected={d.get('selected_text')!r} mulls={d.get('mulligans_taken')} handsize={d.get('hand_size')}")
        print(f"     hand={d.get('hand')}")
# Find first post-mulligan hand (first non-mull decision)
for d in decisions:
    if d.get("action_type") not in ("MULLIGAN","LONDON_MULLIGAN"):
        print(f"\nFIRST PLAY DECISION hand={d.get('hand')}")
        break
