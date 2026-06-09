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
            print("RESULT:",line.split(":",1)[1].strip())

def fmt_hand(h):
    if h is None: return "?"
    # abbreviate
    return ",".join(h)

prev_turn=None
for d in decisions:
    at=d.get("action_type","")
    turn=d.get("turn")
    st=(d.get("selected_text") or "")[:55]
    libsz=d.get("library_size")
    hand=d.get("hand") or []
    val=d.get("value_score")
    # Only show interesting decisions: mulligans, casts, plays, and any decision where Spy is in hand
    spy_in_hand="Balustrade Spy" in " ".join(hand)
    interesting = at in ("MULLIGAN",) or "Cast" in st or "Play" in st or spy_in_hand or "Spy" in st
    if turn != prev_turn:
        print(f"  --- TURN {turn} (lib={libsz}) ---")
        prev_turn=turn
    if interesting:
        marker = " <<<SPY_IN_HAND" if spy_in_hand else ""
        print(f"   [{at}] {st!r} v={val}{marker}")
        if spy_in_hand and (at=="MULLIGAN" or "Cast" not in st):
            # show full hand + candidates briefly when spy is sitting in hand
            cands=d.get("candidate_texts") or []
            print(f"        HAND={fmt_hand(hand)}")
            if len(cands)<=8:
                print(f"        CANDS={cands}")
