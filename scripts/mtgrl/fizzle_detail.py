import json, os, glob, sys

LANDS = {"Forest", "Swamp"}
BASE = "C:/Users/Jack/IdeaProjects/mage/local-training/local_pbt/cp7_eval_sweeps"
SUB = "game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity"

def parse_game(path):
    decisions = []
    result = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("REPLAY_DECISION_JSON:"):
                try:
                    decisions.append(json.loads(line[len("REPLAY_DECISION_JSON:"):].strip()))
                except Exception:
                    pass
            elif line.startswith("RESULT:"):
                result = line.split("RESULT:")[1].strip()
    return decisions, result

def find(fname):
    for dd in os.listdir(BASE):
        p = os.path.join(BASE, dd, SUB, fname)
        if os.path.exists(p):
            return p
    # prefix match
    for dd in os.listdir(BASE):
        gdir = os.path.join(BASE, dd, SUB)
        if not os.path.isdir(gdir): continue
        for f in os.listdir(gdir):
            if f.startswith(fname):
                return os.path.join(gdir, f)
    return None

def main():
    fname = sys.argv[1]
    path = find(fname)
    print("FILE:", path)
    decisions, result = parse_game(path)
    print("RESULT:", result, " n_decisions:", len(decisions))
    spy_idxs = [i for i, d in enumerate(decisions)
                if "Balustrade Spy" in (d.get("selected_text") or "")
                and "cast" in (d.get("selected_text") or "").lower()]
    print("spy cast decision idxs:", spy_idxs)
    for si in spy_idxs:
        d = decisions[si]
        lib = d.get("library", [])
        lands = [c for c in lib if c in LANDS]
        print(f"\n=== SPY CAST at decision idx {si} (turn {d.get('turn')}) ===")
        print(f"  selected_text: {d.get('selected_text')}")
        print(f"  hand: {d.get('hand')}")
        print(f"  library_size: {d.get('library_size')}, lands in lib: {len(lands)} -> {lands}")
        print(f"  full library order (top first): {lib}")
    # Show the SELECT_TARGETS right after the cast and library size trajectory
    print("\n=== library_size trajectory from first spy cast ===")
    if spy_idxs:
        for i in range(spy_idxs[0], min(spy_idxs[0]+8, len(decisions))):
            d = decisions[i]
            print(f"  idx {i} turn{d.get('turn')} {d.get('action_type')}: '{d.get('selected_text')}' lib={d.get('library_size')} gy={d.get('graveyard_size')}")

if __name__ == "__main__":
    main()
