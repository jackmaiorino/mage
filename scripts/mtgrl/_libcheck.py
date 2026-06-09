import json, sys, glob, os
LANDS={"Forest","Swamp","Island","Mountain","Plains"}
def load(path):
    decs=[]; result=None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line=line.strip()
            if line.startswith("RESULT:"): result=line.split("RESULT:",1)[1].strip()
            elif line.startswith("REPLAY_DECISION_JSON:"):
                try: decs.append(json.loads(line.split("REPLAY_DECISION_JSON:",1)[1].strip()))
                except: pass
    return decs,result
files=sys.argv[1:]
for f in files:
    decs,result=load(f)
    # find the Spy cast decision; report its library landcount & size, plus mulligan count
    mull_keeps=0; mull_count=0
    for d in decs:
        if d.get("action_type")=="MULLIGAN":
            mull_count+=1
            if (d.get("selected_text") or "").upper().startswith("KEEP"): pass
    spy=None
    for i,d in enumerate(decs):
        st=d.get("selected_text") or ""
        if "Cast Balustrade Spy" in st:
            spy=d; spy_i=i; break
    lib_at_cast=None; lands_in_lib=None
    if spy is not None:
        lib=spy.get("library")
        if isinstance(lib,list):
            lands_in_lib=sum(1 for c in lib if c in LANDS)
            lib_at_cast=len(lib)
    # min library after cast
    minlib=99
    if spy is not None:
        for d in decs[spy_i:]:
            ls=d.get("library_size")
            if ls is not None: minlib=min(minlib,ls)
    spyturn=spy.get("turn") if spy else None
    print(f"{os.path.basename(f)[:30]:30s} res={result} mullDecs={mull_count} spyT={spyturn} libAtCast={lib_at_cast} landsInLib={lands_in_lib} minLibAfter={minlib}")
