#!/usr/bin/env python3
"""Hot-patch the stale HPC checkout with current code via the running ssh_daemon
(NO Duo). The HPC checkout is a non-git snapshot; the Java bundle rides separately,
but Python/registry/scripts load from the checkout -> must be synced each change.

  py scripts/hpc/hotpatch_via_daemon.py            # patch the default file set
  py scripts/hpc/hotpatch_via_daemon.py a.py:rel/b.py   # extra explicit local:remote pairs
"""
import base64
import glob
import json
import socket
import struct
import sys
from pathlib import Path

PORT = 8765
REMOTE_BASE = "/scratch/zt1/project/msml603/user/jmaior/jmaior/mage"
REPO = Path(__file__).resolve().parents[2]
MLP = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode"

DEFAULT_GLOBS = [
    MLP + "/*.py",
    "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json",
    "scripts/hpc/run_spy_pbt_native.py",
    "scripts/hpc/run_spy_pbt.sh",
    "scripts/hpc/gpu_head.sbatch",
    "scripts/hpc/cpu_worker.sh",
    "scripts/run_local_pbt.py",
]


def _put(local: Path, remote: str):
    data = local.read_bytes()
    # Slurm + bash reject CRLF in scripts; normalize shell scripts to LF on upload.
    if local.suffix in (".sh", ".sbatch"):
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    req = {"op": "put", "remote": remote, "data_b64": base64.b64encode(data).decode("ascii")}
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(120)
    s.connect(("127.0.0.1", PORT))
    s.sendall((json.dumps(req) + "\n").encode("utf-8"))
    hdr = b""
    while len(hdr) < 4:
        c = s.recv(4 - len(hdr))
        if not c:
            break
        hdr += c
    n = struct.unpack(">I", hdr)[0]
    body = b""
    while len(body) < n:
        c = s.recv(n - len(body))
        if not c:
            break
        body += c
    s.close()
    return json.loads(body.decode("utf-8"))


def main():
    pairs = []
    for g in DEFAULT_GLOBS:
        for lp in sorted(glob.glob(str(REPO / g))):
            lp = Path(lp)
            rel = lp.relative_to(REPO).as_posix()
            pairs.append((lp, REMOTE_BASE + "/" + rel))
    for arg in sys.argv[1:]:
        l, r = arg.split(":", 1)
        pairs.append((REPO / l, r if r.startswith("/") else REMOTE_BASE + "/" + r))

    ok = 0
    for lp, rp in pairs:
        if not lp.exists():
            print("SKIP (missing local): %s" % lp); continue
        try:
            resp = _put(lp, rp)
            if resp.get("ok"):
                print("OK  %6d B  %s" % (resp.get("bytes", 0), rp))
                ok += 1
            else:
                print("FAIL %s -> %s" % (lp, resp))
        except Exception as e:
            print("ERR %s: %s" % (lp, e))
    print("--- hot-patched %d/%d files ---" % (ok, len(pairs)))
    return 0 if ok == len(pairs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
