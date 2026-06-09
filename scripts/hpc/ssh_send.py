#!/usr/bin/env python3
"""Talk to the running ssh_daemon (NO Duo). Start ssh_daemon.py first (one Duo).

  py scripts/hpc/ssh_send.py "squeue -u jmaior"            # run a command
  py scripts/hpc/ssh_send.py --timeout 300 "sbatch job.sh"
  py scripts/hpc/ssh_send.py --put local.py /scratch/.../remote.py   # upload
  py scripts/hpc/ssh_send.py --get /scratch/.../log.out copy.out     # download
  py scripts/hpc/ssh_send.py --ping
"""
import argparse
import base64
import json
import socket
import struct
import sys
from pathlib import Path


def _recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c:
            break
        buf += c
    return buf


def _roundtrip(req, port, sock_timeout):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(sock_timeout)
    try:
        s.connect(("127.0.0.1", port))
    except (ConnectionRefusedError, OSError):
        sys.stderr.write("[ssh_send] daemon not running on 127.0.0.1:%d -- "
                         "start scripts/hpc/ssh_daemon.py first (one Duo).\n" % port)
        sys.exit(3)
    s.sendall((json.dumps(req) + "\n").encode("utf-8"))
    hdr = _recv_exact(s, 4)
    if len(hdr) < 4:
        sys.stderr.write("[ssh_send] no response from daemon.\n")
        sys.exit(4)
    return json.loads(_recv_exact(s, struct.unpack(">I", hdr)[0]).decode("utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", nargs="?", default="", help="remote command (quote it)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--ping", action="store_true")
    p.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    p.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    args = p.parse_args()
    st = args.timeout + 30

    if args.ping:
        print(json.dumps(_roundtrip({"op": "ping"}, args.port, 15)))
        return 0
    if args.put:
        data = Path(args.put[0]).read_bytes()
        r = _roundtrip({"op": "put", "remote": args.put[1].replace("\\", "/"),
                        "data_b64": base64.b64encode(data).decode("ascii")}, args.port, st)
        if r.get("ok"):
            print("put %d bytes -> %s" % (r.get("bytes", 0), r.get("remote")))
            return 0
        sys.stderr.write("[ssh_send] put failed: %s\n" % r.get("stderr", r)); return 5
    if args.get:
        r = _roundtrip({"op": "get", "remote": args.get[0]}, args.port, st)
        if r.get("ok"):
            Path(args.get[1]).write_bytes(base64.b64decode(r["data_b64"]))
            print("got %s -> %s" % (args.get[0], args.get[1])); return 0
        sys.stderr.write("[ssh_send] get failed: %s\n" % r.get("stderr", r)); return 5

    r = _roundtrip({"op": "exec", "command": args.command, "timeout": args.timeout}, args.port, st)
    sys.stdout.write(r.get("stdout", ""))
    if r.get("stderr"):
        sys.stderr.write(r["stderr"])
    return int(r.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
