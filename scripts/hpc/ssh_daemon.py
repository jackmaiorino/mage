#!/usr/bin/env python3
"""Persistent SSH daemon for the HPC: authenticate ONCE (one Duo push), then serve
unlimited commands AND file transfers over the SAME connection via a localhost
socket -- NO further Duo.

Why this works: with paramiko, Duo 2FA happens at AUTHENTICATION. New channels on
the already-authenticated transport (open_session / open_sftp_client) do NOT
re-authenticate -> no new Duo.

Start (approve the ONE Duo push -- keep this window OPEN):
  py scripts/hpc/ssh_daemon.py --credential-file "%LOCALAPPDATA%/Codex/umd-hpc/credentials/zaratan.json"
Then (no Duo):
  py scripts/hpc/ssh_send.py "squeue -u jmaior"
  py scripts/hpc/ssh_send.py --put local.py /scratch/.../remote.py
  py scripts/hpc/ssh_send.py --get /scratch/.../log.out local_copy.out
Stop: Ctrl-C. If the SSH connection drops, restart (one more Duo).
"""
import argparse
import base64
import json
import posixpath
import socket
import struct
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from transfer_hpc_file import connect_transport


def _exec(transport, command, timeout):
    ch = transport.open_session()
    ch.settimeout(timeout)
    ch.exec_command(command)
    out = b""
    err = b""
    deadline = time.time() + timeout
    timed_out = False
    while True:
        progressed = False
        while ch.recv_ready():
            out += ch.recv(65536); progressed = True
        while ch.recv_stderr_ready():
            err += ch.recv_stderr(65536); progressed = True
        if ch.exit_status_ready() and not ch.recv_ready() and not ch.recv_stderr_ready():
            break
        if time.time() > deadline:
            timed_out = True; break
        if not progressed:
            time.sleep(0.02)
    try:
        while ch.recv_ready():
            out += ch.recv(65536)
        while ch.recv_stderr_ready():
            err += ch.recv_stderr(65536)
    except Exception:
        pass
    if timed_out:
        try:
            ch.close()           # abort the (possibly hung) remote command
        except Exception:
            pass
        return (out.decode("utf-8", "replace"),
                err.decode("utf-8", "replace") + "\n[ssh_daemon] TIMED OUT after %ds." % timeout, 124)
    code = ch.recv_exit_status() if ch.exit_status_ready() else 124
    try:
        ch.close()
    except Exception:
        pass
    return out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), code


def _send_json(conn, obj):
    data = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack(">I", len(data)) + data)


def _recv_line(conn):
    buf = b""
    while not buf.endswith(b"\n"):
        c = conn.recv(65536)
        if not c:
            break
        buf += c
    return buf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--credential-file", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--ssh-port", type=int, default=22)
    p.add_argument("--known-hosts", default=str(Path.home() / ".ssh" / "known_hosts"))
    args = p.parse_args()

    print("[ssh_daemon] connecting -- APPROVE THE ONE DUO PUSH NOW...", flush=True)
    _cred, transport = connect_transport(
        credential_file=Path(args.credential_file), port=args.ssh_port,
        timeout_seconds=300, known_hosts_path=Path(args.known_hosts))
    transport.set_keepalive(30)
    sftp_holder = {"client": None}

    def get_sftp():
        if sftp_holder["client"] is None:
            sftp_holder["client"] = transport.open_sftp_client()  # channel on authed transport -> no Duo
        return sftp_holder["client"]

    print("[ssh_daemon] CONNECTED. serving on 127.0.0.1:%d (exec + put + get). "
          "Keep this window open." % args.port, flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(8)
    lock = threading.Lock()
    n = 0
    try:
        while True:
            conn, _ = srv.accept()
            try:
                req = json.loads(_recv_line(conn).decode("utf-8"))
                op = req.get("op", "exec")
                if op == "ping" or req.get("command") == "__PING__":
                    _send_json(conn, {"alive": True, "ssh_active": transport.is_active(), "served": n})
                    continue
                if not transport.is_active():
                    _send_json(conn, {"stdout": "", "exit_code": 255, "ok": False,
                                      "stderr": "[ssh_daemon] SSH DEAD -- restart the daemon (one Duo)."})
                    continue
                with lock:
                    if op == "put":
                        remote = req["remote"]
                        data = base64.b64decode(req["data_b64"])
                        _exec(transport, "mkdir -p '%s'" % posixpath.dirname(remote), 30)
                        with get_sftp().file(remote, "wb") as f:
                            f.write(data)
                        n += 1
                        _send_json(conn, {"ok": True, "bytes": len(data), "remote": remote})
                    elif op == "get":
                        with get_sftp().file(req["remote"], "rb") as f:
                            data = f.read()
                        n += 1
                        _send_json(conn, {"ok": True, "data_b64": base64.b64encode(data).decode("ascii")})
                    else:
                        out, err, code = _exec(transport, req.get("command", ""), int(req.get("timeout", 120)))
                        n += 1
                        _send_json(conn, {"stdout": out, "stderr": err, "exit_code": code})
                        print("[ssh_daemon] #%d exit=%d: %s" % (n, code, req.get("command", "")[:80].replace("\n", " ")), flush=True)
            except Exception as e:
                try:
                    _send_json(conn, {"stdout": "", "ok": False, "exit_code": 255, "stderr": "[ssh_daemon] error: %s" % e})
                except Exception:
                    pass
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n[ssh_daemon] shutting down (served %d)." % n, flush=True)
    finally:
        try:
            transport.close()
        except Exception:
            pass
        srv.close()


if __name__ == "__main__":
    raise SystemExit(main())
