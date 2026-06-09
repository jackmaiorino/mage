#!/usr/bin/env python3
"""Run an arbitrary command on the HPC login node via SSH exec (credential-based).

Usage:
  python remote_exec.py --credential-file <zaratan.json> --command "squeue -u $USER"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from transfer_hpc_file import connect_transport


def run(transport, command: str, timeout: int) -> int:
    channel = transport.open_session()
    channel.settimeout(timeout)
    channel.exec_command(command)
    out = b""
    err = b""
    while True:
        if channel.recv_ready():
            c = channel.recv(65536)
            if not c:
                break
            out += c
        elif channel.exit_status_ready() and not channel.recv_ready():
            break
    while channel.recv_ready():
        out += channel.recv(65536)
    while channel.recv_stderr_ready():
        err += channel.recv_stderr(65536)
    code = channel.recv_exit_status()
    while channel.recv_ready():
        out += channel.recv(65536)
    while channel.recv_stderr_ready():
        err += channel.recv_stderr(65536)
    channel.close()
    sys.stdout.write(out.decode("utf-8", errors="replace"))
    if err:
        sys.stderr.write(err.decode("utf-8", errors="replace"))
    return code


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--credential-file", required=True)
    p.add_argument("--command", required=True)
    p.add_argument("--port", type=int, default=22)
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.add_argument("--known-hosts", default=str(Path.home() / ".ssh" / "known_hosts"))
    args = p.parse_args()
    _cred, transport = connect_transport(
        credential_file=Path(args.credential_file),
        port=args.port,
        timeout_seconds=args.timeout_seconds,
        known_hosts_path=Path(args.known_hosts),
    )
    try:
        return run(transport, args.command, args.timeout_seconds)
    finally:
        transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
