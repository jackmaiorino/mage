#!/usr/bin/env python3
"""Download files from remote HPC by piping base64 through SSH exec channel."""
import argparse
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from transfer_hpc_file import connect_transport


def download_file(transport, remote_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)

    channel = transport.open_session()
    channel.exec_command(f"base64 '{remote_path}'")

    out = b""
    while True:
        chunk = channel.recv(65536)
        if not chunk:
            break
        out += chunk
    err = b""
    while True:
        chunk = channel.recv_stderr(4096)
        if not chunk:
            break
        err += chunk

    exit_code = channel.recv_exit_status()
    channel.close()

    if exit_code != 0:
        print(f"  FAILED: {remote_path} (exit {exit_code})", file=sys.stderr)
        if err:
            print(f"  stderr: {err.decode('utf-8', errors='replace')}", file=sys.stderr)
        return False

    data = base64.b64decode(out)
    local_path.write_bytes(data)
    print(f"  {local_path} ({len(data)} bytes)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-file", required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--local-path", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--known-hosts",
        default=str(Path.home() / ".ssh" / "known_hosts"),
    )
    args = parser.parse_args()

    _credential, transport = connect_transport(
        credential_file=Path(args.credential_file),
        port=args.port,
        timeout_seconds=args.timeout_seconds,
        known_hosts_path=Path(args.known_hosts),
    )

    try:
        return 0 if download_file(transport, args.remote_path, Path(args.local_path)) else 1
    finally:
        transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
