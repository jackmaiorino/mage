#!/usr/bin/env python3
"""Upload files to remote by piping base64 through SSH exec channel stdin."""
import argparse
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from transfer_hpc_file import connect_transport


def upload_file(transport, local_path: Path, remote_path: str) -> None:
    data = local_path.read_bytes()
    if local_path.suffix in (".py", ".sh"):
        data = data.replace(b"\r\n", b"\n")
    b64 = base64.b64encode(data)

    channel = transport.open_session()
    channel.exec_command(f"base64 -d > '{remote_path}' && wc -c '{remote_path}'")
    channel.sendall(b64)
    channel.shutdown_write()

    # Read stdout
    out = b""
    while True:
        chunk = channel.recv(4096)
        if not chunk:
            break
        out += chunk
    # Read stderr
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

    print(f"  {remote_path}: {out.decode().strip()}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-file", required=True)
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("files", nargs="+", help="local_path:relative_remote_path")
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
        ok = True
        for spec in args.files:
            local_str, rel_remote = spec.split(":", 1)
            local_path = Path(local_str)
            remote_path = f"{args.remote_base}/{rel_remote}"
            if not upload_file(transport, local_path, remote_path):
                ok = False
        if ok:
            print("All files uploaded.")
        return 0 if ok else 1
    finally:
        transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
