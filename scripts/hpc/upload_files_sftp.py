#!/usr/bin/env python3
"""Upload local files to remote HPC via Paramiko SFTP with manual write."""
import argparse
import sys
from pathlib import Path

# Reuse credential/connection logic from transfer_hpc_file
sys.path.insert(0, str(Path(__file__).parent))
from transfer_hpc_file import connect_transport


def upload_binary(sftp, local_path: Path, remote_path: str) -> None:
    data = local_path.read_bytes()
    # Strip CRLF -> LF for Python files
    if local_path.suffix in (".py", ".sh"):
        data = data.replace(b"\r\n", b"\n")
    # Write via temp file + rename for atomicity
    tmp_path = remote_path + ".tmp"
    try:
        with sftp.open(tmp_path, "wb") as f:
            f.set_pipelined(True)
            offset = 0
            chunk_size = 32768
            while offset < len(data):
                f.write(data[offset : offset + chunk_size])
                offset += chunk_size
        sftp.rename(tmp_path, remote_path)
    except Exception:
        # If rename fails, try direct overwrite
        with sftp.open(remote_path, "wb") as f:
            f.set_pipelined(True)
            offset = 0
            chunk_size = 32768
            while offset < len(data):
                f.write(data[offset : offset + chunk_size])
                offset += chunk_size
    remote_stat = sftp.stat(remote_path)
    print(f"  {remote_path} ({remote_stat.st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-file", required=True)
    parser.add_argument("--remote-base", required=True, help="Remote directory prefix")
    parser.add_argument("files", nargs="+", help="local_path:relative_remote_path pairs")
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

    sftp = transport.open_sftp_client()
    try:
        for spec in args.files:
            local_str, rel_remote = spec.split(":", 1)
            local_path = Path(local_str)
            remote_path = f"{args.remote_base}/{rel_remote}"
            upload_binary(sftp, local_path, remote_path)
    finally:
        sftp.close()
        transport.close()

    print("All files uploaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
