#!/usr/bin/env python3
import argparse
import base64
import ctypes
import ctypes.wintypes as wt
import json
import os
import posixpath
import socket
import sys
from pathlib import Path


CRYPTPROTECT_UI_FORBIDDEN = 0x1


class DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wt.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


crypt32 = ctypes.windll.crypt32
kernel32 = ctypes.windll.kernel32


def _blob_from_bytes(data: bytes) -> DATA_BLOB:
    if not data:
        return DATA_BLOB(0, None)
    buffer = (ctypes.c_byte * len(data)).from_buffer_copy(data)
    return DATA_BLOB(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))


def _bytes_from_blob(blob: DATA_BLOB) -> bytes:
    if blob.cbData == 0:
        return b""
    return ctypes.string_at(blob.pbData, blob.cbData)


def unprotect(ciphertext: bytes) -> bytes:
    in_blob = _blob_from_bytes(ciphertext)
    out_blob = DATA_BLOB()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    ):
        raise ctypes.WinError()
    try:
        return _bytes_from_blob(out_blob)
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)


def load_credential(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["password"] = unprotect(base64.b64decode(data["ciphertext_b64"])).decode("utf-8")
    return data


def ensure_paramiko():
    try:
        import paramiko  # noqa: F401
    except ImportError as exc:
        raise SystemExit("paramiko is not installed. Run: python -m pip install --user paramiko") from exc


def verify_host_key(paramiko, hostname: str, port: int, key, known_hosts_path: Path) -> None:
    host_keys = paramiko.HostKeys()
    if known_hosts_path.exists():
        host_keys.load(str(known_hosts_path))

    candidates = [hostname]
    if port != 22:
        candidates.insert(0, f"[{hostname}]:{port}")

    for candidate in candidates:
        if host_keys.check(candidate, key):
            return

    raise SystemExit(f"Host key for {hostname}:{port} was not found in {known_hosts_path}")


def interactive_handler(password: str, duo_choice: str):
    def handle(title, instructions, prompt_list):
        responses = []
        for prompt, _echo in prompt_list:
            print(f"keyboard_interactive_prompt={prompt}", file=sys.stderr)
            prompt_lc = prompt.lower()
            if "password" in prompt_lc:
                responses.append(password)
            elif "duo" in prompt_lc or "passcode" in prompt_lc or "factor" in prompt_lc or "option" in prompt_lc:
                responses.append(duo_choice)
            else:
                raise RuntimeError(f"Unhandled keyboard-interactive prompt: {prompt!r}")
        return responses

    return handle


def connect_transport(credential_file: Path, port: int, timeout_seconds: int, known_hosts_path: Path):
    ensure_paramiko()
    import paramiko

    credential = load_credential(credential_file)
    hostname = credential["host"]
    username = credential["username"]
    duo_choice = str(credential.get("duo_choice", "1"))

    sock = socket.create_connection((hostname, port), timeout=timeout_seconds)
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=timeout_seconds)

    server_key = transport.get_remote_server_key()
    verify_host_key(paramiko, hostname, port, server_key, known_hosts_path)

    try:
        transport.auth_interactive(
            username=username,
            handler=interactive_handler(credential["password"], duo_choice),
        )
    except TypeError:
        transport.auth_interactive(username, interactive_handler(credential["password"], duo_choice))

    if not transport.is_authenticated():
        raise SystemExit("Authentication failed.")

    return credential, transport


def ensure_remote_dir(sftp, remote_dir: str) -> None:
    normalized = posixpath.normpath(remote_dir)
    if normalized in ("", "."):
        return

    parts = []
    current = normalized
    while current not in ("", "/", "."):
        parts.append(current)
        parent = posixpath.dirname(current)
        if parent == current:
            break
        current = parent

    for path in reversed(parts):
        try:
            sftp.stat(path)
        except OSError:
            sftp.mkdir(path)


def upload_file(sftp, local_path: Path, remote_path: str, mkdirs: bool, transport=None) -> None:
    if mkdirs:
        ensure_remote_dir(sftp, posixpath.dirname(remote_path))
    # Lustre filesystems can silently produce 0-byte files with sftp.put().
    # Work around by uploading to /tmp first (local disk), then cp into Lustre via exec.
    tmp_remote = f"/tmp/.transfer_{os.getpid()}_{local_path.name}"
    try:
        sftp.put(str(local_path), tmp_remote)
        tmp_stat = sftp.stat(tmp_remote)
        expected = local_path.stat().st_size
        if tmp_stat.st_size != expected:
            raise IOError(f"size mismatch after upload to tmp: {tmp_stat.st_size} != {expected}")
        if transport is not None:
            # Cross-filesystem: use cp via exec_command, then remove tmp
            remote_dir = posixpath.dirname(remote_path)
            cmd = f"mkdir -p '{remote_dir}' && cp '{tmp_remote}' '{remote_path}' && rm -f '{tmp_remote}'"
            chan = transport.open_session()
            chan.exec_command(cmd)
            stderr_data = chan.makefile_stderr("r").read()
            exit_status = chan.recv_exit_status()
            chan.close()
            if exit_status != 0:
                raise IOError(f"remote cp failed with exit code {exit_status}: {stderr_data}")
        else:
            sftp.rename(tmp_remote, remote_path)
    except Exception:
        try:
            sftp.remove(tmp_remote)
        except Exception:
            pass
        raise
    remote_stat = sftp.stat(remote_path)
    print(f"uploaded_remote={remote_path}")
    print(f"uploaded_size={remote_stat.st_size}")


def download_file(sftp, local_path: Path, remote_path: str) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(local_path))
    local_size = local_path.stat().st_size
    print(f"downloaded_local={local_path}")
    print(f"downloaded_size={local_size}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Transfer one file to or from UMD HPC with keyboard-interactive auth.")
    parser.add_argument("--credential-file", required=True, help="Path to the saved encrypted credential JSON")
    parser.add_argument("--mode", required=True, choices=("upload", "download"))
    parser.add_argument("--local-path", required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--mkdirs", action="store_true", help="Create parent directories on upload")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--known-hosts",
        default=str(Path.home() / ".ssh" / "known_hosts"),
        help="Known hosts file used for host key verification",
    )
    args = parser.parse_args()

    local_path = Path(args.local_path)
    if args.mode == "upload" and not local_path.exists():
        raise SystemExit(f"Local path does not exist: {local_path}")

    _credential, transport = connect_transport(
        credential_file=Path(args.credential_file),
        port=args.port,
        timeout_seconds=args.timeout_seconds,
        known_hosts_path=Path(args.known_hosts),
    )

    sftp = transport.open_sftp_client()
    try:
        if args.mode == "upload":
            upload_file(sftp, local_path, args.remote_path, args.mkdirs, transport=transport)
        else:
            download_file(sftp, local_path, args.remote_path)
    finally:
        sftp.close()
        transport.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
