#!/usr/bin/env python3
"""Upload local files to Zaratan remote checkout in a single SSH session."""
import ctypes
import ctypes.wintypes as wt
import json
import os
import sys
from pathlib import Path

import paramiko

CRYPTPROTECT_UI_FORBIDDEN = 0x1

class DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]

crypt32 = ctypes.windll.crypt32
kernel32 = ctypes.windll.kernel32

def _blob_from_bytes(data):
    if not data:
        return DATA_BLOB(0, None)
    buf = (ctypes.c_byte * len(data)).from_buffer_copy(data)
    return DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_byte)))

def _bytes_from_blob(blob):
    if blob.cbData == 0:
        return b""
    return ctypes.string_at(blob.pbData, blob.cbData)

def dpapi_decrypt(cipher_b64):
    import base64
    cipher = base64.b64decode(cipher_b64)
    inp = _blob_from_bytes(cipher)
    out = DATA_BLOB()
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(inp), None, None, None, None, CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out)
    )
    if not ok:
        raise OSError("CryptUnprotectData failed")
    result = _bytes_from_blob(out)
    kernel32.LocalFree(out.pbData)
    return result.decode("utf-8")

def load_credential(path):
    with open(path) as f:
        cred = json.load(f)
    cred["password"] = dpapi_decrypt(cred["ciphertext_b64"])
    return cred

def interactive_handler(password, duo_choice):
    def handle(title, instructions, prompt_list):
        responses = []
        for prompt, _echo in prompt_list:
            p = prompt.lower()
            if "password" in p:
                responses.append(password)
            elif "duo" in p or "passcode" in p or "factor" in p or "option" in p:
                responses.append(duo_choice)
            else:
                responses.append("")
        return responses
    return handle

def connect(cred):
    import socket as _socket
    hostname = cred["host"]
    username = cred["username"]
    password = cred["password"]
    duo_choice = str(cred.get("duo_choice", "1"))

    sock = _socket.create_connection((hostname, 22), timeout=60)
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=60)
    try:
        transport.auth_interactive(
            username=username,
            handler=interactive_handler(password, duo_choice),
        )
    except TypeError:
        transport.auth_interactive(username, interactive_handler(password, duo_choice))

    if not transport.is_authenticated():
        raise SystemExit("Authentication failed.")
    return transport

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <credential-file> <remote-base> <local:remote ...>")
        sys.exit(1)

    cred_path = sys.argv[1]
    remote_base = sys.argv[2]
    file_pairs = sys.argv[3:]

    cred = load_credential(cred_path)
    print("Connecting (approve Duo push)...")
    transport = connect(cred)
    sftp = paramiko.SFTPClient.from_transport(transport)

    for pair in file_pairs:
        local_path, remote_rel = pair.split(":", 1)
        remote_path = f"{remote_base}/{remote_rel}"
        print(f"  {local_path} -> {remote_path}")
        with open(local_path, "rb") as f:
            data = f.read()
        # Normalize CRLF to LF for shell/python scripts
        if local_path.endswith((".sh", ".py", ".json", ".sbatch")):
            data = data.replace(b"\r\n", b"\n")
        with sftp.file(remote_path, "wb") as rf:
            rf.write(data)
        print(f"    wrote {len(data)} bytes")

    sftp.close()

    # Run post-upload command
    print("Verifying...")
    chan = transport.open_session(timeout=30)
    chan.exec_command(
        f"cd {remote_base} && git diff --stat && chmod +x scripts/hpc/submit_league_multinode.sh 2>/dev/null; echo DONE"
    )
    out = chan.makefile("rb", -1).read().decode("utf-8", errors="replace")
    print(out)
    chan.close()
    transport.close()
    print("All files transferred.")

if __name__ == "__main__":
    main()
