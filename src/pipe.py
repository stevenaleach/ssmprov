# /_/src/pipe.py
# Scriptable CLI pipe for SSM runner service with turn templating.

import argparse
import socket
import sys
from pathlib import Path
from typing import Optional

HOST = "127.0.0.1"
DEFAULT_PORT = 6502
NULL = b"\x00"

def read_all_stdin() -> str:
    # Read raw stdin as text; we trim only the final single LF if present.
    s = sys.stdin.read()
    if s.endswith("\n"):
        s = s[:-1]
    return s

def ensure_counter_file() -> None:
    p = Path(".counter")
    if not p.exists():
        p.write_text("0", encoding="utf-8")

def read_counter() -> int:
    ensure_counter_file()
    txt = Path(".counter").read_text(encoding="utf-8").strip()
    try:
        return int(txt or "0")
    except ValueError:
        return 0

def write_counter(n: int) -> None:
    Path(".counter").write_text(str(n), encoding="utf-8")

def send_roundtrip_bytes(payload_text: str, port: int, connect_timeout: float, recv_timeout: Optional[float]) -> bytes:
    """
    Send UTF-8 encoded text terminated by NULL; return raw bytes up to (but not including) NULL.
    No decoding here—callers decide how/if to decode. This preserves exact newlines.
    """
    to_send = payload_text.encode("utf-8") + NULL
    with socket.create_connection((HOST, port), timeout=connect_timeout) as s:
        s.settimeout(recv_timeout)
        s.sendall(to_send)
        buf = bytearray()
        while True:
            part = s.recv(4096)
            if not part:
                raise ConnectionError("connection closed before NULL terminator")
            i = part.find(NULL)
            if i != -1:
                buf.extend(part[:i])
                break
            buf.extend(part)
    return bytes(buf)

def build_prompt_body(raw: str, in_role: Optional[str], out_role: Optional[str]) -> tuple[str, Optional[int], Optional[int]]:
    """
    Build the body to send for prompt modes (non-slash).
    Returns (body_text, minted_in, minted_out).
    - If --in and --out:  (Turn N)[IN]: <raw>  + full fence  + (Turn M)[OUT]:
    - If --in only:       (Turn N)[IN]: <raw>  + half divider
    - If no roles:        <raw>
    """
    if not in_role and not out_role:
        return (raw, None, None)
    if out_role and not in_role:
        raise ValueError("--out requires --in")

    current = read_counter()
    minted_in = current + 1
    in_hdr = f"(Turn {minted_in}) [{in_role}]: "

    if out_role:
        minted_out = minted_in + 1
        out_hdr = f"(Turn {minted_out}) [{out_role}]: "
        # Full fence BETWEEN IN body and OUT header.
        body = in_hdr + raw + "\n\n~~~(end)~~~\n\n" + out_hdr
        return (body, minted_in, minted_out)

    # IN-only: add half divider at end
    body = in_hdr + raw + "\n\n~~~("
    return (body, minted_in, None)

def append_transcript_binary(chunks_text_first: list[str], reply_bytes: bytes, debang_prefix_len: int = 0) -> None:
    """
    Append exactly what was sent (as UTF-8 text) and exactly what was received (bytes) to .transcript.txt.
    If debang is active, strip the first prefix_len bytes (characters) from the first chunk after encoding.
    """
    with open(".transcript.txt", "ab") as f:
        for idx, c in enumerate(chunks_text_first):
            b = c.encode("utf-8")
            if idx == 0 and debang_prefix_len > 0:
                b = b[debang_prefix_len:]
            f.write(b)
        f.write(reply_bytes)

def main():
    ap = argparse.ArgumentParser(description="PIPE: pipe/paste text to SSM runner with optional transcript templating.")
    ap.add_argument("--in", dest="in_role", help="Input role name (e.g., USER)")
    ap.add_argument("--out", dest="out_role", help="Output role name (e.g., INTERFACE)")
    ap.add_argument("--bang", dest="bang", help="Raw prefix line(s) prepended as-is before the prompt")
    ap.add_argument("--counter", dest="counter_override", type=int, help="Explicitly set .counter to this value on success")
    ap.add_argument("--port", dest="port", type=int, default=DEFAULT_PORT, help=f"Runner TCP port (default {DEFAULT_PORT})")
    ap.add_argument("--connect-timeout", type=float, default=3.0, help="TCP connect timeout (seconds)")
    ap.add_argument("--recv-timeout", type=float, default=600.0, help="Receive timeout (seconds)")
    ap.add_argument("--debang", action="store_true",
                    help="Strip the leading prefix line(s) from transcript (wire text unchanged)")

    args = ap.parse_args()
    text_in = read_all_stdin()
    if text_in == "":
        sys.exit(0)

    is_slash = text_in.startswith("/")
    prefix = (args.bang + "\n") if args.bang else ""

    minted_in = minted_out = None
    if is_slash:
        send_text = prefix + text_in
    else:
        try:
            body, minted_in, minted_out = build_prompt_body(text_in, args.in_role, args.out_role)
        except ValueError as ve:
            sys.stderr.write(f"[ERROR] {ve}\n")
            sys.exit(2)
        send_text = prefix + body

    try:
        reply_bytes = send_roundtrip_bytes(send_text, args.port, args.connect_timeout, args.recv_timeout)
    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        sys.exit(1)

    # Emit reply to stdout exactly
    try:
        sys.stdout.buffer.write(reply_bytes)
        sys.stdout.buffer.flush()
    except Exception:
        pass

    if is_slash:
        if args.counter_override is not None:
            write_counter(args.counter_override)
        return

    # Figure transcript-sent chunk (optionally sans prefix)
    debang_prefix_len = 0
    transcript_sent = send_text
    if args.debang and prefix:
        nl = send_text.find("\n")
        if nl != -1:
            debang_prefix_len = nl + 1  # number of characters to drop from encoded bytes
        else:
            transcript_sent = ""  # nothing after prefix

    # Append exactly what was sent (UTF-8) + exactly what was received (bytes)
    append_transcript_binary([transcript_sent], reply_bytes, debang_prefix_len=debang_prefix_len)

    # Counter update rules
    if args.counter_override is not None:
        write_counter(args.counter_override)
    else:
        if minted_out is not None:
            write_counter(minted_out)
        elif minted_in is not None:
            write_counter(minted_in)

if __name__ == "__main__":
    main()
