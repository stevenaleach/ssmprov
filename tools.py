#!/usr/bin/env python3
# tools.py — GET / PUT / RUN / QUOTE
# Silent on failure. Only prints if -v/--verbose is given.

import sys, os, re, socket, subprocess, tempfile
from pathlib import Path
from typing import Optional, List, Tuple

HOST = "127.0.0.1"
PORT = 6502
NULL = b"\x00"

TRANSCRIPT = Path(".transcript.txt")
COUNTER    = Path(".counter")

VERBOSE = False  # set by -v/--verbose

# (Turn N) [ROLE]: <content>\n\n~~~(end)~~~\n\n
TURN_RE = re.compile(
    r"\(Turn\s+(\d+)\)\s*\[([^\]]+)\]:\s(.*?)\n\n~~~\(end\)~~~\n\n",
    re.DOTALL
)
HEAD_RE = re.compile(r"\(Turn\s+(\d+)\)\s*\[([^\]]+)\]:\s", re.DOTALL)

def _silent_exit():
    try: sys.exit(0)
    except SystemExit: raise

def _read_text(p: Path) -> str:
    try: return p.read_text(encoding="utf-8", errors="ignore")
    except Exception: return ""

def _append_text(p: Path, s: str):
    try:
        with p.open("a", encoding="utf-8") as f: f.write(s)
    except Exception:
        _silent_exit()

def _recv_until_null(sock, chunk=4096) -> bytes:
    buf = bytearray()
    while True:
        part = sock.recv(chunk)
        if not part: return b""
        i = part.find(NULL)
        if i != -1:
            buf.extend(part[:i])
            return bytes(buf)
        buf.extend(part)

def _echo_roundtrip(text: str, connect_timeout=3.0, recv_timeout=None) -> str:
    try:
        payload = text.encode("utf-8") + NULL
        with socket.create_connection((HOST, PORT), timeout=connect_timeout) as s:
            s.settimeout(recv_timeout)
            s.sendall(payload)
            reply = _recv_until_null(s)
            if reply and reply[-1] == 0:
                reply = reply[:-1]
        return reply.decode("utf-8", errors="replace")
    except Exception:
        return ""

def _parse_turns(txt: str):
    out = []
    for m in TURN_RE.finditer(txt):
        try:
            out.append((int(m.group(1)), m.group(2), m.group(3)))
        except Exception: continue
    return out

def _highest_turn(txt: str) -> int:
    hi = -1
    for m in HEAD_RE.finditer(txt):
        try:
            n = int(m.group(1))
            if n > hi: hi = n
        except Exception: pass
    return hi

def _ensure_counter():
    if not COUNTER.exists():
        try:
            hi = max(0, _highest_turn(_read_text(TRANSCRIPT)))
            COUNTER.write_text(str(hi), encoding="utf-8")
        except Exception:
            try: COUNTER.write_text("0", encoding="utf-8")
            except Exception: _silent_exit()

def _read_counter() -> int:
    _ensure_counter()
    try:
        s = COUNTER.read_text(encoding="utf-8").strip()
        return int(s or "0")
    except Exception:
        try:
            hi = max(0, _highest_turn(_read_text(TRANSCRIPT)))
            COUNTER.write_text(str(hi), encoding="utf-8")
            s = COUNTER.read_text(encoding="utf-8").strip()
            return int(s or "0")
        except Exception:
            return 0

def _next_turn() -> int:
    n = max(_highest_turn(_read_text(TRANSCRIPT)), _read_counter()) + 1
    try: COUNTER.write_text(str(n), encoding="utf-8")
    except Exception: _silent_exit()
    return n

def _find_last_role(turns, roles: List[str]) -> Optional[Tuple[int,str,str]]:
    best = None
    for n, role, content in turns:
        if role in roles and (best is None or n > best[0]):
            best = (n, role, content)
    return best

def _find_turn_by_id(turns, n: int) -> Optional[Tuple[int,str,str]]:
    for tid, role, content in turns:
        if tid == n: return (tid, role, content)
    return None

def _quote_block(tid: int, role: str, body: str) -> str:
    lines = body.splitlines()
    return "\n".join([f"> (Turn {tid}) [{role}]:"] + ["> " + ln for ln in lines])

# ----------------- GET -----------------
def cmd_GET(argv: List[str]):
    if len(argv) not in (1, 2): _silent_exit()
    txt = _read_text(TRANSCRIPT)
    if not txt: _silent_exit()
    turns = _parse_turns(txt)
    if not turns: _silent_exit()

    if len(argv) == 1:
        out_path = Path(argv[0])
        chosen = _find_last_role(turns, ["FILE"])
    else:
        try: n = int(argv[0])
        except Exception: _silent_exit()
        out_path = Path(argv[1])
        t = _find_turn_by_id(turns, n)
        chosen = t if (t and t[1] == "FILE") else None

    if not chosen: _silent_exit()
    _, _, content = chosen
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
    except Exception:
        _silent_exit()

# ----------------- PUT -----------------
def cmd_PUT(argv: List[str]):
    if len(argv) != 1: _silent_exit()
    path = Path(argv[0])
    try: content = path.read_text(encoding="utf-8")
    except Exception: _silent_exit()

    n = _next_turn()
    if not content[-1] == "\n":
        content+="\n"
    body = f"(Turn {n}) [FILE]: " + content + "\n~~~("
    reply = _echo_roundtrip(body)

    if VERBOSE:
        try:
            sys.stdout.write(reply); sys.stdout.flush()
        except Exception: pass

    _append_text(TRANSCRIPT, body)
    _append_text(TRANSCRIPT, reply)

# ----------------- RUN -----------------
def cmd_RUN(argv: List[str]):
    if len(argv) not in (0, 1): _silent_exit()
    n_req = None
    if len(argv) == 1:
        try: n_req = int(argv[0])
        except Exception: _silent_exit()

    txt = _read_text(TRANSCRIPT)
    if not txt: _silent_exit()
    turns = _parse_turns(txt)
    if not turns: _silent_exit()

    if n_req is None:
        chosen = _find_last_role(turns, ["PYTHON", "BASH"])
    else:
        t = _find_turn_by_id(turns, n_req)
        chosen = t if (t and t[1] in ("PYTHON", "BASH")) else None

    if not chosen: _silent_exit()
    tid, role, content = chosen

    env = os.environ.copy()
    out = ""
    try:
        if role == "BASH":
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".sh") as tf:
                tf.write(content); tf.flush(); sh = tf.name
            try:
                p = subprocess.Popen([ "/bin/bash", sh ],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", cwd=os.getcwd(), env=env)
                out, _ = p.communicate()
                out = out or ""
            finally:
                try: os.unlink(sh)
                except Exception: pass

        elif role == "PYTHON":
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".py") as tf:
                tf.write(content); tf.flush(); py = tf.name
            try:
                p = subprocess.Popen([ sys.executable, "-u", py ],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", cwd=os.getcwd(), env=env)
                out, _ = p.communicate()
                out = out or ""
            finally:
                try: os.unlink(py)
                except Exception: pass
        else:
            _silent_exit(); return
    except Exception:
        _silent_exit(); return

    n = _next_turn()
    if not out[-1] == "\n":
        out+="\n"
    body = f"(Turn {n}) [OUTPUT]: " + out + "\n~~~("
    reply = _echo_roundtrip(body)

    if VERBOSE:
        try:
            sys.stdout.write(reply); sys.stdout.flush()
        except Exception: pass

    _append_text(TRANSCRIPT, body)
    _append_text(TRANSCRIPT, reply)

# ---------------- QUOTE ----------------
def cmd_QUOTE(argv: List[str]):
    if len(argv) != 1: _silent_exit()
    try: qn = int(argv[0])
    except Exception: _silent_exit()

    txt = _read_text(TRANSCRIPT)
    if not txt: _silent_exit()
    turns = _parse_turns(txt)
    if not turns: _silent_exit()
    t = _find_turn_by_id(turns, qn)
    if not t: _silent_exit()

    tid, role, content = t
    quoted = _quote_block(tid, role, content)

    n = _next_turn()
    body = f"(Turn {n}) [QUOTE]: " + quoted + "\n\n~~~("
    reply = _echo_roundtrip(body)

    if VERBOSE:
        try:
            sys.stdout.write(reply); sys.stdout.flush()
        except Exception: pass

    _append_text(TRANSCRIPT, body)
    _append_text(TRANSCRIPT, reply)

# ----------------- Main -----------------
def main():
    global VERBOSE
    argv = sys.argv[1:]
    if not argv: _silent_exit()

    # detect -v/--verbose anywhere
    args = []
    for a in argv:
        if a in ("-v", "--verbose"):
            VERBOSE = True
        else:
            args.append(a)
    if not args: _silent_exit()

    cmd, rest = args[0].upper(), args[1:]

    if cmd == "GET":   cmd_GET(rest)
    elif cmd == "PUT": cmd_PUT(rest)
    elif cmd == "RUN": cmd_RUN(rest)
    elif cmd == "QUOTE": cmd_QUOTE(rest)
    else: _silent_exit()

if __name__ == "__main__":
    main()

