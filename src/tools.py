#!/usr/bin/env python3
# tools.py — GET / PUT / RUN / QUOTE tools for transcript-driven SSM system
# Fixes:
# - Strict turn parsing: content starts one space after ": " in the header
# - Fence must be EXACTLY "\n\n~~~(end)~~~\n\n" (blank line before and after)
# - Uses the LAST strict fence before the next header (avoids false positives)
# - Does NOT include the fence nor the blank line before it in content
# - Headers must be at start-of-line; won’t match inside code blocks

import sys, os, re, socket, subprocess, tempfile
from pathlib import Path
from typing import Optional, List, Tuple

HOST = "127.0.0.1"
PORT = 6502
NULL = b"\x00"

TRANSCRIPT = Path(".transcript.txt")
COUNTER    = Path(".counter")

VERBOSE = False  # set by -v/--verbose

# ---------- Strict turn parsing (blank-line-delimited fence) ----------
FENCE = "\n\n~~~(end)~~~\n\n"
# Header must begin at start-of-line; content begins exactly after ": " (one space)
HEAD_RE = re.compile(r"(?m)^\(Turn\s+(\d+)\)\s*\[([^\]]+)\]:\s*", re.UNICODE)

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

def _iter_headers(txt: str):
    """Yield (match, turn_no:int, role:str) in document order for true headers."""
    for m in HEAD_RE.finditer(txt):
        try:
            n = int(m.group(1)); role = m.group(2)
        except Exception:
            continue
        yield (m, n, role)

def _parse_turns(txt: str):
    """
    Return list of (turn_no:int, role:str, content:str) where content is the
    exact substring between the header (one char past ': ') and the **last**
    strict fence '\\n\\n~~~(end)~~~\\n\\n' that occurs before the next header (or EOF).
    - The fence itself and the blank line before it are NOT included in content.
    - This avoids truncating content that merely *contains* '~~~(end)~~~' text.
    - Only matches headers at start-of-line.
    """
    out: List[Tuple[int,str,str]] = []
    headers = list(_iter_headers(txt))
    if not headers:
        return out

    for i, (hm, n, role) in enumerate(headers):
        start = hm.end()  # content starts exactly one space after the colon
        span_end = headers[i + 1][0].start() if i + 1 < len(headers) else len(txt)

        # Find the **last** strict fence inside this turn's span
        fence_pos = txt.rfind(FENCE, start, span_end)
        if fence_pos == -1:
            # No strict fence: malformed or still generating; skip to avoid bleed-through
            continue

        content = txt[start:fence_pos]
        # Do NOT strip trailing newline here; we only exclude the required blank line via FENCE
        out.append((n, role, content))
    return out

def _highest_turn(txt: str) -> int:
    hi = -1
    for _, n, _ in _iter_headers(txt):
        if n > hi: hi = n
    return hi

# ---------------- core helpers ----------------
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

# ----------------- GET (GET_FILE) -----------------
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

# ----------------- PUT (PUT_FILE) -----------------
def cmd_PUT(argv: List[str]):
    if len(argv) != 1: _silent_exit()
    path = Path(argv[0])
    try: content = path.read_text(encoding="utf-8")
    except Exception: _silent_exit()

    n = _next_turn()
    if not content.endswith("\n"):
        content += "\n"
    # IN-half divider (no OUT header). Runner will force to full fence.
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
                p = subprocess.Popen(
                    ["/bin/bash", sh],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", cwd=os.getcwd(), env=env
                )
                out, _ = p.communicate()
                out = out or ""
            finally:
                try: os.unlink(sh)
                except Exception: pass

        elif role == "PYTHON":
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".py") as tf:
                tf.write(content); tf.flush(); py = tf.name
            try:
                p = subprocess.Popen(
                    [sys.executable, "-u", py],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", cwd=os.getcwd(), env=env
                )
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
    if not out.endswith("\n"):
        out += "\n"
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
    