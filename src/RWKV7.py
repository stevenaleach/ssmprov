# RWKV_TCP.py — KV-only chat with llama.cpp (GPU), /save + /load + /save_set + /load_set

import os, sys, pickle, ctypes, json

local_build = True  # use llama.cpp in ~/src/llama.cpp/build/bin
BUILD = os.path.expanduser("~/src/llama.cpp/build/bin")  # libllama.so
CUDA_STUBS = "/usr/lib/x86_64-linux-gnu"  # driver libs live here on Ubuntu
MODEL = os.path.expanduser("~/models/rwkv7/rwkv7-7b2-Q8_0.gguf")

if local_build:
    # Re-exec with env so llama_cpp loads local CUDA libllama.so
    if "LL_REEXEC" not in os.environ:
        env = {
            "LL_REEXEC": "1",
            "HOME": os.path.expanduser("~"),
            "PATH": "/usr/bin",
            "LD_LIBRARY_PATH": f"{BUILD}:{CUDA_STUBS}",
            "LLAMA_CPP_LIB": os.path.join(BUILD, "libllama.so"),
        }
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

from llama_cpp import Llama

MAX_CHARS = 8192
TEMP      = 0.70
TOP_P     = 0.95
TOP_K     = 40
PEN_FREQ  = 0.20  # frequency_penalty (0.0 disables; ~0.1–0.5 typical)
PEN_PRES  = 0.10  # presence_penalty  (0.0 disables; ~0.1–0.6 typical)
PEN_REP   = 1.10  # repeat_penalty    (1.0 disables; ~1.05–1.15 typical)

# stop-marker helpers (used by gen_until_stop)
mark1 = "\n~~~("
mark2 = ")~~~\n\n"
mark3 = "end)~~"


FORCE_AFTER = "end)~~~\n\n"
FORCE_AFTER3 = "~\n\n"   # finish "end)~~" -> "~\n\n"

# -------- llama.cpp driving (no history replay) ------------------------------
def make_llm() -> Llama:
    return Llama(
        model_path=MODEL,
        n_ctx=1048576, # <-- Just to get rid of the warning.
        n_gpu_layers=999,
        n_threads=8,
        verbose=False,
    )

# ---- minimal state capture/apply (KV-only on disk) --------------------------
def _ctx_ptr(llm: Llama):
    # llama-cpp-python exposes the raw ctx at llm._ctx.ctx on recent versions
    _ctx = getattr(llm, "_ctx", None)
    return getattr(_ctx, "ctx", None) or _ctx

def capture_state_min(llm: Llama) -> dict:
    """
    Return a minimal state dict:
      { 'blob': <bytes of compact C-state>, 'n_tokens': <int> }
    Prefers Llama.save_state() bytes; falls back to low-level C API.
    """
    ntok = int(getattr(llm, "n_tokens", 0))
    try:
        blob = llm.save_state()
        if isinstance(blob, (bytes, bytearray, memoryview)):
            return {"blob": bytes(blob), "n_tokens": ntok}
    except Exception:
        pass
    from llama_cpp import llama_cpp as C
    ctx = _ctx_ptr(llm)
    if ctx is None:
        raise RuntimeError("Unable to access llama context pointer for state copy.")
    size = int(C.llama_get_state_size(ctx))
    buf = (ctypes.c_uint8 * size)()
    wrote = int(C.llama_copy_state_data(ctx, buf))
    return {"blob": bytes(buf[:wrote]), "n_tokens": ntok}

def apply_state_min(llm: Llama, st: dict):
    """
    Apply a minimal state dict produced by capture_state_min.
    Tries Llama.load_state(bytes) first; falls back to low-level setter.
    """
    blob = st["blob"]
    llm.reset()
    try:
        llm.load_state(blob)
    except Exception:
        from llama_cpp import llama_cpp as C
        ctx = _ctx_ptr(llm)
        if ctx is None:
            raise RuntimeError("Unable to access llama context pointer for state set.")
        Arr = ctypes.c_uint8 * len(blob)
        if int(C.llama_set_state_data(ctx, Arr.from_buffer_copy(blob))) != len(blob):
            raise RuntimeError("llama_set_state_data wrote fewer bytes than expected")
    llm.n_tokens = int(st.get("n_tokens", 0))

# -------- minimal persistence helpers ----------------------------------------
def save_state_min(state_obj: dict, path="kv.pkl") -> int:
    """Serialize only our minimal dict (bytes + n_tokens). No token history."""
    if not state_obj:
        raise RuntimeError("No state to save yet. Say something first.")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(
            {"blob": state_obj["blob"], "n_tokens": int(state_obj.get("n_tokens", 0))},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, path)
    return os.path.getsize(path)

def load_state_min(llm: Llama, path="kv.pkl") -> dict:
    with open(path, "rb") as f:
        st = pickle.load(f)
    apply_state_min(llm, st)
    return st

# -------- sampling set persistence (JSON) ------------------------------------
def save_knob_set(path: str, *, temp, top_p, top_k, pen_freq, pen_pres, pen_rep) -> int:
    """Save current sampling knobs to JSON, return bytes written."""
    data = {
        "temp": float(temp), "top_p": float(top_p), "top_k": int(top_k),
        "pen_freq": float(pen_freq), "pen_pres": float(pen_pres), "pen_rep": float(pen_rep),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)
    return os.path.getsize(path)

def load_knob_set(path: str) -> dict:
    """Load sampling knobs dict from JSON (tolerant casting)."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {
        "temp": float(d.get("temp", TEMP)),
        "top_p": float(d.get("top_p", TOP_P)),
        "top_k": int(d.get("top_k", TOP_K)),
        "pen_freq": float(d.get("pen_freq", PEN_FREQ)),
        "pen_pres": float(d.get("pen_pres", PEN_PRES)),
        "pen_rep": float(d.get("pen_rep", PEN_REP)),
    }

# -----------------------------------------------------------------------------
def gen_until_stop(
    llm: Llama, *,
    max_chars=MAX_CHARS,
    temp=TEMP,
    top_p=TOP_P,
    top_k=TOP_K,
    pen_freq=PEN_FREQ,
    pen_pres=PEN_PRES,
    pen_rep=PEN_REP,
) -> str:
    """
    Sample until:
      1) max_chars, or
      2) mark2 appears (')~~~\\n\\n') — stop (do NOT eval this token), or
      3) mark3 appears ('end)~~') — force-complete with '~\\n\\n' and eval those tokens, or
      4) mark1 appears ('\\n~~~(') — force-complete to 'end)~~~\\n\\n' and eval those tokens.
    """
    text = ""
    while True:
        tok_id = llm.sample(
            temp=temp,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=pen_freq,
            presence_penalty=pen_pres,
            repeat_penalty=pen_rep,
        )
        if tok_id == 0:
            break

        piece = llm.detokenize([tok_id]).decode("utf-8", errors="ignore")
        text += piece

        # 1) hard cap
        if len(text) > max_chars:
            break

        # 2) full close already present (do NOT eval this token)
        if mark2 in text:
            break

        # 3) partial close "end)~~" -> force "~\n\n"
        j = text.rfind(mark3)
        if j != -1:
            # we just sampled tok_id (which may contain mark3); advance KV with it
            llm.eval([tok_id])

            after  = text[j + len(mark3):]
            target = FORCE_AFTER3  # "~\n\n"

            # longest common prefix between 'after' and 'target'
            m = 0
            while m < len(after) and m < len(target) and after[m] == target[m]:
                m += 1

            missing = target[m:]
            if missing:
                text += missing
                toks = llm.tokenize(missing.encode("utf-8"), add_bos=False)
                if toks:
                    llm.eval(toks)  # ensure model state "saw" the forced trailer
            break

        # 4) opener seen -> force to 'end)~~~\n\n'
        i = text.rfind(mark1)
        if i != -1:
            # advance KV with the token we just sampled (it may contain mark1)
            llm.eval([tok_id])

            after  = text[i + len(mark1):]
            target = FORCE_AFTER  # "end)~~~\n\n"

            # longest common prefix between 'after' and 'target'
            m = 0
            while m < len(after) and m < len(target) and after[m] == target[m]:
                m += 1

            missing = target[m:]
            if missing:
                text += missing
                toks = llm.tokenize(missing.encode("utf-8"), add_bos=False)
                if toks:
                    llm.eval(toks)  # ensure model state "saw" the forced trailer
            break

        # normal step: advance KV one token
        llm.eval([tok_id])

    return text

def turn(llm: Llama,
         user_text: str,
         state_obj, *,
         max_chars=MAX_CHARS,
         temp=TEMP,
         top_k=TOP_K,
         top_p=TOP_P,
         pen_freq=PEN_FREQ,
         pen_pres=PEN_PRES,
         pen_rep=PEN_REP):
    if state_obj is not None:
        apply_state_min(llm, state_obj)

    toks = llm.tokenize(user_text.encode("utf-8"), add_bos=False)
    llm.eval(toks)

    reply = gen_until_stop(llm,
                           max_chars=max_chars,
                           temp=temp,
                           top_p=top_p,
                           top_k=top_k,
                           pen_freq=pen_freq,
                           pen_pres=pen_pres,
                           pen_rep=pen_rep)
    new_state = capture_state_min(llm)
    return reply, new_state

def make_help_text(*, temp, top_p, top_k, pen_freq, pen_pres, pen_rep):
    return f"""\
RWKV TCP Runner — commands & tuning

USAGE
  - Send plain text to generate a reply.
  - Slash-prefixed lines are commands (not seen by the model).

PREFIX (Optional, 1st line)

 !load_checkpoint sampling_settings save_checkpoint

    Loads specified checkpoint to process prompt.
    If a second parameter is passed, it is the sampling settings to use.
    The third, if provided is a new post-turn checkpoint name to save to.

COMMANDS
  /save [path]        Save minimal KV state to file (default: kv.pkl)
  /load [path]        Load minimal KV state from file (default: kv.pkl)
  /save_set [path]    Save current sampling knobs to JSON (default: set.json)
  /load_set [path]    Load sampling knobs from JSON (default: set.json)

  /t [float]          Set/print temperature. Omit value to print current.
  /p [float]          Set/print top_p. Omit value to print current.
  /k [int]            Set/print top_k. Omit value to print current.

  /pen_freq [float]   Set/print frequency_penalty (0 disables).
  /pen_pres [float]   Set/print presence_penalty  (0 disables).
  /pen_rep  [float]   Set/print repeat_penalty    (1.0 disables).

  /?                  Show this help plus current settings.

CURRENT SETTINGS
  temp       = {temp:.3f}
  top_p      = {top_p:.3f}
  top_k      = {top_k:d}
  pen_freq   = {pen_freq:.3f}
  pen_pres   = {pen_pres:.3f}
  pen_rep    = {pen_rep:.3f}

TUNING GUIDANCE (RWKV7 via llama.cpp)
  temperature:
    • Typical 0.6–1.0. Lower = safer/more deterministic; higher = more diverse.
  top_p:
    • 0.90–0.98 covers most use-cases. Lower to tighten output; raise for breadth.
  top_k:
    • 20–100 is common. 0 disables (unbounded). Smaller = stricter sampling.
  frequency_penalty (pen_freq):
    • 0.1–0.5 to reduce verbatim repeats; starts nudging away from repeated tokens.
  presence_penalty (pen_pres):
    • 0.1–0.6 to encourage new tokens (one-time hit if token already appeared).
  repeat_penalty (pen_rep):
    • 1.05–1.15 for mild anti-looping. 1.00 disables. Too high → terse/evasive text.

NOTES
  • Penalties apply to the current prompt tokens + tokens generated in this turn.
  • Stop markers: generation halts on ')~~~', or when the opener '\\n~~~(' is seen
    we force-close to 'end)~~~\\n', or when max_chars is reached.
  • State save/load is O(1) to apply. Snapshots contain only compact C-state bytes.
"""

def main():
    import socket

    #HOST = "0.0.0.0"
    HOST = "127.0.0.1"
    PORT = 6502
    NULL = b"\x00"
    CHUNK = 4096

    def recv_until_null(conn):
        buf = bytearray()
        while True:
            data = conn.recv(CHUNK)
            if not data:
                return None  # client closed
            i = data.find(NULL)
            if i != -1:
                buf.extend(data[:i])
                r = bytes(buf)
                if r and r[-1] == 0:  # keep required NULL strip; add guard
                    r = r[:-1]
                return r
            buf.extend(data)

    if not os.path.exists(MODEL):
        raise FileNotFoundError(MODEL)

    llm = make_llm()
    state_obj = None
    default_path = "kv.pkl"
    default_set_path = "set.json"
    temp = TEMP
    top_p = TOP_P
    top_k = TOP_K
    pen_freq = PEN_FREQ
    pen_pres = PEN_PRES
    pen_rep  = PEN_REP

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        print(f"[RWKV7 TCP] listening on {HOST}:{PORT} (single-connection mode)")

        while True:
            conn, addr = srv.accept()
            try:
                raw = recv_until_null(conn)
                if raw is None:
                    conn.sendall(NULL)
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()

                # Support: leading "!" header => pre/post overrides
                pending_post_save = None  # third arg becomes a post-turn save path (if any)

                def _maybe_post_save():
                    """If a post-save path is pending, save current state_obj and clear flag."""
                    nonlocal pending_post_save, state_obj
                    if pending_post_save:
                        try:
                            if state_obj is None:
                                state_obj = capture_state_min(llm)
                            save_state_min(state_obj, pending_post_save)
                        except Exception:
                            # silent by design (match your normal /save UX when scripted)
                            pass
                        pending_post_save = None

                # Early parse of a leading "!" header
                if line.startswith("!"):
                    header, _, body = line.partition("\n")
                    args = header[1:].strip().split()  # up to 3 names

                    # 1) /load <arg1>
                    if len(args) >= 1:
                        try:
                            state_obj = load_state_min(llm, args[0])
                        except Exception:
                            # silent error: keep behavior consistent with scripted flow
                            pass

                    # 2) /load_set <arg2>
                    if len(args) >= 2:
                        try:
                            cfg = load_knob_set(args[1])
                            temp     = cfg["temp"]
                            top_p    = cfg["top_p"]
                            top_k    = cfg["top_k"]
                            pen_freq = cfg["pen_freq"]
                            pen_pres = cfg["pen_pres"]
                            pen_rep  = cfg["pen_rep"]
                        except Exception:
                            pass

                    # 3) remember post /save <arg3>
                    if len(args) >= 3:
                        pending_post_save = args[2]

                    # 4) continue with remaining text as normal (command or prompt)
                    line = body  # (may be empty; then we'll just do the post-save if requested)



                print("LINE: "+'"'+line+'"')

                parts = line.split(maxsplit=1)
                head = parts[0].lower() if parts else ""
                arg = parts[1].strip() if len(parts) > 1 else ""

                # /save (minimal state)
                if head == "/save":
                    path = arg or default_path
                    try:
                        # If we haven't captured any state yet (fresh runner), grab the current
                        # context (which is blank) so we can save it.
                        if state_obj is None:
                            state_obj = capture_state_min(llm)

                        n = save_state_min(state_obj, path)
                        out = f"[saved -> {path} ({n} bytes)]\n"
                    except Exception as e:
                        out = f"[save error] {e}\n"
                    conn.sendall(out.encode("utf-8", errors="ignore") + NULL)
                    continue
                # /load (minimal state)
                if head == "/load":
                    path = arg or default_path
                    try:
                        state_obj = load_state_min(llm, path)
                        out = f"[loaded <- {path}]\n"
                    except Exception as e:
                        out = f"[load error] {e}\n"
                    conn.sendall(out.encode("utf-8", errors="ignore") + NULL)
                    continue

                # /save_set (sampling knobs)
                if head == "/save_set":
                    path = arg or default_set_path
                    try:
                        n = save_knob_set(
                            path,
                            temp=temp, top_p=top_p, top_k=top_k,
                            pen_freq=pen_freq, pen_pres=pen_pres, pen_rep=pen_rep
                        )
                        out = f"[saved set -> {path} ({n} bytes)]\n"
                    except Exception as e:
                        out = f"[save_set error] {e}\n"
                    conn.sendall(out.encode("utf-8", "ignore") + NULL)
                    continue

                # /load_set (sampling knobs)
                if head == "/load_set":
                    path = arg or default_set_path
                    try:
                        cfg = load_knob_set(path)
                        temp     = cfg["temp"]
                        top_p    = cfg["top_p"]
                        top_k    = cfg["top_k"]
                        pen_freq = cfg["pen_freq"]
                        pen_pres = cfg["pen_pres"]
                        pen_rep  = cfg["pen_rep"]
                        out = f"[loaded set <- {path}]\n"
                    except Exception as e:
                        out = f"[load_set error] {e}\n"
                    conn.sendall(out.encode("utf-8", "ignore") + NULL)
                    continue

                # /t (temperature)
                if head == "/t":
                    if arg:
                        try: temp = float(arg)
                        except Exception: pass
                        conn.sendall(NULL)  # empty ack on set
                    else:
                        conn.sendall(f"temp = {temp}".encode("utf-8","ignore") + NULL)
                    continue

                # /k (top_k)
                if head == "/k":
                    if arg:
                        try: top_k = int(arg)
                        except Exception: pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"top_k = {top_k}".encode("utf-8","ignore") + NULL)
                    continue

                # /p (top_p)
                if head == "/p":
                    if arg:
                        try: top_p = float(arg)
                        except Exception: pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"top_p = {top_p}".encode("utf-8","ignore") + NULL)
                    continue

                # /pen_freq (frequency_penalty)
                if head == "/pen_freq":
                    if arg:
                        try: pen_freq = float(arg)
                        except Exception: pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"pen_freq = {pen_freq}".encode("utf-8","ignore") + NULL)
                    continue

                # /pen_pres (presence_penalty)
                if head == "/pen_pres":
                    if arg:
                        try: pen_pres = float(arg)
                        except Exception: pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"pen_pres = {pen_pres}".encode("utf-8","ignore") + NULL)
                    continue

                # /pen_rep (repeat_penalty)
                if head == "/pen_rep":
                    if arg:
                        try: pen_rep = float(arg)
                        except Exception: pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"pen_rep = {pen_rep}".encode("utf-8","ignore") + NULL)
                    continue

                # /? (help + current settings)
                if head == "/?":
                    out = make_help_text(
                        temp=temp,
                        top_p=top_p,
                        top_k=top_k,
                        pen_freq=pen_freq,
                        pen_pres=pen_pres,
                        pen_rep=pen_rep,
                    )
                    conn.sendall(out.encode("utf-8", errors="ignore") + NULL)
                    continue

                # Unrecognized slash-commands: ignore (match CLI)
                if head.startswith("/"):
                    conn.sendall(NULL)
                    continue

                # ---- normal prompt: raw pre-formatted text -> model ----
                try:
                    reply, state_obj = turn(
                        llm,
                        line,
                        state_obj,
                        max_chars=MAX_CHARS,
                        temp=temp,
                        top_p=top_p,
                        top_k=top_k,
                        pen_freq=pen_freq,
                        pen_pres=pen_pres,
                        pen_rep=pen_rep,
                    )
                    # If a post-save was requested via the "!" header, do it now (after generation)
                    _maybe_post_save()

                except Exception as e:
                    reply = f"[error] {e}"

                conn.sendall(reply.encode("utf-8", errors="ignore") + NULL)

            finally:
                conn.close()

if __name__ == "__main__":
    main()
