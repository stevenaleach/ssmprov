# /_/src/Falcon_Mamba_Instruct.py
#
# "ssmprov" Model Runner.
#
# Model: Falcon3-Mamba-7B-Instruct (GGUF, Q8_0)
#
# Protocol & commands identical to your RWKV TCP runner:
#   - plain text → generate
#   - /save, /load, /save_set, /load_set, /t, /p, /k, /pen_freq, /pen_pres, /pen_rep, /min_p, /?
#   - optional leading "!" header: "!checkpoint set next_checkpoint\n<your prompt>"
#
# It re-execs itself with LD_LIBRARY_PATH so llama_cpp hits your local ~/src/llama.cpp build.

import os, sys, pickle, ctypes, json, socket

# ---- paths / build env -------------------------------------------------------
local_build = True  # use llama.cpp in ~/src/llama.cpp/build/bin
BUILD = os.path.expanduser("~/src/llama.cpp/build/bin")    # libllama.so & friends
CUDA_STUBS = "/usr/lib/x86_64-linux-gnu"                   # driver libs (Ubuntu)
MODEL = os.path.expanduser("~/models/falcon_mamba/Falcon3-Mamba-7B-Instruct-q8_0.gguf")

if local_build:
    # Re-exec with env so llama_cpp loads local CUDA libllama.so (and your GPU kernels)
    if "LL_REEXEC" not in os.environ:
        env = {
            "LL_REEXEC": "1",
            "HOME": os.path.expanduser("~"),
            "PATH": "/usr/bin",
            "LD_LIBRARY_PATH": f"{BUILD}:{CUDA_STUBS}",
            "LLAMA_CPP_LIB": os.path.join(BUILD, "libllama.so"),
        }
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

# ---- llama.cpp driver --------------------------------------------------------
from llama_cpp import Llama

# defaults (kept close to your RWKV runner; Mamba tip: min_p ~= 0.05–0.10 often helps)
MAX_CHARS = 4*1024
TEMP      = 0.18
TOP_P     = 0.88
TOP_K     = 0
PEN_FREQ  = 0.00
PEN_PRES  = 0.00
PEN_REP   = 1.00
MIN_P     = 0.12

# stop-marker helpers (same behavior as RWKV runner)
mark1 = "\n~~~("
mark2 = ")~~~\n\n"
mark3 = "end)~~"
FORCE_AFTER  = "end)~~~\n\n"
FORCE_AFTER3 = "~\n\n"

def make_llm() -> Llama:
    return Llama(
        model_path=MODEL,
        n_ctx=64*1024,     # suppress warning; real context governed by model/kv
        n_gpu_layers=999,  # full offload if possible
        n_threads=8,
        verbose=False,
    )

# ---- low-level ctx access (unchanged) ---------------------------------------
def _ctx_ptr(llm: Llama):
    _ctx = getattr(llm, "_ctx", None)
    return getattr(_ctx, "ctx", None) or _ctx

def capture_state_min(llm: Llama) -> dict:
    """Return minimal state: {'blob': bytes, 'n_tokens': int}."""
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
    """Apply minimal state produced by capture_state_min."""
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

# ---- persistence of state & sampling knobs ----------------------------------
def save_state_min(state_obj: dict, path="kv.pkl") -> int:
    if not state_obj:
        raise RuntimeError("No state to save yet. Say something first.")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"blob": state_obj["blob"], "n_tokens": int(state_obj.get("n_tokens", 0))}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return os.path.getsize(path)

def load_state_min(llm: Llama, path="kv.pkl") -> dict:
    with open(path, "rb") as f:
        st = pickle.load(f)
    apply_state_min(llm, st)
    return st

def save_knob_set(path: str, *, temp, top_p, top_k, pen_freq, pen_pres, pen_rep, min_p) -> int:
    data = {
        "temp": float(temp), "top_p": float(top_p), "top_k": int(top_k),
        "pen_freq": float(pen_freq), "pen_pres": float(pen_pres), "pen_rep": float(pen_rep),
        "min_p": float(min_p),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)
    return os.path.getsize(path)

def load_knob_set(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {
        "temp": float(d.get("temp", TEMP)),
        "top_p": float(d.get("top_p", TOP_P)),
        "top_k": int(d.get("top_k", TOP_K)),
        "pen_freq": float(d.get("pen_freq", PEN_FREQ)),
        "pen_pres": float(d.get("pen_pres", PEN_PRES)),
        "pen_rep": float(d.get("pen_rep", PEN_REP)),
        "min_p": float(d.get("min_p", MIN_P)),
    }

# ---- token loop with your stop/force rules ----------------------------------
def gen_until_stop(
    llm: Llama, *,
    max_chars=MAX_CHARS,
    temp=TEMP,
    top_p=TOP_P,
    top_k=TOP_K,
    pen_freq=PEN_FREQ,
    pen_pres=PEN_PRES,
    pen_rep=PEN_REP,
    min_p=MIN_P,
) -> str:
    text = ""
    while True:
        tok_id = llm.sample(
            temp=temp,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,                  # NEW
            frequency_penalty=pen_freq,
            presence_penalty=pen_pres,
            repeat_penalty=pen_rep,
        )
        if tok_id == 0:
            break

        piece = llm.detokenize([tok_id]).decode("utf-8", errors="ignore")
        text += piece

        if len(text) > max_chars:
            break
        if mark2 in text:
            break

        j = text.rfind(mark3)
        if j != -1:
            llm.eval([tok_id])
            after  = text[j + len(mark3):]
            target = FORCE_AFTER3
            m = 0
            while m < len(after) and m < len(target) and after[m] == target[m]:
                m += 1
            missing = target[m:]
            if missing:
                text += missing
                toks = llm.tokenize(missing.encode("utf-8"), add_bos=False)
                if toks:
                    llm.eval(toks)
            break

        i = text.rfind(mark1)
        if i != -1:
            llm.eval([tok_id])
            after  = text[i + len(mark1):]
            target = FORCE_AFTER
            m = 0
            while m < len(after) and m < len(target) and after[m] == target[m]:
                m += 1
            missing = target[m:]
            if missing:
                text += missing
                toks = llm.tokenize(missing.encode("utf-8"), add_bos=False)
                if toks:
                    llm.eval(toks)
            break

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
         pen_rep=PEN_REP,
         min_p=MIN_P):
    if state_obj is not None:
        apply_state_min(llm, state_obj)

    toks = llm.tokenize(user_text.encode("utf-8"), add_bos=False)
    llm.eval(toks)

    reply = gen_until_stop(
        llm,
        max_chars=max_chars,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        pen_freq=pen_freq,
        pen_pres=pen_pres,
        pen_rep=pen_rep,
        min_p=min_p,
    )
    new_state = capture_state_min(llm)
    return reply, new_state

def make_help_text(*, temp, top_p, top_k, pen_freq, pen_pres, pen_rep, min_p):
    return f"""\
# MAMBA TCP Runner — Commands & Tuning Guide
_Falcon-Mamba-Instruct-7B, GGUF_

## Usage
- Type plain text to generate a reply.
- Slash-prefixed commands configure generation parameters.

## Commands
- `/save [file]` — Save KV state (default: kv.pkl)
- `/load [file]` — Load KV state (default: kv.pkl)
- `/save_set [file]` — Save current tuning knobs (default: set.json)
- `/load_set [file]` — Load tuning knobs (default: set.json)
- `/max [n]` — Prints or sets maximum characters for generation.

- `/t [float]` — Set or print temperature. Controls randomness.
- `/p [float]` — Set or print top_p. Nucleus sampling cutoff.
- `/k [int]` — Set or print top_k. Hard cap on candidate tokens.
- `/min_p [float]` — Set or print min_p. Filters tiny tail probabilities.

- `/pen_freq [float]` — Frequency penalty. Reduces repeat *token counts*.
	0.0 to disable.
- `/pen_pres [float]` — Presence penalty. Discourages *seen tokens at all*.
	0.0 to disable.
- `/pen_rep  [float]` — Repeat penalty. Multiplies logits of repeats.
	1.0 to disable.

- `/?` — Show this help plus current settings.

---

## Tuning Advice
- **Temperature (0–2):** Higher = more random. 0.7–1.0 typical; 0.2 for code.
- **Top_p (0–1):** Keeps top tokens with cumulative prob ≤ p. 0.9–0.95 typical.
- **Top_k (1–200):** Hard cap on candidates. 40–80 reasonable; higher = freer.
- **Min_p (0–1):** Drops extremely unlikely tokens. 0.05–0.10 filters junk tails.

- **Repeat_penalty (>1):** 1.05–1.15 discourages repetition.
- **Freq/pres penalties (0–1):** Small values (<0.5) encourage novelty without
  hurting coherence; often left at 0 for code.

---

## Current Settings
```
temp       = {temp:.3f}
top_p      = {top_p:.3f}
top_k      = {top_k:d}
min_p      = {min_p:.3f}
pen_freq   = {pen_freq:.3f}
pen_pres   = {pen_pres:.3f}
pen_rep    = {pen_rep:.3f}
```
"""


def main():
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
                if r and r[-1] == 0:
                    r = r[:-1]
                return r
            buf.extend(data)

    if not os.path.exists(MODEL):
        raise FileNotFoundError(MODEL)

    llm = make_llm()
    state_obj = None
    default_path = "kv.pkl"
    default_set_path = "set.json"
    temp = TEMP; top_p = TOP_P; top_k = TOP_K
    pen_freq = PEN_FREQ; pen_pres = PEN_PRES; pen_rep = PEN_REP
    min_p = MIN_P

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        print(f"[MAMBA TCP] listening on {HOST}:{PORT} (single-connection mode)")

        while True:
            conn, addr = srv.accept()
            try:
                raw = recv_until_null(conn)
                if raw is None:
                    conn.sendall(NULL)
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                pending_post_save = None

                def _maybe_post_save():
                    nonlocal pending_post_save, state_obj
                    if pending_post_save:
                        try:
                            if state_obj is None:
                                state_obj = capture_state_min(llm)
                            save_state_min(state_obj, pending_post_save)
                        except Exception:
                            pass
                        pending_post_save = None

                # Leading "!" header handling (load / load_set / post-save)
                if line.startswith("!"):
                    header, _, body = line.partition("\n")
                    args = header[1:].strip().split()
                    if len(args) >= 1:
                        try:
                            state_obj = load_state_min(llm, args[0])
                        except Exception:
                            pass
                    if len(args) >= 2:
                        try:
                            cfg = load_knob_set(args[1])
                            temp = cfg["temp"]; top_p = cfg["top_p"]; top_k = cfg["top_k"]
                            pen_freq = cfg["pen_freq"]; pen_pres = cfg["pen_pres"]; pen_rep = cfg["pen_rep"]
                            min_p = cfg["min_p"]
                        except Exception:
                            pass
                    if len(args) >= 3:
                        pending_post_save = args[2]
                    line = body

                parts = line.split(maxsplit=1)
                head = parts[0].lower() if parts else ""
                arg = parts[1].strip() if len(parts) > 1 else ""

                # --- commands (identical to RWKV runner + min_p) ---
                if head == "/save":
                    path = arg or default_path
                    try:
                        if state_obj is None:
                            state_obj = capture_state_min(llm)
                        n = save_state_min(state_obj, path)
                        out = f"[saved -> {path} ({n} bytes)]\n"
                    except Exception as e:
                        out = f"[save error] {e}\n"
                    conn.sendall(out.encode("utf-8", "ignore") + NULL); continue

                if head == "/load":
                    path = arg or default_path
                    try:
                        state_obj = load_state_min(llm, path)
                        out = f"[loaded <- {path}]\n"
                    except Exception as e:
                        out = f"[load error] {e}\n"
                    conn.sendall(out.encode("utf-8", "ignore") + NULL); continue

                if head == "/max":
                    if arg:
                        try:
                            global MAX_CHARS
                            MAX_CHARS = max(1, int(arg))
                        except Exception:
                            pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"max = {MAX_CHARS}".encode("utf-8","ignore") + NULL)
                    continue


                if head == "/save_set":
                    path = arg or default_set_path
                    try:
                        n = save_knob_set(path, temp=temp, top_p=top_p, top_k=top_k,
                                          pen_freq=pen_freq, pen_pres=pen_pres, pen_rep=pen_rep,
                                          min_p=min_p)
                        out = f"[saved set -> {path} ({n} bytes)]\n"
                    except Exception as e:
                        out = f"[save_set error] {e}\n"
                    conn.sendall(out.encode("utf-8", "ignore") + NULL); continue

                if head == "/load_set":
                    path = arg or default_set_path
                    try:
                        cfg = load_knob_set(path)
                        temp = cfg["temp"]; top_p = cfg["top_p"]; top_k = cfg["top_k"]
                        pen_freq = cfg["pen_freq"]; pen_pres = cfg["pen_pres"]; pen_rep = cfg["pen_rep"]
                        min_p = cfg["min_p"]
                        out = f"[loaded set <- {path}]\n"
                    except Exception as e:
                        out = f"[load_set error] {e}\n"
                    conn.sendall(out.encode("utf-8", "ignore") + NULL); continue

                if head == "/t":
                    if arg:
                        try: temp = float(arg)
                        except Exception: pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"temp = {temp}".encode("utf-8","ignore") + NULL)
                    continue

                if head == "/k":
                    if arg:
                        try: top_k = int(arg)
                        except Exception: pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"top_k = {top_k}".encode("utf-8","ignore") + NULL)
                    continue

                if head == "/p":
                    if arg:
                        try: top_p = float(arg)
                        except Exception: pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"top_p = {top_p}".encode("utf-8","ignore") + NULL)
                    continue

                if head == "/min_p":
                    if arg:
                        try: min_p = float(arg)
                        except Exception: pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"min_p = {min_p}".encode("utf-8","ignore") + NULL)
                    continue

                if head == "/pen_freq":
                    if arg:
                        try: pen_freq = float(arg)
                        except Exception: pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"pen_freq = {pen_freq}".encode("utf-8","ignore") + NULL)
                    continue

                if head == "/pen_pres":
                    if arg:
                        try: pen_pres = float(arg)
                        except Exception: pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"pen_pres = {pen_pres}".encode("utf-8","ignore") + NULL)
                    continue

                if head == "/pen_rep":
                    if arg:
                        try: pen_rep = float(arg)
                        except Exception: pass
                        conn.sendall(NULL)
                    else:
                        conn.sendall(f"pen_rep = {pen_rep}".encode("utf-8","ignore") + NULL)
                    continue

                if head == "/?":
                    out = make_help_text(temp=temp, top_p=top_p, top_k=top_k,
                                         pen_freq=pen_freq, pen_pres=pen_pres, pen_rep=pen_rep,
                                         min_p=min_p)
                    conn.sendall(out.encode("utf-8","ignore") + NULL)
                    continue

                if head.startswith("/"):
                    conn.sendall(NULL)
                    continue

                # --- normal prompt ---
                try:
                    reply, state_obj = turn(
                        llm, line, state_obj,
                        max_chars=MAX_CHARS,
                        temp=temp, top_p=top_p, top_k=top_k,
                        pen_freq=pen_freq, pen_pres=pen_pres, pen_rep=pen_rep,
                        min_p=min_p,
                    )
                    _maybe_post_save()
                except Exception as e:
                    reply = f"[error] {e}"

                conn.sendall(reply.encode("utf-8", "ignore") + NULL)

            finally:
                conn.close()

if __name__ == "__main__":
    main()
