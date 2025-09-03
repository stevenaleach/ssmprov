# ssmprov
**"SSM Improv"** – small framework for working with **checkpointed State-Space Model (SSM)** contexts as Unix filters.

- *Always-on* model runner serving any SSM model with save/load checkpoints.
- **Named contexts** become static or dynamic CLI filters via simple Bash wrappers.
- Entire system bootstraps from a single transcript containing source + instructions.

---

## Components

- **Model Runner** – TCP service for an SSM model (e.g., Falcon-Mamba, RWKV, XLSTM).
- **PIPE** – send prompts or slash-commands to runner; supports named checkpoints via bang-lines.
- **tools.py** – provides subcommands:
  - `GET_FILE` – extract FILE turns from transcript
  - `PUT_FILE` – add external files into transcript
  - `RUN` – execute PYTHON/BASH turns, capture OUTPUT
  - `QUOTE` – quote any transcript turn
- **Bash Launchers** – wrappers for runner, PIPE, GET_FILE, PUT_FILE, QUOTE, RUN.
- **primer_cli.txt** – diegetic transcript defining roles, fences, and containing all source files.

---

## Quick Start

```bash
# 1. Prepare working directory
sudo mkdir /_
sudo chown $USER:$USER /_
cd /_

# 2. Copy files from repo (repo has bin/ and src/ just like /_)
cp -r /path/to/repo/bin /_
cp -r /path/to/repo/src /_

# 3. Add /_/bin to PATH in ~/.bashrc
echo 'export PATH="$PATH:/_/bin"' >> ~/.bashrc
source ~/.bashrc

# 4. Ensure dependencies:
# - llama.cpp built
# - llama_cpp installed
# If not using a local llama.cpp build, set local_build=False in the runner script.

# 5. Launch model runner (e.g., Falcon-Mamba)
FALCON_INSTRUCT &

# 6. (Optional) Save a blank checkpoint for reuse
echo "/save blank" | PIPE

# 7. Prime model with all source files & transcript (no roles, raw input)
cat primer_cli.txt | PIPE --counter 12
# The model will automatically close the final turn with ')~~~'

# 7.5 Try the RUN command now, and the last BASH block in the new .transcript.txt
will be run which will re-install all the /_/bin and /_/src files from the transcript.
(which should be the same as those already there), and satisfy the model later since
a proper OUTPUT block will also now have been added to the context reassuring it that
the code was run.

# 8. Save primed checkpoint
echo "/save primed" | PIPE

# 9. Prompting:
echo "Hello, Computer" | PIPE --in USER --out INTERFACE
  ** Note that with the current primer a few example steps without an output role
     to establish how INTERFACE should speak will actually be needed before this will
     respond gracefully.

```

---

## PIPE Command Options

- `--in ROLE / --out ROLE` — Wrap input/output in transcript headers with roles.
- `--bang TEXT` — Prepend raw text before the body (e.g., `--bang '!load settings save'`).
- `--debang` — Omit the --bang text from the transcript while still sending it.
- `--port N` — TCP port (default 6502).

**note**: With no --out role specified, a half end-mark ("\n\n~~~(" ) rather than the full
turn stop mark ("\n\n~~~(end)~~~\n\n") is sent - causing the model and runner to
auto-complete.  This allows for "half-stepping" of input, simply reading in a set of role
turns.  Generation is prompted by specifying an out role which is opened by templating
and the model completes.

All role names are templated as "[name]: ", Including the single space after the colon.
This leaves the model to immediately begin output.  "[name]:\n" was initially tested, but
oddly leads both RWKV and Falcon models to often hallucinate a new, different, role name
instead of beginning output generation while the space remains stable with the models
tested.

### Example

```bash
# Load 'blank', apply 'coder' settings, save as 'session1'
PIPE --bang '!blank coder session1'
```

---

## Model Runner Commands

Send via `PIPE`, prefixing commands with `/`:

| Command         | Purpose                               |
|-----------------|---------------------------------------|
| /save [file]     | Save current model state (default kv.pkl) |
| /load [file]     | Load model state                      |
| /save_set [file] | Save tuning knobs                     |
| /load_set [file] | Load tuning knobs                     |
| /t [val]         | Set/print temperature                 |
| /p [val]         | Set/print top_p                       |
| /k [val]         | Set/print top_k                       |
| /min_p [val]     | Set/print min_p cutoff                |
| /pen_freq [val]  | Set/print frequency penalty           |
| /pen_pres [val]  | Set/print presence penalty            |
| /pen_rep [val]   | Set/print repeat penalty              |
| /?               | Show help and current settings        |


---

## Bang Lines

Bang lines are prepended raw text with format:  
`!load_checkpoint sampling_settings !save_checkpoint`  

- Enter as a single-quoted string at the shell:  
  `PIPE --bang '!load settings save'`
- The first parameter is required.
- If no settings file is given, current sampling settings are used.
- If no save name is given, no post-prompt save is made.

Example: `!live settings live` gives a live session auto-saving after each prompt.

---

## NAME Wrapper Scripts

The `NAME` script can be copied into `/bin` or elsewhere on PATH with the **same name** as any checkpoint.  

It automatically sets `--bang '!NAME settings NAME'`, so:  

```bash
echo "Hello" | NAME --in USER --out INTERFACE

Assuming a checkpoint "Alice":

cp /_/src/NAME /_/bin Alice
echo "Hello, Alice." | Alice --in USER --out INTERFACE

```

- Loads checkpoint `NAME`
- Uses sampling settings `settings`
- Saves back to `NAME` after each call

To make static snapshots (load only, no save), remove the final `name` in the script.

---
