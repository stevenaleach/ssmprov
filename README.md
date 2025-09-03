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
- **primer.txt** – diegetic transcript defining roles, fences, and containing all source files.  

---

## Quick Start

```bash
# 1. Prepare working directory
sudo mkdir /_
sudo chown $USER:$USER /_
cd /_

# 2. Copy files from repo
# (repo has bin/ and src/ just like /_)
cp -r /path/to/repo/bin /_
cp -r /path/to/repo/src /_

# 3. Add /_/bin to PATH in ~/.bashrc
echo 'export PATH="$PATH:/_/bin"' >> ~/.bashrc
source ~/.bashrc

Ensure you already have llama.cpp built and llama_cpp installed.
If not using a local llama.cpp build, edit local_build=False in the runner script.

# 4. Launch model runner (e.g., Falcon-Mamba)
FALCON_INSTRUCT &


# Optional: keep a blank checkpoint too
echo "/save blank" | PIPE

# 5. Prime model with all source files & transcript
cat prime_clir.txt | PIPE 

  The model should autocomplete the turn closing and print ")~~~"

# 6. Save primed checkpoint
echo "/save primed" | PIPE

## PIPE command:

Options:

--in ROLE / --out ROLE — Wrap input/output in transcript headers with roles.

--bang TEXT — Prepend raw text before the body (e.g.,` --bang  '!checkpoint file headers' `).

--debang — Omit the --bang text from the transcript while still sending it.

--port N — TCP port (default 6502).

## Model Runner Commands:

Runner Slash Commands

These work via PIPE input starting with /.
All assume the active runner supports the standard command set.

Command	Purpose
/save [file]	Save current model state (default: kv.pkl)
/load [file]	Load model state
/save_set [file]	Save tuning knobs (temp, top_p, etc.)
/load_set [file]	Load tuning knobs
/t [val]	Set/print temperature
/p [val]	Set/print top_p
/k [val]	Set/print top_k
/min_p [val]	Set/print min_p cutoff
/pen_freq [val]	Set/print frequency penalty
/pen_pres [val]	Set/print presence penalty
/pen_rep [val]	Set/print repeat penalty
/?	Show help and current settings

## Bang Lines:
Bang lines are literal lines that PIPE prepends to the "prompt" sent, format is "!load_checkpoint sampling_settings !save_checkpoint".
These should be entered in single quotes at the shell (PIPE --bang '!load set save').
Only the first parameter is required, current sampling settings will be used if not provided and no post-prompt save will be made if not provided.
"!live settings live" would result in a live session which updates the checkpoint after each prompt.

## NAME:

The script NAME in src, though not yet tested, can (hopefully) be copied into /_/bin or elsewhere in the executable path given the same filename as any save in the model's checkpoint directory.  It takes care of the --bang line by specifying it's own name as the save and load checkpoint and (hard-coded, edit to change) "coder" (which means you must first save a "coder" sampling settings file) sampling.  That then *should* become a named filter that can be piped through in the shell, with input and output role names and whatever specialist behavior that context is doing. This script can be edited, removing the second usage of $name in it, to produce static snapshots.
