#!/usr/bin/env bash
# install.sh — sets up ssmprov files + paths for the demo system
# Idempotent, with backups for conflicting files.
set -euo pipefail

# --- Config: repo-local filenames (expected in current directory) ---
WRAPPERS=(COMPOSER MODEL GET PUT QUOTE RUN)
PY_FILES=(COMPOSER.py RWKV7.py tools.py)
S1="s1.txt"
S2="s2.txt"

# --- Helpers ---------------------------------------------------------
backup_if_diff() {
  # backup_if_diff <src> <dst>
  local src="$1" dst="$2"
  if [[ -e "$dst" ]]; then
    if ! cmp -s "$src" "$dst"; then
      cp -f "$dst" "${dst}.bak"
      echo "Backed up: $dst -> ${dst}.bak"
      cp -f "$src" "$dst"
      echo "Updated:   $dst"
    else
      echo "Unchanged: $dst"
    fi
  else
    cp -f "$src" "$dst"
    echo "Installed: $dst"
  fi
}

ensure_path_line() {
  # ensure PATH line in a shell rc file
  local rc="$1"
  local line='export PATH="$HOME/bin:$PATH"'
  if [[ -f "$rc" ]]; then
    if ! grep -Fq 'export PATH="$HOME/bin:$PATH"' "$rc"; then
      printf '\n%s\n' "$line" >> "$rc"
      echo "Appended PATH to $rc"
    else
      echo "PATH already present in $rc"
    fi
  else
    printf '%s\n' "$line" > "$rc"
    echo "Created $rc with PATH entry"
  fi
}

validate_python_string_file() {
  # validate each non-empty line is a valid Python string literal
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "WARN: $file not found; skipping validation."
    return 1
  fi
  python3 - <<PY
import ast, sys
fn = sys.argv[1]
ok = True
with open(fn, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        s = line.rstrip('\n')
        if not s.strip():
            continue
        try:
            v = ast.literal_eval(s)
            if not isinstance(v, str):
                print(f"{fn}:{i}: not a string literal", file=sys.stderr)
                ok = False
        except Exception as e:
            print(f"{fn}:{i}: invalid Python string literal: {e}", file=sys.stderr)
            ok = False
if not ok:
    sys.exit(2)
PY
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "ERROR: $file contains invalid Python string literal(s). Fix it and re-run."
    exit 2
  fi
}

# --- 1) Ensure ~/bin in PATH (.bashrc + .zshrc) ----------------------
mkdir -p "$HOME/bin"
ensure_path_line "$HOME/.bashrc"
# zsh is common; add the same PATH line there as well
ensure_path_line "$HOME/.zshrc"

# --- 2) Install wrappers to ~/bin (with backup if differing) ---------
for w in "${WRAPPERS[@]}"; do
  if [[ ! -f "./$w" ]]; then
    echo "ERROR: Missing wrapper ./$w in current directory."
    exit 1
  fi
  backup_if_diff "./$w" "$HOME/bin/$w"
  chmod +x "$HOME/bin/$w"
done

# --- 3) Create /_ and expected hidden dirs ---------------------------
if [[ ! -d "/_" ]]; then
  echo "Creating /_ (sudo)..."
  sudo mkdir -p "/_"
fi
sudo chown "$(id -un)":"$(id -gn)" "/_"
sudo chmod 0755 "/_"
mkdir -p "/_/.checkpoints" "/_/.assets"

# --- 4) Detect WORKDIR from MODEL wrapper and create it --------------
MODEL_WRAPPER="$HOME/bin/MODEL"
if [[ -f "$MODEL_WRAPPER" ]]; then
  # Grab the WORKDIR="..." line (support single or double quotes)
  WORKDIR=$(awk -F= '/^[[:space:]]*WORKDIR=/{sub(/^[[:space:]]*WORKDIR=/,""); gsub(/^'\''|'\''$|^"|"$/, "", $0); print $0}' "$MODEL_WRAPPER" | head -n1)
  if [[ -n "${WORKDIR:-}" ]]; then
    # If WORKDIR is relative, make it under /_
    case "$WORKDIR" in
      /*) : ;; # absolute, leave as-is
      *) WORKDIR="/_/$WORKDIR" ;;
    esac
    mkdir -p "$WORKDIR"
    echo "Ensured WORKDIR: $WORKDIR"
  else
    echo "WARN: Could not detect WORKDIR in $MODEL_WRAPPER. Skipped."
  fi
else
  echo "WARN: $MODEL_WRAPPER not found; cannot derive WORKDIR."
fi

# --- 5) Copy s1/s2 into /_ as hidden files (validate first) ----------
if [[ -f "$S1" ]]; then
  validate_python_string_file "$S1"
  backup_if_diff "$S1" "/_/.s1.txt"
  backup_if_diff "$S1" "/_/.s3.txt"
else
  echo "WARN: $S1 not found in repo; skipped installing /_/.s1.txt and /_/.s3.txt"
fi

if [[ -f "$S2" ]]; then
  validate_python_string_file "$S2"
  backup_if_diff "$S2" "/_/.s2.txt"
else
  echo "WARN: $S2 not found in repo; skipped installing /_/.s2.txt"
fi

# --- 6) Install Python source files to ~/src/ssmprov -----------------
DEST="$HOME/src/ssmprov"
mkdir -p "$DEST"
for f in "${PY_FILES[@]}"; do
  if [[ ! -f "./$f" ]]; then
    echo "ERROR: Missing Python file ./$f in current directory."
    exit 1
  fi
  backup_if_diff "./$f" "$DEST/$f"
  chmod 0644 "$DEST/$f"
done

# --- 7) Non-fatal environment sanity checks --------------------------
# Check model path from RWKV7.py (heuristic: search for '.gguf' in a MODEL-like variable)
MODEL_PATH=$(grep -Eo '["'\'']([^"'\'']+\.gguf)["'\'']' "$DEST/RWKV7.py" | head -n1 || true)
if [[ -n "${MODEL_PATH:-}" ]]; then
  MP="${MODEL_PATH%\"}"
  MP="${MP#\"}"
  MP="${MP%\'}"
  MP="${MP#\'}"
  # Expand ~ for display
  if [[ "${MP:0:1}" == "~" ]]; then
    MP_EXPANDED="$HOME${MP:1}"
  else
    MP_EXPANDED="$MP"
  fi
  if [[ ! -f "$MP_EXPANDED" ]]; then
    echo "WARN: Expected model file not found: $MP_EXPANDED"
    echo "      Make sure you have converted your model to 8-bit GGUF and placed it here, or update RWKV7.py."
  fi
else
  echo "NOTE: Could not infer model .gguf path from RWKV7.py (this is only a heuristic)."
fi

# Check local llama.cpp build if RWKV7.py sets local_build = True
if grep -Eq 'local_build[[:space:]]*=[[:space:]]*True' "$DEST/RWKV7.py"; then
  LBLIB="$HOME/src/llama.cpp/build/bin/libllama.so"
  if [[ ! -f "$LBLIB" ]]; then
    echo "WARN: local_build=True but libllama.so not found at $LBLIB"
    echo "      Either build llama.cpp in that location, or set local_build=False in RWKV7.py to use the pip package."
  fi
fi

# Ensure /_/.counter exists (initialize to 0 if missing; do NOT create transcript)
if [[ ! -f "/_/.counter" ]]; then
  echo "0" > "/_/.counter"
  echo "Initialized /_/.counter to 0"
fi

# --- 8) Final instructions -------------------------------------------
echo
echo "✅ Install complete."
echo "• PATH updated in ~/.bashrc and ~/.zshrc (open a new shell or: source ~/.bashrc)"
echo "• Wrappers installed in:   \$HOME/bin"
echo "• Python files installed:   $DEST"
echo "• Working dir prepared:     /_"
[[ -n "${WORKDIR:-}" ]] && echo "• Runner WORKDIR ensured:    $WORKDIR"
echo
echo "Next steps:"
echo "  1) Terminal A: MODEL       (wait for 'listening on 127.0.0.1:6502')"
echo "  2) Terminal B: COMPOSER    (paste your primer text; edit .counter if needed)"
echo "  3) Use RUN/PUT/GET/QUOTE from /_ as you iterate"
