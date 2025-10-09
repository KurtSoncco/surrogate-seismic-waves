#!/bin/bash
#
# ==============================================================================
#!/bin/bash
#
# ==================================================================================================
# CONCISE ABLATION STUDY SCRIPT (Linux/GNU sed & awk)
# Executes all experiments sequentially from the project root directory.
# ==================================================================================================

set -euo pipefail

# --- PATH & INITIAL SETUP ---
#PROJECT_ROOT="/global/home/users/kurtwal98/surrogate-seismic-waves"
PROJECT_ROOT="/home/kurt-asus/surrogate-seismic-waves"

# Move to the project root immediately. All paths below are now relative to it.
cd "$PROJECT_ROOT" || { echo "ERROR: Cannot change directory to $PROJECT_ROOT" 1>&2; exit 1; }

# Define paths relative to the current working directory ($PROJECT_ROOT)
CONFIG_FILE="wave_surrogate/models/fno/config.py"
PYTHON_SCRIPT_PATH="wave_surrogate/models/fno/main.py"
VENV_ACTIVATE_SCRIPT=".venv/bin/activate"

# Export PYTHONPATH once for the entire script
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# --- ORIGINAL VALUES (MUST MATCH config.py) ---
ORIG_FNO_MODES=16
ORIG_FNO_WIDTH=50
ORIG_NUM_FNO_LAYERS=3
ORIG_ENCODER_CHANNELS_LIST='[1, 32, 64, 128, 256, ]'
ORIG_LATENT_DIM=1000
ORIG_ENCODER_KERNEL_SIZE=3
ORIG_RUN_NAME="fno-refactored-run"


# --- HELPER 1: Safely rewrite a config variable, preserving multi-line Python values ---
rewrite_config_var() {
    local var="$1"
    local new_value="$2"
    local file="$3"
    local backup="${file}.bak.$(date +%s)"

    cp "$file" "$backup"

    python3 - "$var" "$new_value" "$file" "$backup" <<'PY'
import sys, re, os

var, newv, file, backup = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Try to find the line where the assignment to var starts.
assign_re = re.compile(r'^\s*' + re.escape(var) + r'\s*(?:\:\s*[^=]+)?\s*=\s*')
start = None
for i, L in enumerate(lines):
    if assign_re.match(L):
        start = i
        break

if start is None:
    # Not found
    print("VAR_NOT_FOUND", file=sys.stderr)
    sys.exit(2)

# From the start line, determine if the assigned value begins with '[' (multi-line list)
start_line = lines[start]
after_eq = start_line.split('=', 1)[1] if '=' in start_line else ''
if '[' in after_eq:
    # Multi-line bracketed value - find matching closing bracket
    depth = 0
    end = None
    for j in range(start, len(lines)):
        for ch in lines[j]:
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is not None:
            break
    if end is None:
        print("NO_CLOSING_BRACKET", file=sys.stderr)
        sys.exit(3)
    # Build new content: keep same prefix (up to '=') and insert new value exactly as provided
    prefix_match = re.match(r'^(\s*' + re.escape(var) + r'\s*(?:\:\s*[^=]+)?\s*=\s*)', lines[start])
    prefix = prefix_match.group(1) if prefix_match else var + " = "
    # Ensure newv ends with newline
    new_text = prefix + newv + "\n"
    new_lines = [new_text]
    out_lines = lines[:start] + new_lines + lines[end+1:]
else:
    # Single-line assignment: replace that line preserving indentation/prefix
    prefix_match = re.match(r'^(\s*' + re.escape(var) + r'\s*(?:\:\s*[^=]+)?\s*=\s*)', start_line)
    prefix = prefix_match.group(1) if prefix_match else var + " = "
    out_lines = lines[:start] + [prefix + newv + "\n"] + lines[start+1:]

# Write temporary then atomically replace
tmp = file + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.writelines(out_lines)

os.replace(tmp, file)
sys.exit(0)
PY

    # python returns non-zero on failure — restore backup if that happens
    if [ $? -ne 0 ]; then
        echo "Error: rewrite_config_var failed for $var; restoring backup." >&2
        mv "$backup" "$file" 2>/dev/null || true
        rm -f "${file}.tmp" 2>/dev/null || true
        return 1
    fi

    rm -f "$backup"
    return 0
}


# --- HELPER 2: Set WANDB_RUN_NAME (string literal) ---
set_wandb_run_name() {
    local file="$1"
    local new_name="$2"
    rewrite_config_var "WANDB_RUN_NAME" "\"$new_name\"" "$file"
}


# --- EXECUTION FUNCTION: Updates config, runs script, reverts (SIMPLIFIED PATHS) ---
run_ablation() {
    local config_var=$1
    local new_value=$2
    local experiment_name=$3
    local original_value=$4
    local config_file=$5
    local pre_exp_backup="${config_file}.pre.${experiment_name}.$(date +%s)"

    echo "--- Starting Ablation: $experiment_name ($config_var = $new_value) ---"

    cp "$config_file" "$pre_exp_backup"

    # 1. Update WANDB_RUN_NAME and Ablation Change
    set_wandb_run_name "$config_file" "$experiment_name" || echo "Warning: Failed to set WANDB_RUN_NAME."
    if ! rewrite_config_var "$config_var" "$new_value" "$config_file"; then
        echo "Error: Failed to update $config_var. Restoring backup and skipping." 1>&2
        mv "$pre_exp_backup" "$config_file" 2>/dev/null || true
        return 1
    fi

    # 2. Execute Pipeline
    echo "Executing experiment..."
    # Use a clearly quoted command to avoid quoting pitfalls.
    local inner_cmd="source \"$VENV_ACTIVATE_SCRIPT\" && python -u \"$PYTHON_SCRIPT_PATH\""

    echo "srun not found - running directly."
    bash -lc "$inner_cmd"

    # 3. Revert config changes (best effort)
    echo "Reverting config changes..."
    rewrite_config_var "$config_var" "$original_value" "$config_file" || echo "Warning: Failed to revert $config_var."
    set_wandb_run_name "$config_file" "$ORIG_RUN_NAME" || echo "Warning: Failed to restore WANDB_RUN_NAME."

    rm -f "$pre_exp_backup"

    echo "--- Completed Ablation: $experiment_name ---"
}


# ==================================================================================================
# ABLATION EXPERIMENTS START HERE
# ==================================================================================================
# BASELINE RUN - Run the original configuration first
#run_ablation "FNO_MODES" "$ORIG_FNO_MODES" "A0-Baseline" "$ORIG_FNO_MODES" "$CONFIG_FILE"

# 1. FNO ARCHITECTURE ABLATION
#run_ablation "FNO_MODES" "8" "A1-Modes-Low-8" "$ORIG_FNO_MODES" "$CONFIG_FILE"
#run_ablation "FNO_MODES" "32" "A2-Modes-High-32" "$ORIG_FNO_MODES" "$CONFIG_FILE"
#run_ablation "FNO_WIDTH" "25" "A3-Width-Narrow-25" "$ORIG_FNO_WIDTH" "$CONFIG_FILE"
#run_ablation "FNO_WIDTH" "100" "A4-Width-Wide-100" "$ORIG_FNO_WIDTH" "$CONFIG_FILE"
#run_ablation "NUM_FNO_LAYERS" "1" "A5-Layers-1" "$ORIG_NUM_FNO_LAYERS" "$CONFIG_FILE"
#run_ablation "NUM_FNO_LAYERS" "5" "A6-Layers-5" "$ORIG_NUM_FNO_LAYERS" "$CONFIG_FILE"


# 2. ENCODER ABLATION
#run_ablation "ENCODER_CHANNELS" "[1, 32, 64, 128, 256, 512,]" "B1-Encoder-Deeper" "$ORIG_ENCODER_CHANNELS_LIST" "$CONFIG_FILE"
#run_ablation "ENCODER_CHANNELS" "[1, 64, 256]" "B2-Encoder-Shallower" "$ORIG_ENCODER_CHANNELS_LIST" "$CONFIG_FILE"
#run_ablation "LATENT_DIM" "2000" "B3-Latent-Dim-Large-2000" "$ORIG_LATENT_DIM" "$CONFIG_FILE"
run_ablation "LATENT_DIM" "500" "B4-Latent-Dim-Small-500" "$ORIG_LATENT_DIM" "$CONFIG_FILE"
#run_ablation "ENCODER_KERNEL_SIZE" "5" "B5-Encoder-Kernel-5" "$ORIG_ENCODER_KERNEL_SIZE" "$CONFIG_FILE"

echo "--- All Ablation Runs Complete ---"