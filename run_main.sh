#!/usr/bin/env bash
# TrialAgents: Launch multiple main.py tasks asynchronously in background, with per-arg logging.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MAIN_PY="$SCRIPT_DIR/main.py"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs/auto_logs"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "[ERROR] PYTHON_BIN not found or not executable: $PYTHON_BIN"
	echo "        Set PYTHON_BIN to a valid interpreter path."
	exit 1
fi

run_trial_job() {
	local args="$1"
	local ts="$(date +%Y%m%d)"
	local safe_args="${args// /_}"
	if [[ -z "$safe_args" ]]; then
		safe_args="default"
	fi
	local log_file="$LOG_DIR/trialagents_${ts}_${safe_args}.log"

	# Check if a process with the same args is already running
	if pgrep -f "$PYTHON_BIN $MAIN_PY ${args}" > /dev/null; then
		echo "[SKIP] Already running: $PYTHON_BIN $MAIN_PY ${args}"
		return
	fi

	echo "[SPAWN] $PYTHON_BIN $MAIN_PY ${args}"
	echo "        Log: $log_file"

	nohup bash -c "
		echo \"[START \$(date '+%F %T')] $PYTHON_BIN $MAIN_PY ${args}\"
		$PYTHON_BIN $MAIN_PY ${args}
		rc=\$?
		if [[ \$rc -eq 0 ]]; then
			echo \"[SUCCESS \$(date '+%F %T')] ${args}\"
		else
			echo \"[FAIL \$(date '+%F %T')] rc=\$rc ${args}\"
		fi
	" >> "$log_file" 2>&1 &
}

# =========================
# Run configuration (modify as needed)
# =========================
# DATASET="${DATASET:-NCT03041311}" 
# NCT00003299, NCT03041311, NCT00694382, NCT00041119, NCT00174655, NCT00079274
LLM="${LLM:-deepseek-chat}"
GROUP="${GROUP:-control}"
# experimental, control

declare -a JOBS=()

# Example: parallel multi-task jobs
JOBS+=("--dataset NCT00003299 --llm ${LLM} --group experimental --start_idx 300")
JOBS+=("--dataset NCT00003299 --llm ${LLM} --group control --start_idx 300")
JOBS+=("--dataset NCT00079274 --llm ${LLM} --group experimental --start_idx 300")
JOBS+=("--dataset NCT00079274 --llm ${LLM} --group control --start_idx 300")

for job in "${JOBS[@]}"; do
	run_trial_job "$job"
done

# usage: bash TrialAgents/run_main.sh
