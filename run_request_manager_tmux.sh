#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${1:-tpu-request-manager}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is not installed." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" -n manager \
  "cd '${SCRIPT_DIR}' && python3 request_manager.py loop --refresh-if-stale"

echo "tmux session '${SESSION_NAME}' created."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
