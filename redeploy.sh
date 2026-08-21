#!/usr/bin/env bash
# Kill any running frontend_agent.py Streamlit process and start a fresh one.
# Needed because Streamlit only auto-reruns the main script on save — it does
# NOT reload imported local modules like chatbot/backend_agent.py, so a stale
# process keeps running old code (and old in-memory checkpoint state) after edits.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PATTERN="streamlit run chatbot/frontend_agent.py"
LOG_FILE="streamlit.log"

if pgrep -f "$PATTERN" > /dev/null; then
    echo "Stopping existing Streamlit process..."
    pkill -f "$PATTERN"
    for _ in $(seq 1 10); do
        pgrep -f "$PATTERN" > /dev/null || break
        sleep 0.5
    done
fi

echo "Starting Streamlit..."
nohup .venv/bin/streamlit run chatbot/frontend_agent.py > "$LOG_FILE" 2>&1 &
disown

sleep 1
NEW_PID=$(pgrep -f "$PATTERN" | head -1)
echo "Streamlit running (PID $NEW_PID). Logs: $LOG_FILE"
