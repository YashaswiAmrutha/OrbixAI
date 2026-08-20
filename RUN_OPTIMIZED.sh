#!/bin/zsh
set -e

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR/backend"

# Network calls remain bounded so a disconnected external provider cannot hang
# forever, but local model generation has no artificial response deadline.
export ORBIX_SERPAPI_TIMEOUT_S="${ORBIX_SERPAPI_TIMEOUT_S:-12}"
export ORBIX_GOOGLE_TIMEOUT_S="${ORBIX_GOOGLE_TIMEOUT_S:-15}"
export ORBIX_AGENT_MAX_STEPS="${ORBIX_AGENT_MAX_STEPS:-4}"
export ORBIX_AGENT_NUM_PREDICT="${ORBIX_AGENT_NUM_PREDICT:-1024}"
export ORBIX_AGENT_NUM_CTX="${ORBIX_AGENT_NUM_CTX:-4096}"
export ORBIX_AGENT_KEEP_ALIVE="${ORBIX_AGENT_KEEP_ALIVE:--1}"

echo "Starting OrbixAI optimized build on http://127.0.0.1:8002"
echo "Verify with: curl http://127.0.0.1:8002/health"
exec python -m uvicorn main:app --reload --port 8002
