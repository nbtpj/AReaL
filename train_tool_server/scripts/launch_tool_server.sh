#!/usr/bin/env bash
# Launch geo_edit tool servers — one per agent + a router on the main port.
# Auto-bootstraps a local Ray head (with the tool_agent resource) if none is
# already running on RAY_PORT.
#
# Usage:
#   bash launch_tool_server.sh                                          # all agents, auto-start ray
#   bash launch_tool_server.sh geo_edit_function geo_chartr1            # specific agents
#   PORT=30888 bash launch_tool_server.sh geo_edit_function geo_chartr1
#   SKIP_RAY_START=1 bash launch_tool_server.sh                         # join external ray (multi-node)
#
# Env overrides:
#   PORT            router port                              (default 30888)
#   LOG_DIR         log directory                            (default tool-server-logs)
#   HOST            router host                              (default 0.0.0.0)
#   RAY_PORT        Ray head GCS port                        (default 6379)
#   NUM_GPUS        GPUs exposed to Ray + tool_agent count   (default 8)
#   SKIP_RAY_START  1 to skip auto-starting Ray              (default 0; useful when
#                   the head lives on another node — caller is responsible for
#                   `ray start --address=<head-ip>:6379` BEFORE this script)
#   PEDIA_MODEL     model root containing PaddleOCR-VL-1.5/, sam3.1/, grounding-dino-base/
#                   (default ./pedia_model — read by paddleocr/sam3/grounding_dino agents).
#                   Example for out-of-repo models:
#                       PEDIA_MODEL=/data/pedia_model bash launch_tool_server.sh
#                   Because this script starts Ray itself, the env auto-propagates
#                   from the bash invocation → ray head → ray actor workers.
#                   (If you set SKIP_RAY_START=1 you MUST also set PEDIA_MODEL
#                   on `ray start` upstream — otherwise actors silently fall
#                   back to ./pedia_model.)
#
# Monitor: tail -f tool-server-logs/*.log
# Stop:    pkill -f "python3 -m train_tool_server\.server" ; \
#          pkill -f "train_tool_server\.router" ; \
#          ray stop --force
set -x

LOG_DIR="${LOG_DIR:-tool-server-logs}"
mkdir -p "$LOG_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30888}"
RAY_PORT="${RAY_PORT:-6379}"
NUM_GPUS="${NUM_GPUS:-8}"
SKIP_RAY_START="${SKIP_RAY_START:-0}"

unset ROCR_VISIBLE_DEVICES

if [ "$SKIP_RAY_START" != "1" ]; then
    if ray status --address="127.0.0.1:${RAY_PORT}" >/dev/null 2>&1; then
        echo "[ray] cluster already up on :${RAY_PORT}, reusing"
    else
        echo "[ray] starting head — port=${RAY_PORT} num-gpus=${NUM_GPUS} tool_agent=${NUM_GPUS}"
        ray start --head --port="${RAY_PORT}" --num-gpus="${NUM_GPUS}" \
            --resources="{\"tool_agent\":${NUM_GPUS}}" || {
            echo "[ray] start FAILED"; exit 1
        }
        sleep 2
    fi
else
    echo "[ray] SKIP_RAY_START=1 — assuming external head reachable; caller's responsibility"
fi

# Agent list: from arguments, or default all
if [ $# -gt 0 ]; then
    AGENT_LIST=("$@")
else
    AGENT_LIST=(
        geo_edit_function
        geo_paddleocr
        geo_sam3
        # geo_chartr1
        geo_grounding_dino
    )
fi

# Kill old processes (scope to the python entrypoints so we don't kill our own
# invoking shell if its argv happens to contain the string "train_tool_server").
pkill -9 -f "python3 -m train_tool_server\.server" 2>/dev/null || true
pkill -9 -f "train_tool_server\.router" 2>/dev/null || true
fuser -k -n tcp "$PORT" 2>/dev/null || true
sleep 2

echo "Launching ${#AGENT_LIST[@]} tool servers..."

# Start each agent on its own port
BACKEND_PORT=$((PORT + 1))
BACKEND_URLS=""
TOOL_TYPES=""

for agent in "${AGENT_LIST[@]}"; do
    fuser -k -n tcp "$BACKEND_PORT" 2>/dev/null || true
    python3 -m train_tool_server.server \
        --tool_type "$agent" \
        --host 127.0.0.1 --port "$BACKEND_PORT" \
        --workers_per_tool 8 --max_concurrent_requests 128 --use_ray True \
        > "$LOG_DIR/${agent}.log" 2>&1 &
    echo "$agent  port=$BACKEND_PORT  pid=$!"
    BACKEND_URLS="${BACKEND_URLS:+$BACKEND_URLS,}\"http://127.0.0.1:$BACKEND_PORT\""
    TOOL_TYPES="${TOOL_TYPES:+$TOOL_TYPES,}\"$agent\""
    BACKEND_PORT=$((BACKEND_PORT + 1))
done

# Start router on the main port, forwarding to all backends
sleep 5
export VT_WORKER_BASE_URLS="[$BACKEND_URLS]"
export VT_WORKER_TOOL_TYPES="[$TOOL_TYPES]"
python3 -c "
import uvicorn
from train_tool_server.router import router_factory
uvicorn.run(router_factory(), host='$HOST', port=$PORT, log_level='info', access_log=False)
" > "$LOG_DIR/router.log" 2>&1 &
echo "router     port=$PORT  pid=$!"

echo ""
echo "tool_server_url=http://$HOST:$PORT/get_observation"
echo "Logs: tail -f $LOG_DIR/*.log"
echo "Stop: pkill -f 'train_tool_server'"
