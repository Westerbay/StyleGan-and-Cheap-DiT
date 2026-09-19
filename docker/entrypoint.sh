#!/usr/bin/env bash
set -Eeuo pipefail

cd /app/stylegan-and-cheap-dit
python api_ldm.py &
api_pid=$!
cd /app/ui-for-generative-models/application
npm run start &
ui_pid=$!

cleanup() {
    kill "$api_pid" "$ui_pid" 2>/dev/null || true
    wait "$api_pid" "$ui_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
wait -n "$api_pid" "$ui_pid"
