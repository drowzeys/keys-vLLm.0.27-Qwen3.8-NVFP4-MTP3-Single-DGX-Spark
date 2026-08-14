#!/usr/bin/env bash
# Reproduce the speed + concurrency tables. Usage: run_bench.sh <base_url>
python3 "$(dirname "$0")/bench.py" "${1:-http://localhost:8078}" "reproduce"
