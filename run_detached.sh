#!/usr/bin/env bash
# Run an asura-voiceover step detached from the opencode terminal cgroup and
# memory-capped so systemd-oomd can never take down the agent session.
# Usage: run_detached.sh <chapter_url> [step]   (step default: video)
set -euo pipefail
cd /home/infinite/Projects/asura-voiceover
STEP="${2:-video}"
exec ./.venv/bin/python asura_voiceover.py "$STEP" "$1"
