#!/usr/bin/env bash
# Run the asura-voiceover video build detached from the opencode terminal cgroup,
# memory-capped so systemd-oomd can never take down the agent session.
set -euo pipefail
cd /home/infinite/Projects/asura-voiceover
exec ./.venv/bin/python asura_voiceover.py video \
  "https://asurascans.com/comics/surviving-the-game-as-a-barbarian-6f7fe6eb/chapter/1"
