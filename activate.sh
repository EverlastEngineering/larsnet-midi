#!/bin/zsh
# This script must be SOURCED, not executed:
#   source ./activate.sh
#   . ./activate.sh
if [[ $- != *i* ]]; then
  echo "Error: This script must be sourced, not executed."
  echo "Run: source ./activate.sh or . ./activate.sh"
  exit 1
fi

# Initialize conda before using activate
eval "$(conda 'shell.zsh' 'hook' 2>/dev/null)"
conda activate drumtomidi