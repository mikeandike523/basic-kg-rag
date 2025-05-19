#!/usr/bin/env bash

# rerun_until_success.sh
# Usage: ./rerun_until_success.sh <command> [args...]
# Will retry <command>+args until it exits with code 0.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <command> [args...]"
  exit 1
fi

while true; do
  "$@"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "✅ Command succeeded (exit code 0)."
    exit 0
  else
    echo "⚠️  Command failed with exit code $rc. Retrying..."
    # optional: sleep a bit between retries
    # sleep 1
  fi
done
